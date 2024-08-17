import json
import time
from pprint import pprint
from typing import Tuple
from dataclasses import dataclass, field, asdict
from functools import partial
import os
import logging
import sys

import numpy as np
import tiktoken
from tqdm import tqdm

import wandb
import tyro

import jax
import jax.numpy as jnp
import flax
from flax.training import checkpoints
from flax.training.train_state import TrainState
from flax.jax_utils import replicate, unreplicate
import optax
import tensorflow as tf
import transformers

from model import GPT, GPTConfig
from dataset import get_dataset
from optimizers.psgd_affine import affine


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


@dataclass(frozen=True)
class WandbConfig:
    """
    wandb logging configuration
    """

    entity: str = "evanatyourservice"
    """username or team name where you're sending runs"""
    project: str = "owt"
    """project name"""
    name: str = ""
    """experiment name"""
    mode: str = "online"
    """'offline', 'online', or 'disabled'"""
    notes: str = ""


@dataclass(frozen=True)
class CosineDecayScheduleConfig:
    init_value: float = 0.0
    peak_value: float = 0.001
    warmup_steps: int = 1000
    decay_steps: int = 50000
    end_value: float = 1e-5


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    out_dir: str = (
        "/Users/evanwalters/gpt_testing"  # output directory for checkpoints (can be gcs path)
    )
    train_pattern: str = (
        "/Users/evanwalters/owt_10k_data/train_??.tfrecord"  # training files glob pattern (can be gcs path)
    )
    val_pattern: str = (
        "/Users/evanwalters/owt_10k_data/val_??.tfrecord"  # validation files glob pattern (can be gcs path)
    )
    shuffle_buffer_size: int = 128
    eval_interval: int = 500
    eval_steps: int = 16  # evaluate for this number of steps (per-device)
    hs_eval_steps: int = 16  # evaluate for this number of steps (per-device)
    eval_only: bool = False  # if True, script exits right after the first eval
    keep_checkpoints: int = 0  # number of historical checkpoints to keep
    batch_size: int = 16  # per-device batch size
    train_steps: int = 50000  # total number of training iterations
    weight_decay: float = 1e-2  # not applied to bias and embedding parameters
    grad_clip: float = 1.0  # gradient norm clipping magnitude
    gradient_accumulation_steps: int = 1  # used to simulate larger batch sizes
    betas: Tuple[float, float] = (0.9, 0.95)  # adamw optimizer betas
    learning_rate: CosineDecayScheduleConfig = field(
        default_factory=CosineDecayScheduleConfig
    )
    wandb: WandbConfig = field(default_factory=WandbConfig)  # wandb logging
    model: GPTConfig = field(default_factory=GPTConfig)  # gpt model config
    remat: bool = False  # set to True to rematerialize gradients during backward pass


@partial(jax.pmap, axis_name="batch", donate_argnums=(0,))
def train_step(
    state: TrainState, tokens: jnp.ndarray, dropout_key
) -> Tuple[jnp.ndarray, TrainState]:

    dropout_key = jax.random.fold_in(dropout_key, state.step)

    def loss_fn(params) -> jnp.ndarray:
        X, Y = tokens[:, :-1], tokens[:, 1:]
        logits = state.apply_fn(X, params=params, dropout_rng=dropout_key, train=True)[
            0
        ]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y).mean()

        # palm style z loss
        loss += 1e-4 * jax.scipy.special.logsumexp(logits, axis=-1).mean() ** 2

        return loss

    # per-device loss and grads
    loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(state.params)
    # loss, grads = jax.value_and_grad(loss_fn, has_aux=False, reduce_axes=('batch',))(state.params)
    # average gradients across devices
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state


@partial(jax.pmap, axis_name="batch")
def eval_step(state: TrainState, tokens: jnp.ndarray) -> jnp.ndarray:
    X, Y = tokens[:, :-1], tokens[:, 1:]
    logits = state.apply_fn(X, params=state.params, train=False)[0]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y).mean()
    loss = jax.lax.pmean(loss, axis_name="batch")
    return loss


@partial(jax.pmap, axis_name="batch")
def eval_step_raw(state: TrainState, tokens: jnp.ndarray) -> jnp.ndarray:
    X, Y = tokens[:, :-1], tokens[:, 1:]
    logits = state.apply_fn(X, params=state.params, train=False)[0]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y)
    return loss


def prepare_hellaswag(config):
    """Read file and tokenize the hellaswag dataset."""
    print("preparing hellaswag")
    enc = tiktoken.get_encoding("gpt2")

    all_data = []
    all_labels = []
    all_lengths = []
    with open("data/hellaswag_val.jsonl", "r") as f:
        # iterate over lines and tokenize
        for line in tqdm(f, total=10042):
            item = json.loads(line)
            context = item["ctx"]
            endings = item["endings"]
            correct_end = item["label"]
            to_concat = []
            lens = []
            for ending in endings:
                input_text = context + " " + ending
                input_ids = enc.encode_ordinary(input_text)
                if len(input_ids) > config.model.block_size:
                    continue
                lens.append(len(input_ids))
                input_ids = np.pad(
                    input_ids, (0, config.model.block_size - len(input_ids))
                )
                to_concat.append(input_ids)
            all_data.append(np.array(to_concat))
            all_labels.append(correct_end)
            all_lengths.append(np.array(lens))

    # cut lists to divisible by n devices
    n_devices = jax.device_count()
    n_data = len(all_data)
    n_data -= n_data % n_devices
    all_data = all_data[:n_data]
    all_labels = all_labels[:n_data]
    all_lengths = all_lengths[:n_data]

    # batch to shape (-1, n_devices, 4, block_size)
    all_data = np.array(all_data).reshape(-1, n_devices, 4, config.model.block_size)
    all_labels = np.array(all_labels).reshape(-1, n_devices)
    all_lengths = np.array(all_lengths).reshape(-1, n_devices, 4)

    print(f"all_data shape: {all_data.shape}")
    return all_data, all_labels, all_lengths


def eval_hellaswag(config, state, data, labels, lengths):
    """Evaluate the hellaswag dataset."""
    correct = 0
    total = 0
    # select n random examples
    n = config.hs_eval_steps
    data = data[np.random.choice(data.shape[0], n, replace=False)]
    labels = labels[np.random.choice(labels.shape[0], n, replace=False)]
    lengths = lengths[np.random.choice(lengths.shape[0], n, replace=False)]
    for i in range(n):
        batch = data[i]  # (n_devices, 4, block_size)
        batch_labels = labels[i]  # (n_devices,)
        batch_lengths = lengths[i]  # (n_devices, 4)
        losses = eval_step_raw(state, batch)  # (n_devices, 4, block_size)
        losses = np.array(jax.device_get(losses))
        for loss_i, label_i, length_i in zip(losses, batch_labels, batch_lengths):
            losses = [l[:length] for l, length in zip(loss_i, length_i)]  # (4, length)
            losses = [l.mean() for l in losses]  # (4,)
            predicted_end = np.argmin(losses)
            if predicted_end == label_i:
                correct += 1
            total += 1
    return correct / total


def count_params(params) -> int:
    p = jax.tree_util.tree_map(
        lambda a: a.size if isinstance(a, jnp.ndarray) else 0, params
    )
    return jax.tree_util.tree_reduce(lambda a, b: a + b, p)


def param_decay_mask(params):
    """pytree mask for non-bias parameters"""
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_param_mask = {
        k: k[-1] not in ("bias", "embedding", "scale") for k in flat_params.keys()
    }
    return flax.traverse_util.unflatten_dict(flat_param_mask)


def init_train_state(key, config: TrainConfig, learning_rate) -> TrainState:

    # if config.remat:
    #     model = flax.linen.remat(
    #         GPT,
    #         static_argnums=(2,),
    #         policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    #     )(config.model)
    # else:
    #     model = GPT(config.model)
    model_config = transformers.GPT2Config(
        n_embd=config.model.num_embeds,
        n_layer=config.model.num_layers,
        n_head=config.model.num_heads,
        use_cache=False,
        initializer_range=0.01,
        scale_attn_by_inverse_layer_idx=True,
        reorder_and_upcast_attn=True,
    )

    model = transformers.FlaxAutoModelForCausalLM.from_config(model_config)
    params = model.params
    # pprint(params)

    optimizer = optax.chain(
        # Apply weight decay only to non-bias parameters
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(
            learning_rate,
            *config.betas,
            weight_decay=config.weight_decay,
            mask=param_decay_mask,
        ),
        # affine(
        #     learning_rate=learning_rate,
        #     preconditioner_update_probability=1.0,
        #     b1=config.betas[0],
        #     b2=config.betas[1],
        #     nesterov=True,
        #     update_global_norm_clip=None,
        #     update_elementwise_clip=False,
        #     weight_decay=config.weight_decay,
        #     mask=param_decay_mask,
        #     max_size_triangular=0,
        #     max_skew_triangular=0,
        #     step_normalizer_order="2nd",
        #     precond_lr=0.01,
        #     precond_init_scale=1.0,
        #     seed=None,
        #     mu_dtype=jnp.bfloat16,
        #     precision="tensorfloat32",
        # ),
        optax.apply_every(config.gradient_accumulation_steps),
    )

    train_state = TrainState.create(
        apply_fn=model.__call__, params=params, tx=optimizer
    )

    return train_state


def get_default_config() -> TrainConfig:
    # use this file to set default values
    path = os.environ.get("GPT_CONFIG", os.path.join("config", "gpt2.yaml"))
    if not os.path.exists(path):
        return TrainConfig()
    logging.info(f"using config file at {path}")
    with open(path, "r") as f:
        return tyro.from_yaml(TrainConfig, f)


if __name__ == "__main__":
    config = tyro.cli(TrainConfig, default=get_default_config())

    if config.wandb is not None and jax.process_index() == 0:
        wandb.init(**asdict(config.wandb))
        wandb.config.update(asdict(config))

    block_size = config.model.block_size

    # ===== datasets =====
    train_ds = get_dataset(
        config.train_pattern,
        config.batch_size,
        block_size,
        config.shuffle_buffer_size,
        seed=config.seed,
    )

    val_ds = get_dataset(config.val_pattern, config.batch_size, block_size)

    # =====  init parameters ============
    key = jax.random.PRNGKey(config.seed)
    key, key_params, key_dropout = jax.random.split(key, 3)
    # make sure dropout keys are different for each device (local and global)
    key_dropout = jax.random.fold_in(key_dropout, jax.process_index())
    keys_dropout = jax.random.split(key_dropout, jax.local_device_count())

    # ===== learning rate schedule =====
    learning_rate = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                config.learning_rate.init_value,
                config.learning_rate.peak_value,
                config.learning_rate.warmup_steps,
            ),
            optax.linear_schedule(
                config.learning_rate.peak_value,
                config.learning_rate.end_value,
                config.learning_rate.decay_steps - config.learning_rate.warmup_steps,
            ),
        ],
        boundaries=[config.learning_rate.warmup_steps],
    )

    train_state = init_train_state(key_params, config, learning_rate)

    num_params = count_params(train_state.params)
    if jax.process_index() == 0:
        # logging.info(f'PARAMETER COUNT: {num_params:,}')
        print(f"PARAMETER COUNT: {num_params:,}")

    best_val_loss = float("inf")

    # ==== restore dataset and train state ==== #
    # restore unreplicated optimizer + model state from last checkpoint.
    # this is a no-op if no checkpoints exist
    if config.keep_checkpoints > 0:
        train_state = checkpoints.restore_checkpoint(
            f"{config.out_dir}/checkpoints/train_state", train_state
        )

    # grab step from last checkpoint
    step = int(train_state.step)

    train_iter = iter(train_ds)
    # We need to be able to save the dataset state for stopping and resuming training
    # we'll save a dataset checkpoint for each shard
    if config.keep_checkpoints > 0:
        dataset_manager = tf.train.CheckpointManager(
            tf.train.Checkpoint(iterator=train_iter),
            f"{config.out_dir}/checkpoints/dataset_{jax.process_index()}",
            max_to_keep=config.keep_checkpoints,
        )
        dataset_manager.restore_or_initialize()

    # replicate parameters to each device
    train_state = replicate(train_state)

    # batch hellaswag dataset
    data, labels, lengths = prepare_hellaswag(config)

    train_loss = 0.0
    for step in range(step, config.train_steps):

        if step % config.eval_interval == 0:
            val_loss = 0.0
            for _, tokens in zip(range(config.eval_steps), val_ds):
                loss = eval_step(train_state, tokens)
                val_loss += loss[0].item()
            val_loss = val_loss / config.eval_steps
            print(val_loss)

            # hellaswag
            hellaswag_acc = eval_hellaswag(config, train_state, data, labels, lengths)
            print(hellaswag_acc)

            if step > 0:
                train_loss = train_loss / config.eval_interval
                print(
                    f"step: {step}, train_loss: {train_loss}, val_loss: {val_loss}",
                    f"hellaswag_acc: {hellaswag_acc}",
                )
            else:
                print(
                    f"step: {step}, val_loss: {val_loss}",
                    f"hellaswag_acc: {hellaswag_acc}",
                )

            if config.eval_only:
                break

            if val_loss < best_val_loss and config.keep_checkpoints > 0:
                best_val_loss = val_loss
                if jax.process_index() == 0:
                    # save train state in process 0
                    checkpoints.save_checkpoint(
                        f"{config.out_dir}/checkpoints/train_state",
                        unreplicate(train_state),
                        step,
                        keep=config.keep_checkpoints,
                        overwrite=True,
                    )
                dataset_manager.save(step)

            if (config.wandb is not None) and (jax.process_index() == 0):
                wandb.log(
                    {
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "hellaswag/acc": hellaswag_acc,
                        "lr": (
                            learning_rate(step)
                            if callable(learning_rate)
                            else learning_rate
                        ),
                        "step": step,
                        "block": step * config.batch_size * jax.device_count(),
                        "tokens": step
                        * config.batch_size
                        * jax.device_count()
                        * block_size,
                    },
                    step=step,
                )

            train_loss = 0.0

        tokens = next(train_iter)
        loss, train_state = train_step(train_state, tokens, keys_dropout)
        train_loss += loss[0].item()
