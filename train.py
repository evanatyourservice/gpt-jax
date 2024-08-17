import json
import time
from pprint import pprint
from typing import Tuple, Optional
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

from dataset import get_dataset
from optimizers.psgd_affine import affine


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
wandb.require("core")
tf.config.experimental.set_visible_devices([], "GPU")
tf.config.experimental.set_visible_devices([], "TPU")


@dataclass(frozen=True)
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int = None
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 0.00001
    initializer_range: float = 0.02
    summary_type: str = "cls_index"
    summary_use_proj: bool = True
    summary_activation: str = None
    summary_proj_to_labels: bool = True
    summary_first_dropout: float = 0.1
    scale_attn_weights: bool = True
    use_cache: bool = True
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False


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
class OptimizerConfig:
    type: str = "adamw"
    learning_rate: float = 0.001
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    betas: Tuple[float, float] = (0.9, 0.95)
    preconditioner_update_probability: float = 1.0
    update_global_norm_clip: Optional[float] = None
    update_elementwise_clip: bool = False
    max_size_triangular: int = 0
    max_skew_triangular: int = 0
    precond_lr: float = 1.0
    precond_init_scale: float = 1.0
    adaptive: bool = True


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    out_dir: str = os.path.expanduser(
        "~/gpt_out_dir"
    )  # output directory for checkpoints (can be gcs path)
    train_pattern: str = (
        "owt_data/train_??.tfrecord"  # training files glob pattern (can be gcs path)
    )
    val_pattern: str = (
        "owt_data/val_??.tfrecord"  # validation files glob pattern (can be gcs path)
    )
    shuffle_buffer_size: int = 128
    eval_interval: int = 500
    eval_steps: int = 16  # evaluate for this number of steps (per-device)
    hs_eval_steps: int = 16  # evaluate for this number of steps (per-device)
    eval_only: bool = False  # if True, script exits right after the first eval
    keep_checkpoints: int = 0  # number of historical checkpoints to keep
    batch_size: int = 16  # per-device batch size
    train_steps: int = 50000  # total number of training iterations
    bfloat16_compute: bool = False  # use bfloat16 for compute
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)  # wandb logging
    model: GPT2Config = field(default_factory=GPT2Config)  # gpt model config
    remat: bool = False  # set to True to rematerialize gradients during backward pass


@partial(
    jax.pmap, axis_name="batch", donate_argnums=(0,), static_broadcasted_argnums=(3,)
)
def train_step(
    state: TrainState, tokens: jnp.ndarray, dropout_key, use_bfloat16: bool
) -> Tuple[jnp.ndarray, TrainState]:

    dropout_key = jax.random.fold_in(dropout_key, state.step)

    def loss_fn(params) -> jnp.ndarray:
        X, Y = tokens[:, :-1], tokens[:, 1:]
        if use_bfloat16:
            X = X.astype(jnp.bfloat16)
            params = optax.tree_utils.tree_cast(params, jnp.bfloat16)
        logits = state.apply_fn(X, params=params, dropout_rng=dropout_key, train=True)[
            0
        ]
        logits = logits.astype(jnp.float32)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y).mean()

        # palm style z loss
        loss += 1e-4 * jax.scipy.special.logsumexp(logits, axis=-1).mean() ** 2

        return loss

    loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(state.params)

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


def prepare_hellaswag(config: TrainConfig):
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
                if len(input_ids) > config.model.n_positions:
                    continue
                lens.append(len(input_ids))
                input_ids = np.pad(
                    input_ids, (0, config.model.n_positions - len(input_ids))
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
    all_data = np.array(all_data).reshape(-1, n_devices, 4, config.model.n_positions)
    all_labels = np.array(all_labels).reshape(-1, n_devices)
    all_lengths = np.array(all_lengths).reshape(-1, n_devices, 4)
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
    model_config = transformers.GPT2Config(**asdict(config.model))
    model = transformers.FlaxAutoModelForCausalLM.from_config(model_config)
    params = model.params
    pprint(params, indent=2, width=120, compact=True)

    optimizer = []
    optimizer.append(optax.clip_by_global_norm(config.optimizer.grad_clip))
    print("using optimizer", config.optimizer.type)
    if config.optimizer.type in ["adam", "adamw"]:
        optimizer.append(
            optax.adamw(
                learning_rate,
                *config.optimizer.betas,
                weight_decay=config.optimizer.weight_decay,
                mask=param_decay_mask,
                mu_dtype=jnp.bfloat16,
            )
        )
    elif config.optimizer.type in ["psgd_affine", "affine"]:
        optimizer.append(
            affine(
                learning_rate=learning_rate,
                preconditioner_update_probability=config.optimizer.preconditioner_update_probability,
                b1=config.optimizer.betas[0],
                b2=config.optimizer.betas[1] if config.optimizer.adaptive else None,
                nesterov=False,
                update_global_norm_clip=config.optimizer.update_global_norm_clip,
                update_elementwise_clip=config.optimizer.update_elementwise_clip,
                weight_decay=config.optimizer.weight_decay,
                mask=param_decay_mask,
                max_size_triangular=config.optimizer.max_size_triangular,
                max_skew_triangular=config.optimizer.max_skew_triangular,
                step_normalizer_order="2nd",
                precond_lr=config.optimizer.precond_lr,
                precond_init_scale=config.optimizer.precond_init_scale,
                seed=None,
                mu_dtype=jnp.bfloat16,
                precision="tensorfloat32",
            )
        )
    else:
        raise ValueError("Unknown optimizer type")

    optimizer.append(optax.apply_every(config.optimizer.gradient_accumulation_steps))

    optimizer = optax.chain(*optimizer)

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
    config = tyro.cli(TrainConfig, default=get_default_config(), use_underscores=True)

    if config.wandb is not None and jax.process_index() == 0:
        wandb.init(**asdict(config.wandb))
        wandb.config.update(asdict(config))

    block_size = config.model.n_positions

    # ===== datasets =====
    train_ds = get_dataset(
        config.train_pattern,
        config.batch_size,
        block_size,
        config.shuffle_buffer_size,
        seed=config.seed,
    )
    train_ds = flax.jax_utils.prefetch_to_device(train_ds, 1)

    val_ds = get_dataset(config.val_pattern, config.batch_size, block_size)

    # =====  init parameters ============
    key = jax.random.PRNGKey(config.seed)
    key, key_params, key_dropout = jax.random.split(key, 3)
    # make sure dropout keys are different for each device (local and global)
    key_dropout = jax.random.fold_in(key_dropout, jax.process_index())
    keys_dropout = jax.random.split(key_dropout, jax.local_device_count())

    # ===== learning rate schedule =====
    optimizer = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                0.0, config.optimizer.learning_rate, config.optimizer.warmup_steps
            ),
            optax.linear_schedule(
                config.optimizer.learning_rate,
                0.0,
                config.train_steps - config.optimizer.warmup_steps,
            ),
        ],
        boundaries=[config.optimizer.warmup_steps],
    )

    train_state = init_train_state(key_params, config, optimizer)

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

    train_losses = []
    print("starting training")
    for step in range(step, config.train_steps):
        tokens = next(train_iter)
        loss, train_state = train_step(
            train_state, tokens, keys_dropout, config.bfloat16_compute
        )
        train_losses.append(loss[0].item())

        if (config.wandb is not None) and (jax.process_index() == 0) and step % 10 == 0:
            train_loss = np.mean(train_losses)
            wandb.log(
                {
                    "train_loss": train_loss,
                    "lr": (optimizer(step) if callable(optimizer) else optimizer),
                    "tokens": step
                    * config.batch_size
                    * jax.device_count()
                    * block_size,
                },
                step=step,
            )

            train_losses = []

        if step % config.eval_interval == 0 and step > 0:
            val_losses = []
            for _ in range(config.eval_steps):
                tokens = next(val_ds)
                loss = eval_step(train_state, tokens)
                val_losses.append(loss[0].item())

            val_loss = np.mean(val_losses)

            # hellaswag
            hellaswag_acc = eval_hellaswag(config, train_state, data, labels, lengths)

            print(
                f"step: {step}, val_loss: {val_loss:.4f}, "
                f"hellaswag_acc: {hellaswag_acc:.4f}"
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
                    {"val_loss": val_loss, "hellaswag_acc": hellaswag_acc}, step=step
                )
