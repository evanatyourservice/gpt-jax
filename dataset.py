import json
from typing import Optional
import jax
import numpy as np
import tensorflow as tf
import tiktoken
from tqdm import tqdm


OPTIONS = tf.data.Options()
OPTIONS.deterministic = True
OPTIONS.autotune.enabled = True


def get_dataset(
    pattern: str,
    batch_size: int = 8,
    block_size: int = 1024,
    shuffle_buffer_size: Optional[int] = None,
) -> tf.data.Dataset.as_numpy_iterator:
    file_ds = tf.data.Dataset.list_files(pattern, shuffle=False)
    file_ds = file_ds.shard(jax.process_count(), jax.process_index())
    file_ds = file_ds.repeat()

    ds = tf.data.TFRecordDataset(file_ds, num_parallel_reads=tf.data.AUTOTUNE)

    # each element of the dataset is a tokenized string
    feature_description = {
        "ids": tf.io.FixedLenFeature([], tf.string, default_value="")
    }

    def parse_example(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        return tf.io.decode_raw(example["ids"], tf.uint16)

    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    # here we shuffle each group of tokens and then unbatch into a single
    # contiguous sequence of ids, we then chunk the sequence into blocks
    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.unbatch().batch(block_size + 1, drop_remainder=True)

    # blocks are then shuffled and batched
    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.batch(jax.local_device_count(), drop_remainder=True)
    ds = ds.with_options(OPTIONS)
    ds = ds.prefetch(5)
    ds = ds.as_numpy_iterator()
    return ds


def prepare_hellaswag(config):
    """Read file and tokenize the hellaswag dataset."""
    print("preparing hellaswag")

    block_size = config.model.n_positions
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
                # +1 for [1:] input [:-1] target shift
                if len(input_ids) > block_size + 1:
                    continue
                lens.append(len(input_ids))
                input_ids = np.pad(input_ids, (0, block_size + 1 - len(input_ids)))
                to_concat.append(input_ids)
            all_data.append(np.array(to_concat))  # (4, block_size)
            all_labels.append(correct_end)  # scalar
            all_lengths.append(np.array(lens))  # (4,)

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    all_lengths = np.array(all_lengths)

    ds = tf.data.Dataset.from_tensor_slices((all_data, all_labels, all_lengths))
    ds = ds.shard(jax.process_count(), jax.process_index())
    ds = ds.repeat()
    ds = ds.batch(
        max(config.batch_size // 4, 1), drop_remainder=True
    )  # (b, 4, block_size)
    ds = ds.batch(jax.local_device_count(), drop_remainder=True)
    ds = ds.prefetch(5)
    ds = ds.as_numpy_iterator()

    return ds
