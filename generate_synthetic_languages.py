import argparse
from collections import deque
from typing import Annotated

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
import torch as t
from beartype import beartype as typed
from beartype.typing import Callable
from beartype.vale import Is
from tokenization import dependencies_tokenizer
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast  # type: ignore
from utils import seed_everything

Even = Annotated[int, Is[lambda x: x % 2 == 0]]


@typed
def nested_dependencies(
    seq_len: Even,
    vocab_size: Even,
    tokenizer: PreTrainedTokenizerFast,
) -> str:
    """
    Returns a sequence of `seq_len` tokens structured as nesting brackets
    of `vocab_size` different types. Token `2 * x` is an open bracket of
    type `x` and `2 * x + 1` is the corresponding closing one.
    """
    p_open = 0.4
    open_types: deque[int] = deque()
    data = [0] * seq_len
    for i in range(seq_len):
        should_open = t.rand(size=()) < p_open
        must_open = not open_types
        must_close = len(open_types) == seq_len - i
        if (should_open or must_open) and not must_close:
            tp = int(t.randint(low=0, high=vocab_size // 2, size=()))
            data[i] = 2 * tp
            open_types.append(tp)
        else:
            tp = open_types.pop()
            data[i] = 2 * tp + 1
    return tokenizer.decode(data)


@typed
def flat_dependencies(
    seq_len: Even,
    vocab_size: Even,
    tokenizer: PreTrainedTokenizerFast,
) -> str:
    """
    Returns a sequence of `seq_len` matched tokens
    of `vocab_size` different types. Token `2 * x` is an open bracket of
    type `x` and `2 * x + 1` is the corresponding closing one.
    """
    p_open = 0.4
    open_types: list[int] = []
    data = [0] * seq_len
    for i in range(seq_len):
        should_open = t.rand(size=()) < p_open
        must_open = not open_types
        must_close = len(open_types) == seq_len - i
        if (should_open or must_open) and not must_close:
            tp = int(t.randint(low=0, high=vocab_size // 2, size=()))
            data[i] = 2 * tp
            open_types.append(tp)
        else:
            pos = int(t.randint(low=0, high=len(open_types), size=()))
            tp = open_types.pop(pos)
            data[i] = 2 * tp + 1
    return tokenizer.decode(data)


@typed
def flat_shuffle(
    seq_len: Even,
    group_len: int,
    vocab_size: Even,
    tokenizer: PreTrainedTokenizerFast,
) -> str:
    """
    Returns a sequence of `seq_len` matched tokens of `vocab_size` different types.
    Open brackets are shuffled ranges of consecutive integers within groups of `group_len` tokens.
    Token `2 * x` is an open bracket of type `x` and `2 * x + 1` is the corresponding closing one.
    """
    p_open = 0.4
    open_types: list[int] = []
    data = [0] * seq_len
    shuffled_range: list[int] = []
    for i in range(seq_len):
        if not shuffled_range:
            range_start = int(t.randint(0, vocab_size // 2 - group_len, size=()))
            range_tensor = t.arange(range_start, range_start + group_len)
            shuffled_range = range_tensor[t.randperm(group_len)].tolist()
        should_open = t.rand(size=()) < p_open
        must_open = not open_types
        must_close = len(open_types) == seq_len - i
        if (should_open or must_open) and not must_close:
            tp = shuffled_range.pop()
            data[i] = 2 * tp
            open_types.append(tp)
        else:
            pos = int(t.randint(low=0, high=len(open_types), size=()))
            tp = open_types.pop(pos)
            data[i] = 2 * tp + 1
    return tokenizer.decode(data)


@typed
def generate_batch(
    generator: Callable[..., pa.RecordBatch],
    *args,
    **kwargs,
) -> Callable[[int, int], pa.RecordBatch]:
    def batch_generator(start: int, end: int) -> pa.RecordBatch:
        nums = end - start
        return pa.RecordBatch.from_arrays(
            [pa.array(generator(*args, **kwargs) for i in range(nums))],
            ["text"],
        )

    return batch_generator


@typed
def write_to_parquet(
    output_file: str,
    batch_size: int,
    total_size: int,
    generator: Callable[[int, int], pa.RecordBatch],
):
    schema = pa.schema([pa.field("text", pa.string())])
    with pq.ParquetWriter(output_file, schema) as writer:
        for start in tqdm(range(0, total_size, batch_size)):
            end = min(start + batch_size, total_size)
            batch = generator(start, end)
            writer.write_batch(batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic language data and write to Parquet files."
    )
    parser.add_argument(
        "generator",
        type=str,
        choices=["nested", "flat", "flat_shuffle"],
        help="Language to generate",
    )
    args = parser.parse_args()

    seed_everything(0)
    seq_len = 512
    group_len = 8
    vocab_size = 500
    tokenizer = dependencies_tokenizer(vocab_size=vocab_size)

    batch_generator = None
    if args.generator == "nested":
        batch_generator = generate_batch(
            generator=nested_dependencies,
            seq_len=seq_len,
            vocab_size=vocab_size,
            tokenizer=tokenizer,
        )
    elif args.generator == "flat":
        batch_generator = generate_batch(
            generator=flat_dependencies,
            seq_len=seq_len,
            group_len=group_len,
            vocab_size=vocab_size,
            tokenizer=tokenizer,
        )
    elif args.generator == "flat_shuffle":
        batch_generator = generate_batch(
            generator=flat_shuffle,
            seq_len=seq_len,
            group_len=group_len,
            vocab_size=vocab_size,
            tokenizer=tokenizer,
        )
    assert batch_generator is not None

    write_to_parquet(
        output_file=f"{args.generator}_train.parquet",
        batch_size=10**3,
        total_size=2 * 10**6,
        generator=batch_generator,
    )
    write_to_parquet(
        output_file=f"{args.generator}_test.parquet",
        batch_size=10**3,
        total_size=10**4,
        generator=batch_generator,
    )
