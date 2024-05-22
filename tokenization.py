from typing import Annotated

from beartype import beartype as typed
from beartype.vale import Is
from tokenizers import models, pre_tokenizers, Tokenizer  # type: ignore
from transformers import PreTrainedTokenizerFast  # type: ignore

Even = Annotated[int, Is[lambda x: x % 2 == 0]]


@typed
def wordlevel_tokenizer(vocab: list[str]) -> PreTrainedTokenizerFast:
    vocab += ["[UNK]", "[PAD]"]
    model = models.WordLevel(
        {word: i for i, word in enumerate(vocab)},
        unk_token="[UNK]",
    )
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )


@typed
def dependencies_tokenizer(vocab_size: Even) -> PreTrainedTokenizerFast:
    vocab = [f"<{i//2+1}" if i % 2 == 0 else f"{i//2+1}>" for i in range(vocab_size)]
    return wordlevel_tokenizer(vocab)
