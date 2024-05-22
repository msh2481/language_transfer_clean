from itertools import islice

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype as typed
from datasets import IterableDataset
from jaxtyping import Float, Int
from tokenization import dependencies_tokenizer
from torch import Tensor as TT
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@typed
def module_device(model: nn.Module) -> str:
    return str(next(model.parameters()).device)


@typed
def tokenize(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str | list[int] | Int[TT, "seq"],
    device: str = "cpu",
) -> dict[str, Int[TT, "batch seq"]]:
    if isinstance(prompt, str):
        result = tokenizer(prompt, return_tensors="pt")
    else:
        result = tokenizer(tokenizer.decode(prompt), return_tensors="pt")
    result["labels"] = result["input_ids"]
    assert (result["input_ids"] < len(tokenizer) - 2).all()
    return {name: value.to(device) for name, value in result.items()}


@typed
def generate_sample(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str | list[int] | Int[TT, "seq"],
    max_new_tokens: int,
    keep_prompt: bool = False,
) -> str:
    inputs = tokenize(tokenizer, prompt, device=module_device(model))
    pad_token_id: int = tokenizer.pad_token_id or tokenizer.eos_token_id
    suffix: int = 0 if keep_prompt else len(inputs["input_ids"][0])
    output: Int[TT, "suffix"] = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=pad_token_id,
        bad_words_ids=[[pad_token_id]],
    )[0, suffix:]
    return tokenizer.decode(output.detach().cpu())


@typed
def get_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str | list[int] | Int[TT, "seq"],
) -> Float[TT, "seq vocab"]:
    with t.no_grad():
        inputs = tokenize(tokenizer, prompt, device=module_device(model))
        logits: Float[TT, "seq vocab"] = model(**inputs).logits.squeeze(0)
        raw_lp: Float[TT, "seq"] = F.log_softmax(logits.cpu().detach(), dim=-1)
        return raw_lp.roll(1, dims=0)


@typed
def logprobs_to_losses(
    lp: Float[TT, "seq vocab"], labels: Int[TT, "seq"]
) -> Float[TT, "seq"]:
    return -lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


@typed
def get_loss(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt: str
) -> float:
    input_ids = tokenize(tokenizer, prompt, device=module_device(model))
    return model(**input_ids).loss.item()


@typed
def get_losses(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt: str
) -> Float[TT, "seq"]:
    """
    Remember, that the first element in the losses tensor is meaningless.
    """
    logprobs: Float[TT, "seq vocab"] = get_logprobs(model, tokenizer, prompt)
    ids: Int[TT, "seq"] = tokenize(tokenizer, prompt)["input_ids"][0]
    losses: Float[TT, "seq"] = logprobs_to_losses(logprobs, ids)
    return losses


@typed
def explore(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str | list[int] | Int[TT, "seq"],
    n_tokens: int = 10,
    show_sample: bool = False,
) -> Float[TT, "seq"]:
    from utils import show_string_with_weights

    ids: Int[TT, "seq"] = tokenize(tokenizer, prompt)["input_ids"][0]
    logprobs: Float[TT, "seq vocab"] = get_logprobs(model, tokenizer, prompt)
    losses: Float[TT, "seq"] = logprobs_to_losses(logprobs, ids)

    # 0 for perfect prediction, 1 for infinite loss
    weights: Float[TT, "seq"] = (losses[-n_tokens:] / 2).tanh()
    tokens: list[str] = [tokenizer.decode(i) for i in ids[-n_tokens:]]
    show_string_with_weights(tokens, weights)

    if show_sample:
        sampled: str = generate_sample(model, tokenizer, ids[:-n_tokens], n_tokens)
        print(sampled)

    return losses


@typed
def explore_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: IterableDataset,
    n_samples: int = 32,
) -> None:
    losses = [
        explore(model, tokenizer, sample["input_ids"])[1:].mean().item()
        for sample in islice(dataset, n_samples)
    ]
    print(f"Mean loss: {sum(losses) / len(losses):.3f}")


@typed
def evaluate_cloze(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    cloze: str,
) -> float:
    assert prompt.count("#") == 1
    assert not prompt.startswith("#")
    prompt = prompt.replace("#", cloze)
    return get_loss(model, tokenizer, prompt)


@typed
def cloze_test(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tests: list[list[str]],
) -> Float[TT, "n"]:
    results = []
    for prompt, correct, incorrect in tests:
        loss_correct = evaluate_cloze(model, tokenizer, prompt, correct)
        loss_incorrect = evaluate_cloze(model, tokenizer, prompt, incorrect)
        results.append(loss_incorrect - loss_correct)
    return t.tensor(results)
