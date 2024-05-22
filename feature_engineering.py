from collections import Counter

import nltk  # type: ignore
import pandas as pd  # type: ignore
import torch as t
from datasets import load_dataset  # type: ignore
from tqdm import tqdm  # type: ignore
from transformers import AutoTokenizer  # type: ignore

if __name__ == "__main__":
    nltk.download("averaged_perceptron_tagger")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download("words")

    dataset = load_dataset("roneneldan/TinyStories", streaming=True)
    train_dataset = dataset["train"]

    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")
    all_tokens = [tokenizer.decode([i]) for i in range(len(tokenizer))]

    tokenized_dataset = train_dataset.map(lambda x: tokenizer(x["text"]), batched=True)
    counter: Counter[int] = Counter()
    for tokens in tqdm(tokenized_dataset):
        counter.update(tokens["input_ids"])
    d = [counter[i] for i in range(len(tokenizer))]
    features = pd.DataFrame({"token": all_tokens, "frequency": d})
    features["pos_tag"] = features["token"].apply(
        lambda x: nltk.pos_tag([x.strip()])[0][1]
    )
    features["start_space"] = features["token"].str.startswith(" ")
    features.at[201, "token"] = "\\n"
    features.to_csv("word_features.csv", index_label="id", escapechar="\\")
