import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import torch as t
from beartype import beartype as typed
from jaxtyping import Float
from language_modeling import cloze_test
from probes import mixed_probe
from sklearn.cluster import KMeans  # type: ignore
from torch import Tensor as TT
from tqdm import tqdm  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore


SVD_CUTOFF = 200
CLUSTERS_CUTOFF = 25

COMPUTE_SPECTRUM = False
COMPUTE_CLUSTERS = False
COMPUTE_CLOZE = False
COMPUTE_PROBE = True


@typed
def spectrum(data: Float[TT, "n d"]) -> Float[TT, "d"]:
    data = data - data.mean(0)
    vals = t.linalg.svdvals(data)
    return vals / vals.max()


@typed
def clusters(data: Float[TT, "n d"], max_clusters: int) -> Float[TT, "k"]:
    one_cluster = (data - data.mean(0)).square().sum()
    results = [one_cluster]
    for k in tqdm(range(2, max_clusters + 1)):
        kmeans = KMeans(n_clusters=k, n_init=1, max_iter=100, algorithm="elkan")
        kmeans.fit(data)
        results.append(kmeans.inertia_)
    return t.tensor(results) / one_cluster


@typed
def get_embeddings(model_name: str) -> Float[TT, "n d"]:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model.get_input_embeddings().weight.detach()


@typed
def analyze_cloze(model_name: str) -> dict[str, float]:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results = {}
    for task_name, prompts in tasks.items():
        results[task_name] = cloze_test(model, tokenizer, prompts).mean().item()
    return results


@typed
def analyze_probe(
    embedding: Float[TT, "n d"], features: pd.DataFrame
) -> dict[str, float]:
    return mixed_probe(embedding, features)


repo_name = open("SECRET.txt").read().strip()
embeddings_model_names = {
    "scratch": "roneneldan/TinyStories-8M",
    "nested": f"{repo_name}/embeddings-nested-english",
    "flat": f"{repo_name}/embeddings-flat-english",
    "shuffle": f"{repo_name}/embeddings-flat_shuffle-english",
}

embeddings = {
    name: get_embeddings(path) for name, path in embeddings_model_names.items()
}

styles = {
    "scratch": "-",
    "nested": ":",
    "flat": "--",
    "shuffle": "-.",
}

if COMPUTE_SPECTRUM:
    plt.figure(figsize=(10, 10), dpi=200)
    for name, emb in tqdm(embeddings.items()):
        sp = spectrum(emb)
        plt.plot(sp[:SVD_CUTOFF], styles[name], label=name)
    plt.yscale("log")
    plt.ylim(0.03, 1)
    plt.legend(fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=24)
    plt.tick_params(axis="both", which="major", length=10, width=1)
    plt.tick_params(axis="both", which="minor", length=5, width=1)
    plt.savefig(f"img/spectrum.png")
    print("Spectrum plot ready")

if COMPUTE_CLUSTERS:
    plt.figure(figsize=(10, 10), dpi=200)
    save = {}
    for name, emb in tqdm(embeddings.items()):
        save[name] = clusters(emb, max_clusters=CLUSTERS_CUTOFF)
        plt.plot(range(1, CLUSTERS_CUTOFF + 1), save[name], styles[name], label=name)
    plt.ylim(0.5, 1)
    plt.legend(fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=24)
    plt.tick_params(axis="both", which="major", length=10, width=1)
    plt.tick_params(axis="both", which="minor", length=5, width=1)
    plt.savefig(f"img/clusters.png")
    print("Clusters plot ready")

if COMPUTE_CLOZE:
    with open("cloze_tasks.json") as f:
        tasks = json.load(f)

    model_names = {
        "nested E": f"{repo_name}/embeddings-nested-english",
        "nested ELT": f"{repo_name}/tuned-nested-english",
        "flat E": f"{repo_name}/embeddings-flat-english",
        "flat ELT": f"{repo_name}/tuned-flat-english",
        "shuffle E": f"{repo_name}/embeddings-flat_shuffle-english",
        "shuffle ELT": f"{repo_name}/tuned-flat_shuffle-english",
        "scratch 8M": "roneneldan/TinyStories-8M",
        "scratch 33M": "roneneldan/TinyStories-33M",
    }

    task_names = {
        "synonyms and antonyms": "synonyms and antonyms",
        "single - plural": "single plural",
        "logical relations": "logical relations",
        "subject-verb agreement": "subject verb agreement",
        "prepositions": "prepositions",
        "conjunctions": "conjunctions",
        "temporal understanding": "temporal understanding",
        "spatial understanding": "spatial understanding",
        "quantitative reasoning": "quantitative reasoning",
        "emotions": "emotions",
        "narrative understanding": "narrative understanding",
        "ethics": "ethics",
    }

    # mapping model_name -> task_name -> score
    scores: dict[str, dict[str, float]] = {}
    for model_name, model_path in tqdm(model_names.items()):
        scores[model_name] = analyze_cloze(model_path)

    # format to LaTeX table described above
    table_header = (
        r"""
\begin{table}[ht]
\centering
\begin{tabular}{|p{70pt}|*{8}{p{27pt}|}}
\hline & """
        + " & ".join(
            [rf"\texttt{{\scriptsize{{{model}}}}}" for model in model_names.keys()]
        )
        + r"""\\
\hline
"""
    )

    table_rows = ""
    averages: dict[str, list[float]] = {key: [] for key in model_names.keys()}

    for task, task_name in task_names.items():
        row = f"    \\scriptsize{{{task}}} "
        for model in model_names.keys():
            score = scores[model][task_name]
            row += f"& {score:.2f} "
            averages[model].append(score)
        row += r"\\ \hline" + "\n"
        table_rows += row

    avg_scores = {
        model: sum(scores) / len(scores) for model, scores in averages.items()
    }

    average_row = (
        r"\hline\scriptsize{{\textbf{{Average}}}} & "
        + " & ".join([f"{avg_scores[model]:.2f}" for model in model_names.keys()])
        + r" \\"
    )

    table_footer = r"""
\hline
\end{tabular}
\caption{Results on the Tiny-Cloze benchmark. Fine-tuning on \texttt{flat\_shuffle} gives the highest average score across three synthetic languages.}
\label{tab:cloze}
\end{table}
"""

    latex_table = table_header + table_rows + average_row + table_footer

    with open("results_table.tex", "w") as f:
        f.write(latex_table)
    print("Cloze test table ready")

if COMPUTE_PROBE:
    features = pd.read_csv("word_features.csv", escapechar="\\")
    features["frequency"] = features["frequency"].astype(float).apply(np.log1p)
    features = pd.get_dummies(features, columns=["pos_tag"])
    to_remove = [c for c in features.columns if (features[c] != False).sum() < 200]
    features = features.drop(to_remove, axis=1)

    names = ["scratch", "nested", "flat", "shuffle"]
    results = {name: analyze_probe(embeddings[name], features) for name in names}

    with open("probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Probe results ready")
