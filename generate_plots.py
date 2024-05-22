import matplotlib.pyplot as plt
import torch as t
from beartype import beartype as typed
from jaxtyping import Float
from sklearn.cluster import KMeans
from torch import Tensor as TT
from tqdm import tqdm
from transformers import AutoModelForCausalLM


SVD_CUTOFF = 200
CLUSTERS_CUTOFF = 25


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


def get_embeddings(model):
    model = AutoModelForCausalLM.from_pretrained(model)
    return model.get_input_embeddings().weight.detach()


model_names = {
    "scratch": "roneneldan/TinyStories-8M",
    "nested": "Mlxa/embeddings-nested-english",
    "flat": "Mlxa/embeddings-flat-english",
    "shuffle": "Mlxa/embeddings-flat_shuffle-english",
}
embeddings = {name: get_embeddings(path) for name, path in model_names.items()}
styles = {
    "scratch": "-",
    "nested": ":",
    "flat": "--",
    "shuffle": "-.",
}


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
