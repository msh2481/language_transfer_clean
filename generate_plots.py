import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch as t
from beartype import beartype as typed
from jaxtyping import Bool, Float, Int
from torch import Tensor as TT
from sklearn.cluster import KMeans


@typed
def spectrum(data: Float[TT, "n d"]) -> Float[TT, "d"]:
    data = data - data.mean(0)
    vals = t.linalg.svdvals(data)
    return vals / vals.max()


@typed
def clusters(data: Float[TT, "n d"], max_clusters: int = 30) -> Float[TT, "k"]:

    one_cluster = (data - data.mean(0)).square().sum()
    results = [one_cluster, one_cluster]
    for k in range(2, max_clusters):
        kmeans = KMeans(n_clusters=k, n_init=1, max_iter=100, algorithm="elkan")
        kmeans.fit(data)
        results.append(kmeans.inertia_)
    return t.tensor(results) / one_cluster



def get_embeddings(model):
    model = AutoModelForCausalLM.from_pretrained(model)
    return model.get_input_embeddings().weight.detach()

model_names = {
    "nested": "Mlxa/embeddings-nested-english",
    "flat": "Mlxa/embeddings-flat-english",
    "shuffle": "Mlxa/embeddings-flat_shuffle-english",
    "scratch": "roneneldan/TinyStories-8M"
}
embeddings = {name: get_embeddings(path) for name, path in model_names.items()}

SVD_CUTOFF = 200
CLUSTERS_CUTOFF = 100

plt.figure(figsize=(10, 10), dpi=200)
for name, emb in tqdm(embeddings.items()):
    sp = spectrum(emb)
    plt.plot(sp[:SVD_CUTOFF], label=name)
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
for name, emb in emb_dict.items():
    print(name)
    save[name] = clusters(emb, max_clusters=CLUSTERS_CUTOFF)
    plt.plot(save[name], label=name)
plt.yscale("log")
plt.ylim(0.3, 1)
plt.legend(fontsize=24)
plt.tick_params(axis="both", which="major", labelsize=24)
plt.tick_params(axis="both", which="major", length=10, width=1)
plt.tick_params(axis="both", which="minor", length=5, width=1)
plt.savefig(f"img/clusters.png")
print("Clusters plot ready")