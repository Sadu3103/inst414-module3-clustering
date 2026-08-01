import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import TruncatedSVD
import pandas as pd

RANDOM_STATE = 414
CATEGORIES = [
    "comp.sys.mac.hardware",
    "rec.sport.hockey",
    "sci.med",
    "talk.politics.guns",
    "soc.religion.christian",
]

raw = fetch_20newsgroups(
    subset="all",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
    random_state=RANDOM_STATE,
)

docs, labels = [], []
for text, y in zip(raw.data, raw.target):
    t = text.strip()
    if len(t.split()) >= 40:
        docs.append(t)
        labels.append(y)
labels = np.array(labels)
print(f"posts retained: {len(docs)} of {len(raw.data)}")

vec = TfidfVectorizer(
    stop_words="english",
    min_df=5,
    max_df=0.4,
    max_features=20000,
    sublinear_tf=True,
    norm="l2",
)
X = vec.fit_transform(docs)
terms = np.array(vec.get_feature_names_out())
print(f"matrix: {X.shape[0]} posts x {X.shape[1]} terms")

ks, sils, inertias = list(range(2, 11)), [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    lab = km.fit_predict(X)
    sils.append(silhouette_score(X, lab, metric="cosine", random_state=RANDOM_STATE))
    inertias.append(km.inertia_)
    print(f"k={k}  silhouette(cosine)={sils[-1]:.4f}  inertia={km.inertia_:.1f}")

BEST_K = 5
km = KMeans(n_clusters=BEST_K, n_init=25, random_state=RANDOM_STATE)
assign = km.fit_predict(X)

order = km.cluster_centers_.argsort()[:, ::-1]
summary = {}
for c in range(BEST_K):
    top = terms[order[c, :12]].tolist()
    members = np.where(assign == c)[0]
    center = km.cluster_centers_[c]
    sims = (X[members] @ center) / (np.linalg.norm(center) + 1e-12)
    closest = members[np.argsort(-sims)[:3]]
    summary[c] = {"size": int(len(members)), "top_terms": top,
                  "examples": [docs[i][:340].replace("\n", " ") for i in closest]}
    print(f"\n=== cluster {c}  n={len(members)} ===")
    print("terms:", ", ".join(top))
    for i, ex in enumerate(summary[c]["examples"], 1):
        print(f"  ex{i}: {ex}")

ari = adjusted_rand_score(labels, assign)
print(f"\nAdjusted Rand Index vs held-out newsgroup labels: {ari:.3f}")
ct = pd.crosstab(pd.Series([raw.target_names[i] for i in labels], name="true_forum"),
                 pd.Series(assign, name="cluster"))
print(ct.to_string())
ct.to_csv("contingency_table.csv")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ks, sils, "o-", color="#2b6cb0")
ax[0].axvline(BEST_K, ls="--", c="crimson")
ax[0].set(xlabel="k", ylabel="mean silhouette (cosine)", title="Silhouette by k")
ax[1].plot(ks, inertias, "o-", color="#2f855a")
ax[1].axvline(BEST_K, ls="--", c="crimson")
ax[1].set(xlabel="k", ylabel="inertia (SSE)", title="Elbow curve")
plt.tight_layout(); plt.savefig("fig1_k_selection.png", dpi=150)

svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
P = svd.fit_transform(X)
plt.figure(figsize=(7, 6))
for c in range(BEST_K):
    m = assign == c
    plt.scatter(P[m, 0], P[m, 1], s=6, alpha=0.5, label=f"cluster {c} (n={m.sum()})")
plt.legend(markerscale=3, fontsize=8)
plt.title("Forum posts in 2-D LSA space, colored by cluster")
plt.xlabel("SVD component 1"); plt.ylabel("SVD component 2")
plt.tight_layout(); plt.savefig("fig2_clusters_svd.png", dpi=150)

pd.DataFrame({"k": ks, "silhouette_cosine": sils, "inertia": inertias}).to_csv("k_selection.csv", index=False)
json.dump({"ari": ari, "k": BEST_K, "n_docs": len(docs), "clusters": summary},
          open("cluster_summary.json", "w"), indent=2)
print("\nwrote: fig1_k_selection.png fig2_clusters_svd.png k_selection.csv contingency_table.csv cluster_summary.json")
