# INST414 Module 3: Clustering Online Forum Posts

TF-IDF + cosine k-means over 3,835 Usenet forum posts (20 Newsgroups archive, 5 boards).
Selects k, characterizes clusters, validates against held-out board labels (ARI 0.477).

Run:

    pip install scikit-learn matplotlib pandas
    python cluster_forums.py

Outputs: fig1_k_selection.png, fig2_clusters_svd.png, k_selection.csv, contingency_table.csv, cluster_summary.json
