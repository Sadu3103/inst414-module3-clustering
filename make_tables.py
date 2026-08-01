import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def table_png(col_labels, rows, fname, title, widths=None, hl_col=None):
    fig, ax = plt.subplots(figsize=(len(col_labels)*1.35+1.2, len(rows)*0.42+1.1))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 1.55)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d8dee4")
        if r == 0:
            cell.set_facecolor("#2b6cb0"); cell.set_text_props(color="w", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f4f6f8")
        if hl_col is not None and c == hl_col and r > 0:
            cell.set_facecolor("#ffe9b3")
    ax.set_title(title, fontsize=13, weight="bold", pad=14)
    plt.tight_layout(); plt.savefig(fname, dpi=200, bbox_inches="tight"); plt.close()

ks = [2,3,4,5,6,7,8,9,10]
sil = [".0100",".0097",".0154",".0178",".0173",".0185",".0187",".0190",".0183"]
inr = ["3750.8","3734.1","3708.5","3694.6","3684.3","3675.2","3669.3","3663.6","3658.5"]
table_png(["k","silhouette (cosine)","inertia"],
          [[str(k), s, i] for k, s, i in zip(ks, sil, inr)],
          "table1_k_selection.png", "Choosing k: silhouette and inertia across k = 2 to 10")
for (r,) in []: pass

rows = [["comp.sys.mac.hardware","0","0","627","0","85"],
        ["rec.sport.hockey","0","0","1","658","98"],
        ["sci.med","5","66","5","0","715"],
        ["soc.religion.christian","623","0","2","0","243"],
        ["talk.politics.guns","2","0","0","1","704"]]
fig, ax = plt.subplots(figsize=(9.5, 3.4)); ax.axis("off")
t = ax.table(cellText=rows, colLabels=["true board","c0","c1","c2","c3","c4"], cellLoc="center", loc="center")
t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 1.6)
big = {(1,3),(2,4),(3,5),(4,1),(5,5)}
for (r,c), cell in t.get_celld().items():
    cell.set_edgecolor("#d8dee4")
    if r == 0: cell.set_facecolor("#2b6cb0"); cell.set_text_props(color="w", weight="bold")
    elif (r,c) in big: cell.set_facecolor("#ffd88a"); cell.set_text_props(weight="bold")
    elif r % 2 == 0: cell.set_facecolor("#f4f6f8")
    if c == 0 and r > 0: cell.set_text_props(ha="left")
ax.set_title("Clusters vs held out board labels (ARI = 0.477)", fontsize=13, weight="bold", pad=14)
plt.tight_layout(); plt.savefig("table2_contingency.png", dpi=200, bbox_inches="tight")
print("done")
