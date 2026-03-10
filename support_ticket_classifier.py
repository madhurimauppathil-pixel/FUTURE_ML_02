

import re, random, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42)

# ─── LIGHT PALETTE ────────────────────────────────────────────────────────────
P = {
    "bg":        "#F8F6F2",       # warm off-white
    "card":      "#FFFFFF",
    "card2":     "#F0EEF9",       # lavender tint
    "stroke":    "#E2DDF5",
    "text":      "#1E1B2E",       # deep navy-black
    "muted":     "#8B85A8",
    "sub":       "#C4BFD9",

    # category palette — dusty pastels
    "c1": "#B8A9F5",  # soft violet
    "c2": "#A8D8EA",  # sky blue
    "c3": "#FFD6A5",  # peach
    "c4": "#B5EAD7",  # mint
    "c5": "#FFB7B2",  # blush
    "c6": "#C7CEEA",  # periwinkle

    # priority
    "high":   "#FF8B94",
    "medium": "#FFCA85",
    "low":    "#85D6B3",

    # accent
    "accent": "#7C5CBF",
    "accent2":"#5BA4CF",
}

CAT_COLORS  = [P["c1"],P["c2"],P["c3"],P["c4"],P["c5"],P["c6"]]
PRIO_COLORS = {"High": P["high"], "Medium": P["medium"], "Low": P["low"]}
CATEGORIES  = ["Billing","Technical Support","Account Management",
               "Feature Request","Shipping & Delivery","General Inquiry"]
PRIORITIES  = ["High","Medium","Low"]
OUT         = "."

# ─── GLOBAL STYLE ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  P["bg"],
    "axes.facecolor":    P["card"],
    "axes.edgecolor":    P["stroke"],
    "axes.linewidth":    1.2,
    "axes.labelcolor":   P["text"],
    "axes.labelsize":    10,
    "xtick.color":       P["muted"],
    "ytick.color":       P["muted"],
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "text.color":        P["text"],
    "grid.color":        P["stroke"],
    "grid.linestyle":    "-",
    "grid.alpha":        1.0,
    "axes.grid":         True,
    "axes.grid.axis":    "x",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "figure.dpi":        150,
})

def save(fig, name):
    path = f"{OUT}/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=P["bg"], edgecolor="none")
    plt.close(fig)
    print(f"  ✓  {name}")

def title_bar(fig, label, y=0.97):
    """Thin coloured rule + title treatment."""
    fig.text(0.06, y, label, fontsize=15, fontweight="bold",
             color=P["text"], va="top")
    fig.text(0.06, y-0.045, "FUTURE_ML_02  ·  Support Ticket Classifier",
             fontsize=8, color=P["muted"], va="top", style="italic")
    # decorative rule
    fig.add_artist(mpatches.FancyArrowPatch(
        (0.06, y-0.065), (0.94, y-0.065),
        arrowstyle="-", color=P["stroke"],
        linewidth=1.5, transform=fig.transFigure, clip_on=False))

def pill(ax, x, y, text, color, textcolor=None, w=0.14, h=0.055, fontsize=8.5):
    """Rounded pill badge."""
    tc = textcolor or P["text"]
    r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.01",
                       facecolor=color, edgecolor="none",
                       transform=ax.transAxes, clip_on=False, zorder=5)
    ax.add_patch(r)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=tc, fontweight="bold", transform=ax.transAxes, zorder=6)

# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════
TEMPLATES = {
    "Billing": {
        "priority_weights": [0.55, 0.35, 0.10],
        "texts": [
            "I was charged twice for my subscription this month. Please refund immediately.",
            "My invoice shows an incorrect amount. I need this corrected urgently.",
            "I cannot access my account after my payment was declined.",
            "Why is my bill higher than usual? There are unrecognised charges on my account.",
            "I cancelled my subscription but I am still being billed every month.",
            "The payment gateway keeps failing when I try to update my credit card details.",
            "I was promised a discount but it was never applied to my invoice.",
            "My free trial ended and I was charged without any prior notice.",
            "I need a VAT receipt for my last three payments for tax purposes.",
            "The promo code I entered is not reducing the total price at checkout.",
        ]},
    "Technical Support": {
        "priority_weights": [0.45, 0.40, 0.15],
        "texts": [
            "The application crashes every time I try to export a PDF report.",
            "I cannot log in as the two-factor authentication code is not arriving.",
            "My data is not syncing between the mobile app and the web dashboard.",
            "The API is returning a 503 error for all POST requests since this morning.",
            "After the latest update the dashboard no longer loads on Chrome.",
            "I am unable to upload files larger than 5 MB even though my plan allows 50 MB.",
            "The integration with Slack stopped sending notifications three days ago.",
            "Performance is extremely slow — reports that used to take seconds now time out.",
            "I keep getting a session expired error even though I just logged in.",
            "The search functionality returns no results regardless of the query.",
        ]},
    "Account Management": {
        "priority_weights": [0.30, 0.45, 0.25],
        "texts": [
            "I need to reset my password but the reset email never arrives.",
            "How do I add a new team member to my organisation account?",
            "I want to change the primary email address linked to my account.",
            "Please delete my account and all associated data as per GDPR.",
            "I accidentally deleted my project — is there any way to restore it?",
            "I need to transfer ownership of my account to a colleague.",
            "My account was locked after multiple failed login attempts. Please unlock it.",
            "How do I enable single sign-on for my enterprise account?",
            "I need to download all my data before closing the account.",
            "Can I merge two separate accounts into one?",
        ]},
    "Feature Request": {
        "priority_weights": [0.10, 0.35, 0.55],
        "texts": [
            "It would be really helpful to have a dark mode option in the dashboard.",
            "Please add CSV export functionality to the analytics section.",
            "Can you integrate with Zapier so I can automate my workflows?",
            "A bulk-delete option for old records would save a lot of time.",
            "I would love to see a Kanban board view in addition to the list view.",
            "Could you add support for multiple currencies in the billing section?",
            "It would be great to have scheduled reports sent automatically by email.",
            "Please add a public API endpoint for retrieving user activity logs.",
            "A mobile app for Android would be very useful for my team.",
            "Please consider adding keyboard shortcuts for power users.",
        ]},
    "Shipping & Delivery": {
        "priority_weights": [0.50, 0.35, 0.15],
        "texts": [
            "My order has not arrived after 14 days. The tracking page shows no updates.",
            "I received the wrong item in my shipment. Please arrange a replacement.",
            "The package was marked as delivered but I never received it.",
            "My order was damaged in transit. I need a refund or replacement urgently.",
            "I need to change the delivery address before my order is dispatched.",
            "The estimated delivery date keeps changing. When will my order arrive?",
            "I was not home during delivery and the courier left no card.",
            "Half of my order arrived but the remaining items are missing.",
            "I would like to upgrade to express shipping — is that still possible?",
            "My international shipment has been stuck in customs for a week.",
        ]},
    "General Inquiry": {
        "priority_weights": [0.10, 0.30, 0.60],
        "texts": [
            "What are the differences between your Basic and Pro subscription plans?",
            "Do you offer any student or non-profit discounts?",
            "What is your data privacy and retention policy?",
            "How long does it take to process a refund after approval?",
            "Is your platform compliant with GDPR and CCPA regulations?",
            "Do you provide onboarding assistance for new enterprise customers?",
            "What languages does your customer support team operate in?",
            "Can I try the platform before committing to a paid plan?",
            "What uptime guarantee is included in the SLA?",
            "Are there any tutorials or documentation available for new users?",
        ]},
}

STOPWORDS = set("""a an the is are was were be been being have has had do does
did will would could should may might shall can i me my myself we our ours
ourselves you your yours yourself he him his she her hers it its they them
their theirs what which who this that these those am as at by for from in into
of on or so than to too up very with""".split())

def clean(text):
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return " ".join(t for t in text.split() if t not in STOPWORDS and len(t)>2)

rows = []
for i in range(600):
    cat  = random.choice(CATEGORIES)
    tmpl = TEMPLATES[cat]
    text = random.choice(tmpl["texts"])
    sfx  = random.choice(["  Please help soon.","  Thank you in advance.",
                           "  This is quite urgent.","","  Kindly look into this."])
    prio = random.choices(PRIORITIES, weights=tmpl["priority_weights"])[0]
    rows.append({"ticket_id":f"TKT-{10000+i}","text":text+sfx,
                 "category":cat,"priority":prio})

df = pd.DataFrame(rows)
df["clean"] = df["text"].apply(clean)
df["wc"]    = df["clean"].apply(lambda x: len(x.split()))

le_cat  = LabelEncoder(); le_prio = LabelEncoder()
df["y_cat"]  = le_cat.fit_transform(df["category"])
df["y_prio"] = le_prio.fit_transform(df["priority"])

X = df["clean"]; y_cat = df["y_cat"]; y_prio = df["y_prio"]
X_tr,X_te,yc_tr,yc_te,yp_tr,yp_te = train_test_split(
    X,y_cat,y_prio,test_size=0.2,random_state=42,stratify=y_cat)

MODELS = {
    "Logistic Reg.":  LogisticRegression(max_iter=1000,random_state=42),
    "Linear SVM":     LinearSVC(max_iter=2000,random_state=42),
    "Random Forest":  RandomForestClassifier(n_estimators=200,random_state=42),
    "Naive Bayes":    MultinomialNB(alpha=0.5),
}
results = {}
for name, clf in MODELS.items():
    pipe = Pipeline([("v", TfidfVectorizer(ngram_range=(1,2),max_features=8000,
                                           sublinear_tf=True,min_df=2)),
                     ("c", clf)])
    pipe.fit(X_tr, yc_tr)
    preds = pipe.predict(X_te)
    cv    = cross_val_score(pipe,X,y_cat,cv=5,scoring="accuracy")
    results[name] = {"pipe":pipe,"preds":preds,
                     "acc":accuracy_score(yc_te,preds),
                     "f1": f1_score(yc_te,preds,average="weighted"),
                     "cv_mean":cv.mean(),"cv_std":cv.std()}

best_name = max(results, key=lambda k: results[k]["f1"])
best_pipe  = results[best_name]["pipe"]
best_preds = results[best_name]["preds"]

prio_pipe = Pipeline([("v",TfidfVectorizer(ngram_range=(1,2),max_features=8000,
                                            sublinear_tf=True,min_df=2)),
                      ("c",LogisticRegression(max_iter=1000,random_state=42))])
prio_pipe.fit(X_tr,yp_tr)
prio_preds = prio_pipe.predict(X_te)

print("Models trained.")
for n,r in results.items():
    print(f"  {n:18s}  acc={r['acc']:.3f}  f1={r['f1']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
#  VIZ 1 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(17,7.5), facecolor=P["bg"])
title_bar(fig, "Dataset Overview")
gs = GridSpec(1,3,figure=fig,left=0.06,right=0.96,top=0.82,bottom=0.10,wspace=0.38)

# -- Category bar
ax = fig.add_subplot(gs[0])
cats = df["category"].value_counts().sort_values()
y_pos = np.arange(len(cats))
for i,(cat,val) in enumerate(cats.items()):
    ci = CATEGORIES.index(cat) % len(CAT_COLORS)
    ax.barh(y_pos[i], val, height=0.55, color=CAT_COLORS[ci],
            edgecolor="white", linewidth=1.2)
    ax.text(val+2, y_pos[i], str(val), va="center", fontsize=9, color=P["muted"])
ax.set_yticks(y_pos); ax.set_yticklabels(cats.index, fontsize=8)
ax.set_xlim(0, cats.max()*1.2)
ax.set_title("Tickets by Category", fontsize=11, fontweight="bold",
             color=P["text"], pad=10)
ax.set_xlabel("Count")
ax.grid(True, axis="x", color=P["stroke"], lw=0.8)
ax.set_facecolor(P["card"])
for sp in ax.spines.values(): sp.set_color(P["stroke"])

# -- Priority donut
ax2 = fig.add_subplot(gs[1])
prios = df["priority"].value_counts().reindex(PRIORITIES)
colors_p = [PRIO_COLORS[p] for p in prios.index]
wedges, _, autotexts = ax2.pie(
    prios.values, colors=colors_p,
    autopct="%1.0f%%", startangle=90, pctdistance=0.72,
    wedgeprops=dict(edgecolor="white", linewidth=2.5, width=0.5))
for at in autotexts:
    at.set_fontsize(10); at.set_color(P["text"]); at.set_fontweight("bold")
# centre label
ax2.text(0,0,"Priority\nSplit", ha="center", va="center",
         fontsize=9, color=P["muted"], fontstyle="italic")
legend_patches = [mpatches.Patch(color=PRIO_COLORS[p],label=p) for p in PRIORITIES]
ax2.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5,-0.12),
           ncol=3, fontsize=8.5, frameon=False)
ax2.set_title("Priority Distribution", fontsize=11, fontweight="bold",
              color=P["text"], pad=10)
ax2.set_facecolor(P["bg"])

# -- Heatmap
ax3 = fig.add_subplot(gs[2])
pivot = df.groupby(["category","priority"]).size().unstack(fill_value=0)[PRIORITIES]
short = [c.replace(" & ","\n& ").replace(" / ","\n/ ") for c in pivot.index]
sns.heatmap(pivot, ax=ax3,
            cmap=sns.light_palette("#B8A9F5", as_cmap=True),
            annot=True, fmt="d", linewidths=1.5, linecolor=P["bg"],
            cbar_kws={"shrink":0.7,"pad":0.02},
            annot_kws={"size":9,"color":P["text"],"fontweight":"bold"})
ax3.set_xticklabels(PRIORITIES, fontsize=9)
ax3.set_yticklabels(short, fontsize=7, rotation=0)
ax3.set_xlabel("Priority", fontsize=9); ax3.set_ylabel("")
ax3.set_title("Priority × Category", fontsize=11, fontweight="bold",
              color=P["text"], pad=10)
ax3.tick_params(length=0)
cbar = ax3.collections[0].colorbar
cbar.ax.tick_params(labelsize=7, colors=P["muted"])

save(fig, "01_dataset_overview.png")

# ══════════════════════════════════════════════════════════════════════════════
#  VIZ 2 — TEXT STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1,2,figsize=(14,5.5), facecolor=P["bg"])
fig.subplots_adjust(left=0.08,right=0.95,top=0.82,bottom=0.12,wspace=0.35)
title_bar(fig,"Text Statistics After Cleaning")

# Word count histogram
ax = axes[0]
n, bins, patches = ax.hist(df["wc"], bins=22,
                            color=P["c1"], edgecolor="white", linewidth=0.8, alpha=0.9)
for patch in patches:
    patch.set_facecolor(P["c1"])
mean_wc = df["wc"].mean()
ax.axvline(mean_wc, color=P["accent"], lw=2, linestyle="--", label=f"Mean = {mean_wc:.1f} words")
ax.fill_betweenx([0,n.max()*1.1],[mean_wc-0.4],[mean_wc+0.4],
                  color=P["accent"], alpha=0.12)
ax.set_title("Word Count Distribution", fontsize=11, fontweight="bold", color=P["text"], pad=10)
ax.set_xlabel("Words per Ticket"); ax.set_ylabel("Frequency")
ax.legend(fontsize=9, frameon=False)
ax.set_facecolor(P["card"])
ax.set_ylim(0, n.max()*1.15)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
for sp in ["left","bottom"]: ax.spines[sp].set_color(P["stroke"])

# Avg word count per category
ax2 = axes[1]
wc_cat = df.groupby("category")["wc"].mean().sort_values()
for i,(cat,v) in enumerate(wc_cat.items()):
    ci = CATEGORIES.index(cat) % len(CAT_COLORS)
    ax2.barh(i, v, height=0.52, color=CAT_COLORS[ci],
             edgecolor="white", linewidth=1.0)
    ax2.text(v+0.08, i, f"{v:.1f}", va="center", fontsize=9, color=P["muted"])
ax2.set_yticks(range(len(wc_cat))); ax2.set_yticklabels(wc_cat.index, fontsize=8)
ax2.set_xlim(0, wc_cat.max()*1.22)
ax2.set_title("Avg Words by Category", fontsize=11, fontweight="bold", color=P["text"], pad=10)
ax2.set_xlabel("Average Word Count")
ax2.set_facecolor(P["card"])
for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
for sp in ["left","bottom"]: ax2.spines[sp].set_color(P["stroke"])

save(fig, "02_text_statistics.png")

# ══════════════════════════════════════════════════════════════════════════════
#  VIZ 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1,2,figsize=(15,6), facecolor=P["bg"])
fig.subplots_adjust(left=0.06,right=0.96,top=0.82,bottom=0.14,wspace=0.38)
title_bar(fig,"Model Performance Comparison — Category Classification")

names = list(results.keys())
x = np.arange(len(names)); w = 0.34

# Accuracy + F1
ax = axes[0]
acc_v = [results[m]["acc"]*100 for m in names]
f1_v  = [results[m]["f1"]*100  for m in names]
b1 = ax.bar(x-w/2, acc_v, w, color=P["c1"], edgecolor="white", lw=1.2, label="Accuracy")
b2 = ax.bar(x+w/2, f1_v,  w, color=P["c2"], edgecolor="white", lw=1.2, label="Weighted F1")
for b,v in list(zip(b1,acc_v))+list(zip(b2,f1_v)):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4,
            f"{v:.1f}%", ha="center", fontsize=8, color=P["muted"], fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5, rotation=10, ha="right")
ax.set_ylim(0, 112); ax.set_ylabel("Score (%)")
ax.set_title("Test Accuracy & F1 Score", fontsize=11, fontweight="bold", color=P["text"])
ax.legend(fontsize=9, frameon=False)
ax.set_facecolor(P["card"])
ax.axhline(100, color=P["stroke"], lw=1, zorder=0)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
for sp in ["left","bottom"]: ax.spines[sp].set_color(P["stroke"])
ax.grid(True, axis="y", color=P["stroke"], lw=0.7)
ax.set_axisbelow(True)

# CV bars with error
ax2 = axes[1]
cv_v = [results[m]["cv_mean"]*100 for m in names]
cv_e = [results[m]["cv_std"]*100  for m in names]
bars = ax2.bar(x, cv_v, 0.5, color=P["c4"], edgecolor="white", lw=1.2,
               yerr=cv_e, error_kw=dict(ecolor=P["accent"],capsize=5,lw=1.8,capthick=1.8))
for b,v,e in zip(bars,cv_v,cv_e):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+e+0.5,
             f"{v:.1f}%", ha="center", fontsize=8, color=P["muted"], fontweight="bold")
ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=8.5, rotation=10, ha="right")
ax2.set_ylim(0, 114); ax2.set_ylabel("CV Accuracy (%)")
ax2.set_title("5-Fold Cross-Validation", fontsize=11, fontweight="bold", color=P["text"])
ax2.set_facecolor(P["card"])
for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
for sp in ["left","bottom"]: ax2.spines[sp].set_color(P["stroke"])
ax2.grid(True, axis="y", color=P["stroke"], lw=0.7)
ax2.set_axisbelow(True)

save(fig, "03_model_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
#  VIZ 4 — CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
cm = confusion_matrix(yc_te, best_preds)
fig, ax = plt.subplots(figsize=(10,8), facecolor=P["bg"])
fig.subplots_adjust(left=0.22, right=0.92, top=0.88, bottom=0.18)
title_bar(fig, f"Confusion Matrix — {best_name}")

labels = [c.replace(" & ","\n& ").replace(" / ","\n/ ") for c in le_cat.classes_]
cmap = sns.light_palette(P["accent"], as_cmap=True)
sns.heatmap(cm, ax=ax, annot=True, fmt="d", cmap=cmap,
            xticklabels=labels, yticklabels=labels,
            linewidths=2, linecolor=P["bg"],
            cbar_kws={"shrink":0.75,"pad":0.03},
            annot_kws={"size":13,"fontweight":"bold","color":P["text"]})
ax.set_xlabel("Predicted Category", fontsize=10, labelpad=12)
ax.set_ylabel("True Category",      fontsize=10, labelpad=12)
ax.tick_params(axis="both", length=0, labelsize=8)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=7, colors=P["muted"])

# highlight diagonal
for i in range(len(le_cat.classes_)):
    ax.add_patch(FancyBboxPatch((i+0.05, i+0.05), 0.9, 0.9,
                                boxstyle="round,pad=0.02",
                                facecolor="none", edgecolor=P["accent"],
                                linewidth=2, zorder=3))

save(fig, "04_confusion_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
#  VIZ 5 — PER-CLASS METRICS
# ══════════════════════════════════════════════════════════════════════════════
rep = classification_report(yc_te, best_preds,
                             target_names=le_cat.classes_, output_dict=True)
mdf = pd.DataFrame(rep).T.iloc[:-3][["precision","recall","f1-score"]]

fig, ax = plt.subplots(figsize=(13,5.5), facecolor=P["bg"])
fig.subplots_adjust(left=0.06,right=0.96,top=0.82,bottom=0.18)
title_bar(fig,f"Per-Class Metrics — {best_name}")

x2 = np.arange(len(mdf)); w2 = 0.26
b1 = ax.bar(x2-w2,   mdf["precision"]*100, w2, color=P["c1"], edgecolor="white", lw=1, label="Precision")
b2 = ax.bar(x2,       mdf["recall"]*100,    w2, color=P["c2"], edgecolor="white", lw=1, label="Recall")
b3 = ax.bar(x2+w2,   mdf["f1-score"]*100,  w2, color=P["c4"], edgecolor="white", lw=1, label="F1-Score")
for bars in [b1,b2,b3]:
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                f"{b.get_height():.0f}", ha="center", fontsize=7.5, color=P["muted"])

ax.set_xticks(x2); ax.set_xticklabels(mdf.index, rotation=12, ha="right", fontsize=8.5)
ax.set_ylim(0, 115); ax.set_ylabel("Score (%)")
ax.legend(fontsize=9.5, frameon=False, loc="lower right")
ax.set_facecolor(P["card"])
ax.axhline(80, color=P["sub"], lw=1.2, linestyle=":", alpha=0.8)
ax.text(len(mdf)-0.5, 81.5, "80% threshold", fontsize=8, color=P["sub"], ha="right")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
for sp in ["left","bottom"]: ax.spines[sp].set_color(P["stroke"])
ax.grid(True, axis="y", color=P["stroke"], lw=0.7); ax.set_axisbelow(True)

save(fig, "05_per_class_metrics.png")

# ══════════════════════════════════════════════════════════════════════════════
#  VIZ 6 — PRIORITY RESULTS
# ══════════════════════════════════════════════════════════════════════════════
prio_labels = list(le_prio.classes_)
pm  = confusion_matrix(yp_te, prio_preds)
fig, axes = plt.subplots(1,2,figsize=(14,5.5), facecolor=P["bg"])
fig.subplots_adjust(left=0.06,right=0.96,top=0.82,bottom=0.12,wspace=0.38)
title_bar(fig,"Priority Classification Results")

cmap_p = sns.light_palette(P["high"], as_cmap=True)
sns.heatmap(pm, ax=axes[0], annot=True, fmt="d", cmap=cmap_p,
            xticklabels=prio_labels, yticklabels=prio_labels,
            linewidths=2, linecolor=P["bg"],
            annot_kws={"size":14,"fontweight":"bold","color":P["text"]})
axes[0].set_title("Priority Confusion Matrix", fontsize=11, fontweight="bold", color=P["text"])
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
axes[0].tick_params(length=0)
for i in range(3):
    axes[0].add_patch(FancyBboxPatch((i+0.06,i+0.06),0.88,0.88,
                                     boxstyle="round,pad=0.02",facecolor="none",
                                     edgecolor=P["high"],linewidth=1.8,zorder=3))

pr_rep = classification_report(yp_te,prio_preds,target_names=prio_labels,output_dict=True)
pr_df  = pd.DataFrame(pr_rep).T.iloc[:-3][["precision","recall","f1-score"]]
x3 = np.arange(len(pr_df)); w3 = 0.28
axes[1].bar(x3-w3, pr_df["precision"]*100, w3, color=P["c5"], edgecolor="white",lw=1,label="Precision")
axes[1].bar(x3,    pr_df["recall"]*100,    w3, color=P["c3"], edgecolor="white",lw=1,label="Recall")
axes[1].bar(x3+w3, pr_df["f1-score"]*100,  w3, color=P["c4"], edgecolor="white",lw=1,label="F1-Score")
axes[1].set_xticks(x3); axes[1].set_xticklabels(pr_df.index, fontsize=10)
axes[1].set_ylim(0,105); axes[1].set_ylabel("Score (%)")
axes[1].set_title("Per-Priority Metrics", fontsize=11, fontweight="bold", color=P["text"])
axes[1].legend(fontsize=9, frameon=False)
axes[1].set_facecolor(P["card"])
for sp in ["top","right"]: axes[1].spines[sp].set_visible(False)
for sp in ["left","bottom"]: axes[1].spines[sp].set_color(P["stroke"])
axes[1].grid(True,axis="y",color=P["stroke"],lw=0.7); axes[1].set_axisbelow(True)

save(fig, "06_priority_results.png")

# ══════════════════════════════════════════════════════════════════════════════
#  VIZ 7 — LIVE DEMO (card layout)
# ══════════════════════════════════════════════════════════════════════════════
DEMO = [
    "I was charged twice for my subscription. Please refund immediately.",
    "The API is returning 503 errors for all POST requests since this morning.",
    "My order has not arrived after 14 days. The tracking shows no updates.",
    "What are the differences between Basic and Pro plans?",
    "It would be great to have a dark mode option in the dashboard.",
    "I need to reset my password but the reset email never arrives.",
]

def predict(text):
    c = clean(text)
    cat  = le_cat.inverse_transform(best_pipe.predict([c]))[0]
    prio = le_prio.inverse_transform(prio_pipe.predict([c]))[0]
    return cat, prio

fig = plt.figure(figsize=(16, 9), facecolor=P["bg"])
title_bar(fig, "Live Prediction Showcase")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.set_facecolor(P["bg"])

# column headers
hdrs = ["#", "Ticket Text", "Category", "Priority"]
hx   = [0.05, 0.10, 0.72, 0.87]
hy   = 0.76
for h, x in zip(hdrs, hx):
    ax.text(x, hy, h, fontsize=9.5, fontweight="bold",
            color=P["muted"], transform=ax.transAxes, va="center")

# header underline
ax.plot([0.04, 0.96], [hy-0.025, hy-0.025], color=P["stroke"], lw=1.2,
        transform=ax.transAxes, clip_on=False)

for i, ticket in enumerate(DEMO):
    cat, prio = predict(ticket)
    y = hy - 0.09 - i*0.098

    # alternating row bg
    row_col = P["card"] if i % 2 == 0 else P["card2"]
    rect = FancyBboxPatch((0.04, y-0.038), 0.92, 0.076,
                          boxstyle="round,pad=0.008",
                          facecolor=row_col, edgecolor=P["stroke"],
                          linewidth=0.8, transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)

    # row number circle
    ci = Circle((0.065, y), 0.016, facecolor=P["c6"],
                edgecolor="none", transform=ax.transAxes, clip_on=False)
    ax.add_patch(ci)
    ax.text(0.065, y, str(i+1), ha="center", va="center",
            fontsize=8, fontweight="bold", color=P["accent"],
            transform=ax.transAxes)

    # ticket text
    disp = (ticket[:68]+"…") if len(ticket)>68 else ticket
    ax.text(0.10, y, disp, fontsize=8.5, color=P["text"],
            transform=ax.transAxes, va="center")

    # category pill
    cat_idx = CATEGORIES.index(cat) % len(CAT_COLORS)
    cat_col = CAT_COLORS[cat_idx]
    r1 = FancyBboxPatch((0.70, y-0.025), 0.155, 0.05,
                        boxstyle="round,pad=0.005",
                        facecolor=cat_col, edgecolor="none",
                        transform=ax.transAxes, clip_on=False)
    ax.add_patch(r1)
    ax.text(0.778, y, cat, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=P["text"],
            transform=ax.transAxes)

    # priority pill
    p_col = PRIO_COLORS[prio]
    r2 = FancyBboxPatch((0.865, y-0.025), 0.072, 0.05,
                        boxstyle="round,pad=0.005",
                        facecolor=p_col, edgecolor="none",
                        transform=ax.transAxes, clip_on=False)
    ax.add_patch(r2)
    ax.text(0.901, y, prio, ha="center", va="center",
            fontsize=8, fontweight="bold", color=P["text"],
            transform=ax.transAxes)

# legend row
lx = 0.06; ly = 0.07
ax.text(lx, ly+0.01, "Priority:", fontsize=8, color=P["muted"],
        transform=ax.transAxes, fontweight="bold")
for j,(prio,col) in enumerate(PRIO_COLORS.items()):
    rx = lx + 0.08 + j*0.085
    rp = FancyBboxPatch((rx, ly-0.01), 0.065, 0.032,
                        boxstyle="round,pad=0.004",
                        facecolor=col, edgecolor="none",
                        transform=ax.transAxes, clip_on=False)
    ax.add_patch(rp)
    ax.text(rx+0.033, ly+0.006, prio, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=P["text"],
            transform=ax.transAxes)

save(fig, "07_live_demo.png")

# ══════════════════════════════════════════════════════════════════════════════
#  PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*55)
print(f"  Best model: {best_name}")
print(f"  Category accuracy : {results[best_name]['acc']:.4f}")
print(f"  Category F1       : {results[best_name]['f1']:.4f}")
print(f"  Priority accuracy : {accuracy_score(yp_te, prio_preds):.4f}")
print("─"*55)
print(classification_report(yc_te, best_preds, target_names=le_cat.classes_))
print("✅  FUTURE_ML_02 — All outputs saved.")
