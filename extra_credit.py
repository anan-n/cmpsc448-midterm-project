import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

DATA = Path("data/six_llms.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA).dropna(subset=["input", "output", "label"]).copy()
df["input"] = df["input"].astype(str)
df["output"] = df["output"].astype(str)
df["label"] = df["label"].astype(str)

# ---------- RQ4: measurable response characteristics ----------
def words(text):
    return re.findall(r"\b[\w']+\b", str(text).lower())

def features(text):
    s = str(text)
    w = words(s)
    sentences = [x for x in re.split(r"[.!?]+", s) if x.strip()]
    return pd.Series({
        "word_count": len(w),
        "lexical_diversity": len(set(w)) / len(w) if w else 0,
        "mean_word_length": np.mean([len(x) for x in w]) if w else 0,
        "sentence_count": len(sentences),
        "bullet_marker": int(bool(re.search(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", s))),
        "code_marker": int("```" in s)
    })

print("Calculating RQ4 response features...")
feat = df["output"].apply(features)
df = pd.concat([df, feat], axis=1)

summary = df.groupby("label").agg(
    mean_words=("word_count", "mean"),
    median_words=("word_count", "median"),
    mean_lexical_diversity=("lexical_diversity", "mean"),
    mean_word_length=("mean_word_length", "mean"),
    mean_sentences=("sentence_count", "mean"),
    pct_with_bullets=("bullet_marker", lambda x: 100 * x.mean()),
    pct_with_code_blocks=("code_marker", lambda x: 100 * x.mean())
).round(3)
summary.to_csv(OUT / "rq4_feature_summary.csv")
print("\nRQ4 summary (saved to outputs/rq4_feature_summary.csv):")
print(summary.to_string())

stats_rows = []
for column in ["word_count", "lexical_diversity", "mean_word_length", "sentence_count"]:
    groups = [df.loc[df.label == label, column].to_numpy()
              for label in sorted(df.label.unique())]
    h, p = kruskal(*groups)
    stats_rows.append({"feature": column, "kruskal_H": h, "p_value": p})
stats = pd.DataFrame(stats_rows)
stats.to_csv(OUT / "rq4_statistical_tests.csv", index=False)
print("\nRQ4 Kruskal-Wallis tests (descriptive evidence of differences, not proof of causation):")
print(stats.to_string(index=False))

# ---------- RQ3: cross-domain generalization ----------
# Transparent keyword rules assign each question a broad subject domain.
# These are heuristic labels, not ground-truth annotations.
DOMAIN_RULES = {
    "computer_science": [
        "array", "linked list", "binary search", "binary tree", "data structure",
        "algorithm", "recursion", "programming", "software", "database",
        "computer network", "operating system", "cybersecurity", "machine learning",
        "artificial intelligence", "python", "java code", "sorting", "hash table",
        "stack", "queue", "compiler", "computer science"
    ],
    "math_statistics": [
        "calculus", "algebra", "geometry", "equation", "probability", "statistics",
        "mean", "median", "variance", "standard deviation", "theorem", "integral",
        "derivative", "matrix", "mathematical", "permutation", "combination"
    ],
    "engineering_physics": [
        "ohm's law", "newton", "voltage", "current", "circuit", "electric",
        "engineering", "force", "motion", "thermodynamics", "nuclear",
        "robotics", "energy", "magnet", "resistance"
    ],
    "life_health": [
        "biology", "plant", "animal", "zoology", "agriculture", "pharmaceutics",
        "medicine", "medical", "disease", "human body", "health", "anatomy",
        "physiology", "biology", "ecosystem", "cell"
    ],
    "business_economics": [
        "microeconomics", "macroeconomics", "economics", "economic", "management",
        "marketing", "finance", "accounting", "business", "entrepreneur",
        "supply chain", "organization"
    ],
    "social_sciences": [
        "sociology", "constitution", "criminology", "geography", "international relations",
        "political science", "psychology", "government", "law", "gender",
        "mass communication", "disaster", "society", "social science"
    ],
    "humanities_arts": [
        "parts of speech", "literature", "civilization", "graphic design", "philosophy",
        "television", "film", "theatre", "theater", "tourism", "history", "art",
        "music", "language", "poetry", "religion", "culture"
    ]
}

def assign_domain(question):
    q = str(question).lower()
    for domain, keywords in DOMAIN_RULES.items():
        if any(k in q for k in keywords):
            return domain
    return "other"

df["domain"] = df["input"].map(assign_domain)
df["question_group"] = df["input"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()

domain_counts = df.groupby(["domain", "label"]).size().unstack(fill_value=0)
domain_counts.to_csv(OUT / "rq3_domain_counts.csv")
print("\nRQ3 examples per domain and LLM (saved to outputs/rq3_domain_counts.csv):")
print(domain_counts.to_string())

heldout = "computer_science"
ood = df[df.domain == heldout].copy()
source = df[df.domain != heldout].copy()

if ood["question_group"].nunique() < 10:
    raise ValueError(
        "Too few held-out computer-science questions were found. "
        "Inspect outputs/rq3_domain_counts.csv and adjust domain keywords."
    )
if source["question_group"].nunique() < 30:
    raise ValueError("Too few non-computer-science questions for a useful split.")

# Split by question, not by row, so the same prompt cannot occur in train and test.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_pos, in_test_pos = next(splitter.split(
    source, groups=source["question_group"]
))
train = source.iloc[train_pos]
in_test = source.iloc[in_test_pos]

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True, strip_accents="unicode",
        ngram_range=(1, 2), max_features=100000,
        min_df=2, sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ))
])

print("\nTraining RQ3 TF-IDF + Logistic Regression on non-CS domains...")
model.fit(train["output"], train["label"])

def evaluate(name, test):
    pred = model.predict(test["output"])
    row = {
        "evaluation": name,
        "n_examples": len(test),
        "n_unique_questions": test["question_group"].nunique(),
        "accuracy": accuracy_score(test["label"], pred),
        "macro_f1": f1_score(test["label"], pred, average="macro", zero_division=0)
    }
    print(f"\n{name}: accuracy={row['accuracy']:.4f}, macro-F1={row['macro_f1']:.4f}, "
          f"n={row['n_examples']}")
    print(classification_report(test["label"], pred, zero_division=0))
    return row

results = [
    evaluate("in_domain_non_CS_test", in_test),
    evaluate("held_out_computer_science", ood)
]
pd.DataFrame(results).to_csv(OUT / "rq3_domain_generalization.csv", index=False)
print("\nRQ3 comparison saved to outputs/rq3_domain_generalization.csv")
print("\nFinished. Review the domain counts before interpreting RQ3.")
