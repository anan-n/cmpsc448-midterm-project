import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer

DATA = Path("data/six_llms.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA).dropna(subset=["input", "output", "label"]).copy()
df["input"] = df["input"].astype(str)
df["output"] = df["output"].astype(str)
df["label"] = df["label"].astype(str)

def words(text):
    return re.findall(r"\b[\w']+\b", str(text).lower())

def extract_features(text):
    s = str(text)
    w = words(s)
    sentences = [x for x in re.split(r"[.!?]+", s) if x.strip()]

    return {
        "word_count": len(w),
        "lexical_diversity": len(set(w)) / len(w) if w else 0,
        "mean_word_length": np.mean([len(x) for x in w]) if w else 0,
        "sentence_count": len(sentences),
        "bullet_marker": int(bool(re.search(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", s))),
        "code_marker": int("```" in s),
    }

print("Extracting response features...")
feature_df = pd.DataFrame(
    df["output"].apply(extract_features).tolist(),
    index=df.index
)
df = pd.concat([df, feature_df], axis=1)

# Group by normalized question to prevent prompt leakage.
df["question_group"] = (
    df["input"].str.lower()
    .str.replace(r"\W+", " ", regex=True)
    .str.strip()
)

# Use the same 80/20 grouped split approach as RQ3.
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_idx, test_idx = next(
    splitter.split(df, groups=df["question_group"])
)

train = df.iloc[train_idx]
test = df.iloc[test_idx]

feature_columns = [
    "word_count",
    "lexical_diversity",
    "mean_word_length",
    "sentence_count",
    "bullet_marker",
    "code_marker",
]

X_train = train[feature_columns]
X_test = test[feature_columns]
y_train = train["label"]
y_test = test["label"]

# Standardize numerical features before training.
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
)

print("Training feature-based classifier...")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro")

from sklearn.metrics import classification_report

print("\nPer-class performance:")
print(classification_report(y_test, predictions, digits=3))

results = pd.DataFrame([{
    "model": "LogisticRegression_handcrafted_features",
    "n_train": len(train),
    "n_test": len(test),
    "n_test_questions": test["question_group"].nunique(),
    "accuracy": accuracy,
    "macro_f1": macro_f1,
}])

results.to_csv(OUT / "rq4_feature_classifier.csv", index=False)

print("\nRQ4 Feature-Based Classifier Results")
print(results.to_string(index=False))
print("\nSaved results to outputs/rq4_feature_classifier.csv")
