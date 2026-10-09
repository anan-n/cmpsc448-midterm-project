import re
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

SEED = 42
EPOCHS = 5
MAX_LEN = 160
MAX_VOCAB = 12000
BATCH_SIZE = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("Device:", device)

df = pd.read_csv("data/six_llms.csv")
labels = sorted(df["label"].unique())
label_to_id = {name: i for i, name in enumerate(labels)}
y = df["label"].map(label_to_id).to_numpy()

# Keep identical questions in the same split to reduce prompt leakage.
def normalize_question(s):
    return re.sub(r"\W+", " ", str(s).lower()).strip()

df["question_group"] = df["input"].map(normalize_question)
unique_questions = df["question_group"].drop_duplicates().to_numpy()

train_groups, temp_groups = train_test_split(
    unique_questions, test_size=0.20, random_state=SEED
)
val_groups, test_groups = train_test_split(
    temp_groups, test_size=0.50, random_state=SEED
)

group = df["question_group"]
train_idx = np.flatnonzero(group.isin(train_groups).to_numpy())
val_idx = np.flatnonzero(group.isin(val_groups).to_numpy())
test_idx = np.flatnonzero(group.isin(test_groups).to_numpy())

print(f"Split rows: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
print("Labels:", labels)

def tokenize(text):
    return re.findall(r"\b\w+\b|[^\w\s]", str(text).lower())

class TextDataset(Dataset):
    def __init__(self, sequences, targets):
        self.x = torch.tensor(sequences, dtype=torch.long)
        self.y = torch.tensor(targets, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(64, 64, k) for k in (3, 4, 5)
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64 * 3, n_classes)
    def forward(self, x):
        z = self.embedding(x).transpose(1, 2)
        pooled = [torch.relu(c(z)).amax(dim=2) for c in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, n_classes)
    def forward(self, x):
        z = self.embedding(x)
        _, (h, _) = self.lstm(z)
        z = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(z))

modes = {
    "input_only": df["input"].tolist(),
    "output_only": df["output"].tolist(),
    "input_plus_output": (
        "Question: " + df["input"].astype(str)
        + " Answer: " + df["output"].astype(str)
    ).tolist()
}

results = []

for mode, texts in modes.items():
    print(f"\n{'='*60}\nMode: {mode}", flush=True)

    # Build vocabulary from training data only.
    counter = Counter()
    for i in train_idx:
        counter.update(tokenize(texts[i]))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in counter.most_common(MAX_VOCAB - 2):
        vocab[token] = len(vocab)

    def encode(text):
        ids = [vocab.get(t, 1) for t in tokenize(text)[:MAX_LEN]]
        ids += [0] * (MAX_LEN - len(ids))
        return ids

    X = np.asarray([encode(t) for t in texts], dtype=np.int64)
    loaders = {
        "train": DataLoader(
            TextDataset(X[train_idx], y[train_idx]),
            batch_size=BATCH_SIZE, shuffle=True
        ),
        "validation": DataLoader(
            TextDataset(X[val_idx], y[val_idx]),
            batch_size=BATCH_SIZE
        ),
        "test": DataLoader(
            TextDataset(X[test_idx], y[test_idx]),
            batch_size=BATCH_SIZE
        )
    }

    for model_name, model_class in [
        ("CNN", CNNClassifier), ("LSTM", LSTMClassifier)
    ]:
        print(f"\nTraining {model_name}...", flush=True)
        model = model_class(len(vocab), len(labels)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        best_val_f1 = -1
        best_state = None

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for xb, yb in loaders["train"]:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            true_val, pred_val = [], []
            with torch.no_grad():
                for xb, yb in loaders["validation"]:
                    logits = model(xb.to(device))
                    true_val.extend(yb.numpy())
                    pred_val.extend(logits.argmax(1).cpu().numpy())

            val_f1 = f1_score(true_val, pred_val, average="macro", zero_division=0)
            print(
                f"  Epoch {epoch+1}/{EPOCHS} "
                f"loss={total_loss/len(loaders['train']):.4f} "
                f"val_macro_f1={val_f1:.4f}",
                flush=True
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        model.load_state_dict(best_state)
        model.eval()
        true_test, pred_test = [], []
        with torch.no_grad():
            for xb, yb in loaders["test"]:
                logits = model(xb.to(device))
                true_test.extend(yb.numpy())
                pred_test.extend(logits.argmax(1).cpu().numpy())

        accuracy = accuracy_score(true_test, pred_test)
        macro_f1 = f1_score(
            true_test, pred_test, average="macro", zero_division=0
        )
        print(f"TEST accuracy={accuracy:.4f}, macro-F1={macro_f1:.4f}")

        results.append({
            "input_mode": mode,
            "model": model_name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "best_validation_macro_f1": best_val_f1
        })

        with open(OUT / f"report_{mode}_{model_name}.txt", "w") as f:
            f.write(classification_report(
                true_test, pred_test, labels=list(range(len(labels))),
                target_names=labels, zero_division=0
            ))

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

pd.DataFrame(results).to_csv(OUT / "model_comparison.csv", index=False)
with open(OUT / "label_mapping.json", "w") as f:
    json.dump(label_to_id, f, indent=2)

print("\nFINAL RESULTS")
print(pd.DataFrame(results).to_string(index=False))
print("\nSaved outputs/model_comparison.csv and per-model reports.")
