import pandas as pd
from pathlib import Path

INPUT_FILE = "TXD-22 A Large-Scale Benchmark Dataset.csv"
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

LLMS = ["ChatGPT", "Claude", "Gemini", "DeepSeek", "Llama", "Qwen"]
N_PER_LLM = 2000

print("Reading dataset...")
df = pd.read_csv(
    INPUT_FILE,
    encoding="latin1",
    encoding_errors="replace",
    usecols=["Question", "Answer", "Source"]
)

# Normalize labels and remove incomplete examples.
df["Source"] = df["Source"].astype(str).str.strip()
df["Question"] = df["Question"].fillna("").astype(str).str.strip()
df["Answer"] = df["Answer"].fillna("").astype(str).str.strip()

df = df[df["Source"].isin(LLMS)].copy()
df = df[(df["Question"] != "") & (df["Answer"] != "")]

print("\nUsable examples per model before sampling:")
print(df["Source"].value_counts().reindex(LLMS, fill_value=0))

counts = df["Source"].value_counts()
if any(counts.get(model, 0) < N_PER_LLM for model in LLMS):
    raise ValueError("At least one LLM has fewer than 2000 usable examples.")

# Balanced, reproducible sample.
df = (
    df.groupby("Source", group_keys=False)
      .sample(n=N_PER_LLM, random_state=42)
      .sample(frac=1, random_state=42)
      .reset_index(drop=True)
)

df = df.rename(columns={
    "Question": "input",
    "Answer": "output",
    "Source": "label"
})

df.to_csv(OUT_DIR / "six_llms.csv", index=False)

print("\nSaved:", OUT_DIR / "six_llms.csv")
print("Total rows:", len(df))
print("\nFinal class counts:")
print(df["label"].value_counts().sort_index())
print("\nMissing values:")
print(df[["input", "output", "label"]].isna().sum())
