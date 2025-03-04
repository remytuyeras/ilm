import os
import pandas as pd

df = pd.read_parquet("hf://datasets/garage-bAInd/Open-Platypus/data/train-00000-of-00001-4fe2df04669d1669.parquet")

file_path = os.path.join("data/training_input.txt")

with open(file_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(row["instruction"] + "\n")
        f.write(row["output"] + "\n")

print(f"File saved at: {file_path}")
