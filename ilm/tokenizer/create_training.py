import os
import sys
import pandas as pd
import requests

if "--platypus" in sys.argv[1:]:
    df = pd.read_parquet("hf://datasets/garage-bAInd/Open-Platypus/data/train-00000-of-00001-4fe2df04669d1669.parquet")

    file_path = os.path.join("data/training_input.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(row["instruction"] + "\n")
            f.write(row["output"] + "\n")

    print(f"File saved at: {file_path}")

if "--shkspr" in sys.argv[1:]:
    
    input_file_path = os.path.join("data/training_old_english.txt")
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)
