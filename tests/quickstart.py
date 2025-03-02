import sys
sys.path.insert(1, "./")
from tokenizer.intuit import create_tokenizer, load_tokenizer

option = 2
if option == 1:
    tokenizer, detokenizer = create_tokenizer(source_file="data/training_input.txt", target_file="data/tokenizer_v1.json")
elif option == 2:
    tokenizer, detokenizer = load_tokenizer("data/tokenizer_v1.json")

line_index = 20
with open("data/training_input.txt", "r", encoding="utf-8") as file:
    for index, line in enumerate(file):
        if index == line_index:
            sample_line = line
            break

tokens = tokenizer(sample_line)
print(tokens)
print(detokenizer(tokens))