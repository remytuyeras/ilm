import sys
sys.path.insert(1, "./")
from ilm.tokenizer.intuit import create_tokenizer, load_tokenizer

# tokenizer_json = "data/tokenizer_v1.json"
# training_text = "data/training_input.txt"

tokenizer_json = "data/tokenizer_v2.json"
training_text = "data/training_old_english.txt"

option = 2
if option == 1:
    tokenizer, detokenizer = create_tokenizer(source_file=training_text, target_file=tokenizer_json)
elif option == 2:
    tokenizer, detokenizer = load_tokenizer(tokenizer_json)

line_index = 20
with open(training_text, "r", encoding="utf-8") as file:
    for index, line in enumerate(file):
        if index == line_index:
            sample_line = line
            break

tokens = tokenizer(sample_line)
print(tokens)
print(detokenizer(tokens))