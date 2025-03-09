import os
import sys
sys.path.insert(1, "./")
import ilm
import torch
from torch.nn import functional as F

# =========== Tokenizer =========== #
tokenizer, detokenizer = ilm.load_tokenizer("data/tokenizer_v1.json")

# =========== Get model =========== #
# torch.manual_seed(1234)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

ilmodel = ilm.IntuinisticLanguageModel(vocab_size=64, device=device)

if "--load" not in sys.argv[1:]:
    # ----- Tokenize dataset and create batches ----- #
    with open("data/training_input.txt","r") as f:
        raw_text = f.read()

    manager = ilm.TrainingManager(raw_text, tokenizer, device=device)

    manager.batch_size = 32
    manager.block_size = 20 * 3

    # ----- Create model ----- #
    ilmodel.train_model(manager, epoch_num=100000)
    os.makedirs("models", exist_ok=True)
    ilmodel.save_model(model_path = "models/iml_model_v0.0.1.pth")

else:
    ilmodel.load_model(model_path = "models/iml_model_v0.0.1.pth")
    
# =========== Use model =========== #
added_words = 50
while True:
    string = input(">>> ")
    if string == "!exit":
        break
    success = False
    while not(success):
        try:
            single_context = ilm.format_context(string, tokenizer=tokenizer).unsqueeze(0) # turn (T, ) to (1, T)
            success = True
        except:
            print("Language not recongized!")
            string = input(">>> ")
            
        
    generated_tokens = ilmodel.generate(single_context, 
                                        max_new_tokens=3*added_words,
                                        temperature=0
                                        ).detach().cpu()[0].tolist() # turn (1, T) to #list=T
    # print("[GEN]", generated_tokens)
    out = ilm.gather_tokens(generated_tokens, syllable_num=3)
    print("".join([str(x) for x in detokenizer(out)]).replace(string, ">>> ", 1))


