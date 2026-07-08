import os
import sys
sys.path.insert(1, "./")
import ilm
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt

# tokenizer_json = "data/tokenizer_v1.json"
# training_text = "data/training_input.txt"

tokenizer_json = "data/tokenizer_v2.json"
training_text = "data/training_old_english.txt"
        
# =========== Tokenizer =========== #
tokenizer, detokenizer = ilm.load_tokenizer(tokenizer_json)

# =========== Get model =========== #
# torch.manual_seed(1234)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
vocab_size = 64 # vocabulary
block_size = 3 * 20 # time
batch_size = 32 # batch 
embedding_dim = 80 # channels
head_size = 8
layer_num = 6

# dropout = ...
# epoch_num = ...
# lr = ...

dropout = 0.5
epoch_num = 4000
lr = 1e-3

'''
python sandbox/sandbox.py --load models/m1.v0.0.x.pth
python sandbox/sandbox.py --improve models/m1.v0.0.x.pth --patch --major --minor


#m1 model
dropout = 0.5
epoch_num = 1000
lr = 4e-3
-> loss=loss_trn:2.3543, loss_val:2.3831

#0.0.0
dropout = 0.2
epoch_num = 2000
lr = 5e-4
-> loss=loss_trn:2.1278, loss_val:2.1666

#0.0.1
dropout = 0.2
epoch_num = 2000
lr = 5e-4
-> loss=loss_trn:2.0490, loss_val:2.0909 (interesting results)

============
#0.0.2
dropout = 0.4
epoch_num = 1000
lr = 1e-3
-> loss=loss_trn:2.0494, loss_val:2.1018 (bad)

#0.0.3
dropout = 0.2
epoch_num = 2000
lr = 1e-4
-> loss=loss_trn:2.0417, loss_val:2.0860
============

#0.1.0
dropout = 0.2
epoch_num = 2000
lr = 1e-4
-> loss=loss_trn:1.9906, loss_val:2.0386 (okay)

#0.1.1
dropout = 0.4
epoch_num = 1000
lr = 8e-4
-> loss=loss_trn:2.0028, loss_val:2.0559 (repetitive but interesting somehow)

#0.1.2
dropout = 0.3
epoch_num = 5000
lr = 8e-5
-> loss=loss_trn:1.9300, loss_val:1.9770 (okay)

#0.1.3
dropout = 0.4
epoch_num = 10000
lr = 1e-5
-> loss=loss_trn:1.9220, loss_val:1.9592

#0.1.4
dropout = 0.6
epoch_num = 1000
lr = 9e-4
-> loss=loss_trn:1.9500, loss_val:2.0116 (interesting)

#0.1.5
dropout = 0
epoch_num = 4000
lr = 8e-5
-> loss=loss_trn:1.8982, loss_val:1.9553 (very interesting)

#0.1.6
dropout = 0.2
epoch_num = 4000
lr = 8e-5
-> loss=loss_trn:1.8744, loss_val:1.9349

#0.1.7
dropout = 0.4
epoch_num = 1000
lr = 3e-4
-> loss=loss_trn:1.8731, loss_val:1.9443 (interesting)

#1.0.0
dropout = 0
epoch_num = 10000
lr = 8e-5
-> loss=loss_trn:1.8325, loss_val:1.8894 (too many yous)

#1.0.1
dropout = 0.7
epoch_num = 1000
lr = 3e-4
-> loss=loss_trn:1.8245, loss_val:1.9058

#1.0.2
dropout = 0.7
epoch_num = 1000
lr = 3e-4
-> loss=loss_trn:1.8131, loss_val:1.8963

#1.0.3
dropout = 0.2
epoch_num = 4000
lr = 1e-4
-> loss=loss_trn:1.8008, loss_val:1.8813 (quite good)

#1.1.0
dropout = 0.2
epoch_num = 6000
lr = 1e-4
-> loss=loss_trn:1.7713, loss_val:1.8676 (quite interesting)

#1.1.1
dropout = 0.2
epoch_num = 6000
lr = 1e-4
-> loss=loss_trn:1.7580, loss_val:1.8592 (looks bad though)

#1.1.2
dropout = 0.7
epoch_num = 1000
lr = 3e-4
-> loss=loss_trn:1.7712, loss_val:1.8686 (lots of yous)

#1.1.3
dropout = 0.3
epoch_num = 6000
lr = 8e-5
-> loss=loss_trn:1.7412, loss_val:1.8459 (interesting)

#1.1.4
dropout = 0.7
epoch_num = 1000
lr = 3e-4
-> loss=loss_trn:1.7539, loss_val:1.8575 (too many yous)

#1.1.5
dropout = 0.2
epoch_num = 4000
lr = 1e-4
-> loss=loss_trn:1.7436, loss_val:1.8436 (not so great)

#m2.0.0.0
dropout = 0.2
epoch_num = 4000
lr = 1e-4
-> loss=loss_trn:1.9336, loss_val:1.9526

#m2.0.0.1
dropout = 0.2
epoch_num = 4000
lr = 1e-4
-> loss=loss_trn:1.9336, loss_val:1.9526

#m2.0.1.0
dropout = 0.4
epoch_num = 2000
lr = 1e-3
-> loss=loss_trn:1.7735, loss_val:1.8409
-> more diverse, escaped king loop, Shakespeare-like but incoherent

#m2.0.1.1
dropout = 0.4
epoch_num = 2000
lr = 1e-3
-> loss=loss_trn:1.7217, loss_val:1.8091
-> strong Shakespeare formatting, speaker labels, less looping, still grammatically unstable

#m2.0.2.0
dropout = 0.4
epoch_num = 2000
lr = 1e-3
-> loss=loss_trn:1.6905, loss_val:1.7910
-> better loss, but starts repeating known phrase patterns


#m2.0.2.1
dropout = 0.5
epoch_num = 4000
lr = 1e-3
-> loss=loss_trn:1.6870, loss_val:1.7818
-> best loss, but output looks more overtrained/noisy


python sandbox/sandbox.py --load models/m1.v1.1.5.pth

'''


ilmodel = ilm.IntuinisticLanguageModel(vocab_size=vocab_size, 
                                       embedding_dim=embedding_dim,
                                       block_size=block_size,
                                       head_size=head_size,
                                       layer_num=layer_num,
                                       device=device,
                                       dropout=dropout,
                                       )

if "--create" in sys.argv[1:]:
    
    findit = [s for s in sys.argv[1:] if s.endswith(".pth")]
    if findit == []:
        print("No path to model.")
        exit()
        
    model_path = findit[0]

    # ----- Tokenize dataset and create batches ----- #
    with open(training_text,"r") as f:
        raw_text = f.read()

    manager = ilm.TrainingManager(raw_text, tokenizer, device=device)

    manager.batch_size = batch_size
    manager.block_size = block_size

    # ----- Create model ----- #
    ilmodel.train_model(manager,
                        epoch_num=epoch_num,
                        lr=lr
                        )
    
    os.makedirs("models", exist_ok=True)
    ilmodel.save_model(model_path = model_path)

elif "--improve" in sys.argv[1:]:
    
    findit = [s for s in sys.argv[1:] if s.endswith(".pth")]
    if findit == []:
        print("No path to model.")
        exit()
        
    model_to_improve = findit[0]
    ilmodel.load_model(model_path = model_to_improve)

    with open(training_text,"r") as f:
        raw_text = f.read()

    manager = ilm.TrainingManager(raw_text, tokenizer, device=device)

    manager.batch_size = batch_size
    manager.block_size = block_size

    # ----- Create model ----- #
    ilmodel.train_model(manager,
                        epoch_num=epoch_num,
                        lr=lr
                        )
    
    findit = [s for s in sys.argv[1:] if s in ["--major", "--minor", "--patch"]]
    if findit == []:
        print("No semantic versioning given. Creating patch.")
        semver = "patch"
    else:
        semver = findit[0].replace("--","")
    improved_model = ilm.increment_version(name=model_to_improve,semver_increment=semver)
    ilmodel.save_model(model_path = improved_model)

elif "--load" in sys.argv[1:]:
    findit = [s for s in sys.argv[1:] if s.endswith(".pth")]
    if findit == []:
        print("No path to model.")
        exit()
        
    model_path = findit[0]

    ilmodel.load_model(model_path = model_path)

print(f"Model has {sum(p.numel() for p in ilmodel.parameters())/1e3}k parameters")
    
# =========== Use model =========== #

ilm.user_interface(ilmodel, tokenizer, detokenizer, completed_words = 100, syllable_num = 3)


