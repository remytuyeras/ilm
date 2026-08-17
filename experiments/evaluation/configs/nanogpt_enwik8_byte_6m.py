# Controlled byte-level nanoGPT baseline for the canonical enwik8 split.

out_dir = '../../experiments/evaluation/runs/enwik8_byte_gpt_6m_seed13'
dataset = 'ilm_enwik8_byte'
seed = 13

eval_interval = 200
eval_iters = 200
log_interval = 20
always_save_checkpoint = True
wandb_log = False

gradient_accumulation_steps = 1
batch_size = 32
block_size = 74

n_layer = 6
n_head = 6
n_embd = 300
dropout = 0.5
bias = False

learning_rate = 1e-3
max_iters = 6000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
decay_lr = True
warmup_iters = 100
lr_decay_iters = 6000
min_lr = 1e-4

device = 'mps'
dtype = 'float32'
compile = False
