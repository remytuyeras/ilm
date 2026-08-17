# Configurations

Store a named, machine-readable configuration here before beginning a seed
matrix. Do not change a configuration after evaluating its test split. Record
the tokenizer checksum, model seed, generation seed, and exact architecture.

`nanogpt_char_6m.py` is consumed from within `baselines/nanoGPT`. It uses the
project's frozen `ilm_tinyshakespeare` character data and defines the 6M
character baseline before command-line smoke-test overrides.
