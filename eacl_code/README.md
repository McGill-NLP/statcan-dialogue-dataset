# Running paper-related code (`eacl_code`)

Note that the `eacl_code` directory contains the source used in the original manuscript. It is only recommended to use it to reproduce our work, to understand how the data was processed or the baseline were trained. If you intend to just use the dataset, please head to the [website](https://mcgill-nlp.github.io/statcan-dialogue-dataset) for more information.

## Setup

First, clone the project and go inside the project root
```bash
git clone https://github.com/McGill-NLP/statcan-dialogue-dataset
cd statcan-dialogue-dataset
```

Before starting, make sure that `python3` is using Python 3.8. For example, you can do that with `pyenv` or `conda`:

```bash
pyenv install 3.8.13
pyenv global 3.8.13
```

To install all the packages needed, create a virtual environment and run the following commands:

```bash
python3 -m venv venv
source venv/bin/activate  # in Windows, use venv\Scripts\activate.bat

# Install for full reproducibility
pip3 install -r ./eacl_code/requirements-freeze.txt

# alternatively, install for development
pip3 install -r ./eacl_code/requirements-dev.txt

# Install torch-scatter (requires first installing torch)
pip3 install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
```

## Controlling path to data directories with environment variables

The path to the data files can be specified in the command line. Moreover, the default value, as well as in some scripts, you might need to access to some environment variables:

```bash
# Where all data are located, default: "~/.statcan_dialogue_dataset"
export STATCAN_DATA_DIR="path/to/full/data"

# Where the raw data are located, default: "${STATCAN_DATA_DIR}/raw"
export STATCAN_RAW_DATA_DIR="path/to/raw/data"

# Where some of the larger data files are located, default: "${STATCAN_DATA_DIR}/large"
export STATCAN_LARGE_DATA_DIR="path/to/large/data"

# Where the model checkpoints are located, default: "${STATCAN_DATA_DIR}/checkpoints"
export STATCAN_CHECKPOINT_DIR="path/to/checkpoints"

# Where the temporary files are stored, default: "${STATCAN_DATA_DIR}/temp"
export STATCAN_TEMP_DIR="path/to/temp"
```

For example, if you want to set the data directory to be called `data/` inside the project root:
```bash
export STATCAN_DATA_DIR="$(pwd)/data"
```

Make sure that they link to the data you have downloaded.


## `eacl_code/scripts/` - Running preprocessing on the raw data

You can individually run the scripts in `./eacl_code/scripts/` from the project root. Those files are applied directly to the raw data in order to generate the data files relevant to specific tasks. You will likely not need them for training the models reported in the paper, but they are provided for completeness.


## `eacl_code/modeling_scripts/` - Running model training and inference

Everything needs to be run from the project root.

You can reproduce the results of BM25 with the following command:
```
python3 ./eacl_code/modeling_scripts/run_bm25.py
```

Before running retrieval models, you can (re-)generate the training data with the BM25 hard negatives:
```
python3 ./eacl_code/modeling_scripts/add_hard_negatives.py
```

If you want to train TAPAS, you will need to first generate the tokens, which can take a long time due to the size of the tables:
```
python3 ./eacl_code/scripts/create_tapas_tokens.py
```

Then, you can train DPR or TAPAS for the retrieval task with the CLI script `modeling/train_retriever.py`. To see instructions:
```
python3 ./eacl_code/modeling_scripts/train_retriever.py --help
```

With a trained model, you can re-generate the DPR passages cache to use in the demo apps by running:
```
python3 ./eacl_code/modeling_scripts/cache_dpr_passages.py
```

You can create the augmented generation datasets by running, which will take `data/retrieval/{train | valid | test}_en.csv` and generate `data/retrieval/{train | valid | test}_en_augmented.csv`:
```
python3 ./eacl_code/modeling_scripts/augment_generation_data.py
```

The same can be done for french:
```
python3 ./eacl_code/modeling_scripts/augment_generation_data.py \
    --model_ckpt=path/to/checkpoints/statscan-dpr/eager-forest-4035/epoch-29/ \
    --lang=fr \
    --ctx_colname=basic_and_member_fr
```
where you need to replace `path/to/checkpoints` with the path to the DPR model checkpoints.


You can now train the T5 model with the CLI script `modeling/train_generation.py`. To see instructions:
```
python3 ./eacl_code/modeling_scripts/train_generation.py --help
```

To run T5 on the evaluation data (using the `model.generate` function and beam search) with various metrics (BLEU, ROUGE, METEOR, etc.), you can use the CLI script `modeling/evaluate_generation.py`. To see instructions:
```
python3 ./eacl_code/modeling_scripts/evaluate_generation.py --help
```


## `eacl_code/apps` - Running interactive demos and web apps

We created the `apps/dpr_demo.py` for a simple demo of the DPR model.

Prior to running the app, you need to build an index cache (see above). To run an app, simply run from project root (do NOT `cd ./eacl_code/apps` as this will break).

Then, run the app:

```
python3 ./eacl_code/apps/dpr_demo.py
```
