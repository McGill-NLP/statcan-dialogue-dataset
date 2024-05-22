# Retrieval task

This directory contains relevant files for the retrieval task. Specifically, we include code to process the statcan dataset into a retrieval-based dataset.

# Preliminary steps

First, install everything:
```bash
git clone https://github.com/mcGill-NLP/statcan-dialogue-dataset
cd statcan-dialogue-dataset

python -m venv venv
source venv/bin/activate
pip install -e .
pip install -r retrieval_task/requirements.txt
```

Then, download the statcan dataset via the statcan-dialogue-dataset package, you will need your huggingface token as the environment variable `HF_TOKEN`:
```python
import os
import statcan_dialogue_dataset as sdd

# Download task_data.zip from same dataset, using huggingface_hub and unzip
saved_path = sdd.download_huggingface(api_token=os.environ['HF_TOKEN'], data_dir="./")
sdd.extract_task_data_zip(saved_path, remove_zip=True, load_dir='./', data_dir="./retrieval_task/data_original")
```

Run the process module to convert the statcan dataset into a retrieval-based dataset. This will create a `data_out` directory in `retrieval_task`:
```bash
python retrieval_task/process.py
```

To analyze the data:
```bash
python retrieval_task/analyze.py
```

If you want to upload the processed data to huggingface, you can use the following code:
```bash
python retrieval_task/upload_to_hf.py
```