---
title: "Documentation"
permalink: /docs/
sidebar:
  title: "Home"
  nav: sidebar-docs  # See /docs/_data/navigation.yml
---

Welcome to the documentation for the `statcan_dialogue_dataset` package! You can find the user guide below. For more advanced usage, please use the sidebar to navigate to go to specific API reference pages.

# User Guide

This user guide will help you get started with the `statcan_dialogue_dataset` package. It will cover most of the common use cases.


## Quickstart

First, you need to request access (please go to the main [webpage](https://mcgill-nlp.github.io/statcan-dialogue-dataset) for more information). 

Then, you can install our Python library to load the dataset:

```python
pip install statcan-dialogue-dataset
```

Now, inside Python, you can use various functions to work with the data:

```python
import statcan_dialogue_dataset as sdd

# Download the dataset (from dataverse)
sdd.download_dataverse("your_api_token")
# Alternatively, Download from huggingface
sdd.download_huggingface("your_api_token")

# Extract ZIP into ~/.statcan_dialogue_dataset/
sdd.extract_task_data_zip(remove_zip=True)

# Load task specific split for a specific language
train_ret = sdd.load_task_data(task="retrieval", split="train", lang="en")

# Load all task conversations
task_conversations = sdd.load_task_conversations()

# Load table metadata
table = sdd.metadata.Table.from_code(10100002)
```

For more information, head to the [Core API Reference](https://mcgill-nlp.github.io/statcan-dialogue-dataset/docs/core/) page.

## Download task data

You can find a link to access the dataset on the [official webpage](https://mcgill-nlp.github.io/statcan-dialogue-dataset). You will need to create an account on the Dataverse platform and request access. We will review each request and give manual approval. When you receive access, you can download the dataset by clicking on `Access Dataset` button on Dataverse and choose to download the "Original Format ZIP".

Alternatively, you can also create an API token (Click on your username on the top right navigation bar, then "API Token"). Then, you can download the dataset with the following command:

```python
import statcan_dialogue_dataset as sdd

saved_path = sdd.download_dataverse(api_token="your_api_token", data_dir="path/to/your/data_dir")
```

If you requested your data from Huggingface instead, you can use the following command:

```python
saved_path = sdd.download_huggingface(api_token="your_api_token")
```

Once you are done, you can extract the files with the following command:

```python
sdd.extract_task_data_zip(saved_path, remove_zip=True)
```

If you do not specify `data_dir`, it will be downloaded to `~/.statcan_dialogue_dataset/`, and extracted from there when you call `sdd.extract_task_data_zip()`. If you do not specify `remove_zip`, the zip file will be kept.

## Using the task data

### Loading task data as dataframes

Once you have downloaded and extracted the dataset, you can access them inside Python:
```python
# Load the retrieval task data in English (without BM25 hard negatives)
train_ret = sdd.load_task_data(task="retrieval", split="train", lang="en", with_hn=False)
# Load the generation task data in French (with DPR retrieved results)
dev_gen = sdd.load_task_data(task="generation", split="dev", lang="fr", with_augmented=True)
```

You will get dataframes containijng relevant columns for the desired task that was defined in the paper. For more information, please refer to the documentations or the paper or use `help(sdd.load_task_data)`.

### Loading conversation data as dictionary

If the CSV format is inconvenient for you, and you wish to use the JSON/dictionary format of the data, you can also load all task conversations as a dictionary:

```python
index_splits = sdd.load_index_splits()
task_conversations = sdd.load_task_conversations()
```

`index_splits` can be used to reconstruct the splits in each of the task splits (but you still need to truncate the conversations to get the turn-level conversation-target pairs). `task_conversations` contains the relevant conversations in the original format (i.e. not preprocessed for a task, and without duplicates) and will have the following structure:

```
{
    <index>: {
        "index": <index>,
        "conversation": [
            {
                "speaker": <'user' or 'agent'>,
                "name": <name of speaker>,
                "timestamp": <timestamp>,
                "urls": [...],
                "text": <the message by the speaker>
            },
            ...
        ],
        "language": {
            'automatic': <language selected by the user in the chat system>,
            'fasttext': {
                'detected': <'en' or 'fr>,
                'confidence': <confidence of the language detection model>
            },
            'langid': {
                'detected': <'en' or 'fr>,
                'confidence': <confidence of the language detection model>
            }
        },
        "urls": [...],
    },
    ...
}
```

For more information, please refer to the API reference in the documentations.

## Table metadata usage

You will find that the target of your retrieval task is a PID (table's product ID). We have provided the `sdd.metadata.Table` class as a convenient way to access the metadata of a table. You can use it as follows:

```python
import statcan_dialogue_dataset as sdd

tab = sdd.metadata.Table.from_code(10100002)
tab
# Table(code='10100002', title='Central government debt', subjects=[...], surveys=[...], frequency=Frequency(code='6', title='Monthly', lang='en'), lang='en', start_date=datetime.date(2009, 4, 1), end_date=datetime.date(2021, 7, 1), release_time=datetime.datetime(2021, 9, 27, 8, 30), archive_info='CURRENT - a cube available to the public and that is current', dimensions=[...], footnotes=None)
```

You can also use `from_title` to load a table's metadata. Other classes like `sdd.metadata.Subject` and `sdd.metadata.Survey` are available, with special methods useful for navigating them. For more information, please refer to the [API reference for the `sdd.metadata` module](https://mcgill-nlp.github.io/statcan-dialogue-dataset/docs/metadata).

## Download full tables

For most of the tasks, the metadata is sufficient; tables only required for TAPAS.

The full tables are separately distributed from the task data, because they can already be downloaded individually on statcan.gc.ca. You can access them without having request access, and are hosted on a separate platform. You can download the full tables by visiting the download page ([English Tables](https://zenodo.org/record/7765406); [French Tables](https://zenodo.org/record/7772918)). You can also run the following command:

```python
import statcan_dialogue_dataset as sdd

sdd.download_full_tables()
sdd.extract_full_tables()
```
