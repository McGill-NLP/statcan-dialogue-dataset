---
title: Core API Reference
permalink: /docs/core/
---

This page contains the API reference for the core API of the StatCan Dialogue Dataset. It contains many useful functions for working with the dataset (including loading, processing, downloading).

# Reference for `statcan_dialogue_dataset`

The core API for the StatCan Dialogue Dataset. To use this, start with:

```python
import statcan_dialogue_dataset as sdd
```

Then, you can start using the functions in this module.

## `download_dataverse`

```python
sdd.download_dataverse(api_token, server_url="https://borealisdata.ca", persistent_id="doi:10.5683/SP3/NR0BMY", filename="task_data.zip", data_dir=None, overwrite=False)
```

### Description

Download a file from a Dataverse repository. By default, this downloads the
file from the Borealis Data repository (McGill Dataverse) and uses the DOI
for the StatCan Dialogue Dataset. You will need to provide the API token created
from the account that has been granted access to the file.


### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `api_token` | `str` |  | The API token for the account that has been granted access to the file. You can create an API token from the account settings page. See the Dataverse documentation for more information: https://guides.dataverse.org/en/latest/api/auth.html |
| `server_url` | `str` | `"https://borealisdata.ca"` | The URL of the Dataverse repository. By default, this is the URL for the Borealis Data repository (McGill Dataverse). |
| `persistent_id` | `str` | `"doi:10.5683/SP3/NR0BMY"` | The persistent identifier for the file. By default, this is the DOI for the StatCan Dialogue Dataset. You can find the persistent identifier for a file by going to the file's page on the Dataverse repository and clicking the "Share" button. The persistent identifier can be found in the URL of the page. |
| `filename` | `str` | `"task_data.zip"` | The name of the file to download. By default, this is the name of the file in the StatCan Dialogue Dataset. |
| `data_dir` | `str or Path` | `None` | The directory to download the file to. By default, this is the directory returned by `utils.get_data_dir()`. If the directory does not exist, it will be created. |
| `overwrite` | `bool` | `False` | Whether to overwrite the file if it already exists. By default, this is False, which means that the file will not be downloaded if it already exists. |


### Note

Once this is downloaded, you can load the task data using the `extract_task_data_zip` function.

## `download_huggingface`

```python
sdd.download_huggingface(api_token, repository_url="https://huggingface.co/datasets/McGill-NLP/statcan-dialogue-dataset", branch="main", filename="task_data.zip", data_dir=None, overwrite=False)
```

### Description

This downloads the task data from HuggingFace. It requires an API token to be
passed in, which can be generated from your HuggingFace account, after you
have been granted access to the data repository.


### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `api_token` | `str` |  | The API token for the account that has been granted access to the file. You can create an API token from the account settings page. See the HuggingFace documentation for more information: https://huggingface.co/docs/hub/security-tokens |
| `repository_url` | `str` | `"https://huggingface.co/datasets/McGill-NLP/statcan-dialogue-dataset"` | The URL of the HuggingFace repository. By default, this is the URL for the StatCan Dialogue Dataset repository. |
| `branch` | `str` | `"main"` | The branch of the repository to download the file from. By default, this is the "main" branch. |
| `filename` | `str` | `"task_data.zip"` | The name of the file to download. By default, this is the name of the file that contains the task data. |
| `data_dir` | `str or Path` | `None` | The directory to download the file to. By default, this is the directory returned by `utils.get_data_dir()`. If the directory does not exist, it will be created. |
| `overwrite` | `bool` | `False` | Whether to overwrite the file if it already exists in the directory. By default, this is False. |


### Note

Once this is downloaded, you can load the task data using the `extract_task_data_zip` function.

## `download_full_tables`

```python
sdd.download_full_tables(data_dir=None, lang="en", deposition_id="auto", show_progress=True)
```

### Description

Download the full tables from Zenodo.


### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data_dir` | `str or Path` | `None` | The directory to download the tables to. If None, the default data directory is used. If the directory does not exist, it will be created. |
| `lang` | `str` | `'en' ("en")` | The language of the tables to download. Must be 'en' or 'fr'. |
| `deposition_id` | `int or str` | `'auto' ("auto")` | The Zenodo deposition ID to download the tables from. If 'auto', the appropriate deposition ID is used based on the value of `lang`. |
| `show_progress` | `bool` | `True` | Whether to show a progress bar while downloading the tables. If True, the `tqdm` package must be installed. |


### Returns

```
Path
```

The path to the downloaded tables.

## `extract_task_data_zip`

```python
sdd.extract_task_data_zip(filename="task_data.zip", data_dir=None, load_dir=None, remove_zip=False)
```

### Description

Extracts the ZIP file from Huggingface/Dataverse from the given path. The file is extracted
to the data directory. For instructions on how to download the data from dataverse,
visit: https://mcgill-nlp.github.io/statcan-dialogue-dataset


### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `filename` | `str or Path` | `"task_data.zip"` | The path to the ZIP file to extract. |
| `data_dir` | `str or Path` | `None` | The path to the data directory. If None, the default data directory is used. |
| `load_dir` | `str or Path` | `None` | The path to the directory to load the data from. If None, the data directory is used. This is useful if load_dir is different from data_dir, e.g. if the data is extracted to a different directory. |
| `remove_zip` | `bool` | `True (False)` | Whether to remove the ZIP file after extraction. |


## `load_table`

```python
sdd.load_table(code, data_dir=None, load_from_zip=True, lang="en")
```

### Description

Loads a table from the data directory.


### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `code` | `str` |  | The PID code of the table to load. |
| `data_dir` | `str or Path` | `None` | The path to the data directory. If None, the default data directory is used. |
| `load_from_zip` | `bool` | `True` | Whether to load the table from the ZIP file. If False, it is assumed that the tables-<lang> directory exists in the data directory, which contains the <code>.csv.zip files. |
| `lang` | `str` | `"en"` | The language of the table to load. Must be either "en" or "fr". |


### Returns

```
pd.DataFrame
```

The table with the given PID code.

### Note

The tables must be downloaded first using the `download_full_tables` function.

## `extract_full_tables`

```python
sdd.extract_full_tables(data_dir=None, remove_zip=False, lang="en")
```

## `load_task_data`

```python
sdd.load_task_data(task="retrieval", lang="en", split="train", with_hn=False, with_augmented=False, data_dir=None)
```

### Description

Loads the data for a given task, language and split. The data is returned as a
pandas DataFrame.


### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `task` | `str` | `'retrieval' ("retrieval")` | The task to load the data for. Must be either 'retrieval' or 'generation'. |
| `lang` | `str` | `'en' ("en")` | The language to load the data for. Must be either 'en' or 'fr'. |
| `split` | `str` | `'train' ("train")` | The split to load the data for. Must be either 'train', 'valid' or 'test'. |
| `with_hn` | `bool` | `False` | Whether to include the BM25 hard negatives for the retrieval task. |
| `with_augmented` | `bool` | `False` | Whether to include the augmented data for the generation task. |
| `data_dir` | `str or Path` | `None` | The path to the base data directory. If None, the default data directory is used. It must contain subdirectories "retrieval" and "generation", which contain the data for the retrieval and generation tasks respectively. |


## `load_retrieval_metadata`

```python
sdd.load_retrieval_metadata(data_dir=None)
```

### Description

Loads the metadata for the retrieval task. The metadata is returned as a pandas DataFrame.
The 'pid' column is set as the index. Note this is not the same thing as sdd.metadata module.

## `load_index_splits`

```python
sdd.load_index_splits(data_dir=None)
```

### Description

Loads the list of indices for each split (train, valid, test). Those indices are
used to split the data into the different splits. This is only needed when you are
using the json form of the original conversation data.

## `load_task_conversations`

```python
sdd.load_task_conversations(data_dir=None)
```

### Description

Loads the conversations for the retrieval and generation tasks. The conversations
are returned as a dictionary with the following structure:
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

## `find_pid`

```python
sdd.find_pid(url)
```

### Description

Given a URL, finds the pid value (or returns None if not found, or -1 if the PID is invalid).

