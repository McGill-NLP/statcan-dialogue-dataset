# Statcan Dialogue Dataset

<div align="center">

[**💻Code**](https://github.com/mcGill-NLP/statcan-dialogue-dataset) | [**📄Paper**](https://arxiv.org/abs/2304.01412) | [**🌐Homepage**](https://mcgill-nlp.github.io/statcan-dialogue-dataset) |
| :--: | :--: | :--: |

> **[The StatCan Dialogue Dataset: Retrieving Data Tables through Conversations with Genuine Intents](https://arxiv.org/abs/2304.01412)**\
> *[Xing Han Lu](https://xinghanlu.com), [Siva Reddy](https://sivareddy.in), [Harm de Vries](https://www.harmdevries.com/)*\
> EACL 2023

![Banner Image showing a sample conversation between a user and an agent](/images/banner.svg)


</div>

This repository contains the code for our project. For detailed information, instructions on requesting access, API user guide and documentation, please visit the [website](https://mcgill-nlp.github.io/statcan-dialogue-dataset). You can find useful links above. A quickstart is provided below.

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
# Get subjects or surveys
subjects = table.get_subjects()
surveys = table.get_surveys()
```

For more information, head to the [Core API Reference](https://mcgill-nlp.github.io/statcan-dialogue-dataset/docs/core/) page.


## Reproduce paper results

The full `eacl_code` directory contains the code used to produce the results in the paper. To reproduce the results, start by reading the `README.md` file in the `eacl_code` directory.


## StatCan resources

You may find the following external resources useful:
* [Web Data Service User Guide](https://www.statcan.gc.ca/eng/developers/wds/user-guide)
* [Web Data Service API](https://www.statcan.gc.ca/eng/developers/wds)
* [Developer homepage](https://www.statcan.gc.ca/eng/developers)
