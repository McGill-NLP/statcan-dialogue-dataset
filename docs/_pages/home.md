---
permalink: /
layout: splash
header:
    overlay_color: rgb(237, 27, 47)
    actions:
        - label: "Paper"
          url: "https://arxiv.org/abs/2304.01412"
          icon: "fas fa-book"
        - label: "Code"
          url: "https://github.com/McGill-NLP/statcan-dialogue-dataset"
          icon: "fab fa-github"
        - label: "Huggingface"
          url: "https://huggingface.co/datasets/McGill-NLP/statcan-dialogue-dataset"
          icon: "fas fa-smile-beam"
        - label: "Dataverse"
          url: "https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/NR0BMY"
          icon: "fas fa-database"
        - label: "Tweets"
          url: "https://twitter.com/xhluca/status/1648728708142727180"
          icon: "fab fa-twitter"
        - label: "Video"
          url: "https://aclanthology.org/2023.eacl-main.206.mp4"
          icon: "fas fa-video"

title: "StatCan Dialogue Dataset"
excerpt: Xing Han Lu, Siva Reddy, Harm de Vries
---


![Sample image showing the conversation between a user and an agent]({{ '/assets/images/banner.svg' | relative_url }})

Welcome to the website of our *EACL 2023* paper:

> **[The StatCan Dialogue Dataset: Retrieving Data Tables through Conversations with Genuine Intents](https://arxiv.org/abs/2304.01412)**


## Data access

You can request access to the data via two ways: [McGill Dataverse](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/NR0BMY) and [Huggingface Dataset](https://huggingface.co/datasets/McGill-NLP/statcan-dialogue-dataset). In both cases, in order to use our dataset, you must agree to the terms of use and restrictions before requesting access (links at the top of the page). We will manually review each request and grant access or reach out to you for further information. To facilitate the process, make sure that:

1. Your Dataverse/Huggingface account is linked to your professional/research website, which we may review to ensure the dataset will be used for the intended purpose
2. Your request is made with an academic (e.g. .edu) or professional email (e.g. @servicenow.com). To do this, your have to set your primary email to your academic/professional email, or create a new Huggingface/Dataverse account.

If your academic institution does not end with .edu, or you are part of a professional group that does not have an email address, please contact us (see email in paper).

## Python Library

We have published a Python library to help you work with the dataset. To get started, please refer to the [documentation](https://mcgill-nlp.github.io/statcan-dialogue-dataset/docs/) for a user guide and API references.
For other code-related details, please check out our [GitHub repository](https://github.com/McGill-NLP/statcan-dialogue-dataset/).

## Citation

If you use our dataset, please cite as follows:

```bibtex
@inproceedings{lu-etal-2023-statcan,
    title = "The {S}tat{C}an Dialogue Dataset: Retrieving Data Tables through Conversations with Genuine Intents",
    author = "Lu, Xing Han  and
      Reddy, Siva  and
      de Vries, Harm",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2304.01412",
    pages = "2799--2829",
}
```

## Video

You can watch the video presentation of our paper at EACL 2023 below:

<div style="position: relative; width: 100%; padding-bottom: 56.25%;">
    <iframe src="https://aclanthology.org/2023.eacl-main.206.mp4" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>