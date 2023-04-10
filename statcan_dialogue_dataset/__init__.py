from datetime import datetime, date
import json
import os
from textwrap import dedent
import warnings
from pathlib import Path
import urllib.request
from urllib.parse import urlparse, parse_qs
import zipfile

import pandas as pd

from .version import __version__
from . import utils, metadata



def download_dataverse(
    api_token,
    server_url="https://borealisdata.ca",
    persistent_id="doi:10.5683/SP3/NR0BMY",
    filename="task_data.zip",
    data_dir=None,
    overwrite=False,
    bundle_name="dataverse_files.zip",
    remove_bundle=False,
):
    """
    Download a file from a Dataverse repository. By default, this downloads the
    file from the Borealis Data repository (McGill Dataverse) and uses the DOI
    for the StatCan Dialogue Dataset. You will need to provide the API token created
    from the account that has been granted access to the file.

    Parameters
    ----------
    api_token : str
        The API token for the account that has been granted access to the file. You can
        create an API token from the account settings page. See the Dataverse documentation
        for more information: https://guides.dataverse.org/en/latest/api/auth.html
    
    server_url : str, default "https://borealisdata.ca"
        The URL of the Dataverse repository. By default, this is the URL for the Borealis
        Data repository (McGill Dataverse).
    
    persistent_id : str, default "doi:10.5683/SP3/NR0BMY"
        The persistent identifier for the file. By default, this is the DOI for the
        StatCan Dialogue Dataset. You can find the persistent identifier for a file
        by going to the file's page on the Dataverse repository and clicking the
        "Share" button. The persistent identifier can be found in the URL of the page.
    
    filename : str, default "task_data.zip"
        The name of the file to download. By default, this is the name of the file
        in the StatCan Dialogue Dataset.
    
    data_dir : str or Path, default None
        The directory to download the file to. By default, this is the directory returned
        by `utils.get_data_dir()`. If the directory does not exist, it will be created.
    
    overwrite : bool, default False
        Whether to overwrite the file if it already exists. By default, this is False,
        which means that the file will not be downloaded if it already exists.

    bundle_name : str, default "dataverse_files.zip"
        The name of the bundle file that contains the file to download. By default,
        this is the default name used by Dataverse. Note that it is indeed possible that it
        contains a zip file (so a zip in a zip), which is the case by default (task_data.zip 
        is contained in dataverse_files.zip). You generally don't need to change this nor 
        the `filename` above.
    
    remove_bundle : bool, default False
        Whether to remove the bundle file after extracting the file.
    
    Note
    ----
    Once this is downloaded, you can load the task data using the `extract_task_data_zip` function.
    """
    data_dir = utils.infer_and_create_dir(data_dir, utils.get_data_dir())
    
    if (data_dir / filename).exists() and not overwrite:
        print(f"File {filename} already exists in {data_dir}. Skipping download.")
        return
    elif (data_dir / bundle_name).exists() and not overwrite:
        print(f"File {bundle_name} already exists in {data_dir}. Skipping download.")
    else:
        url = f"{server_url}/api/access/dataset/:persistentId?persistentId={persistent_id}"

        # Need to pass the API token in the header
        headers = {"X-Dataverse-key": api_token}

        # Make the request with urllib
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            with open(data_dir / bundle_name, "wb") as f:
                f.write(response.read())

    # Extract the file from the bundle
    with zipfile.ZipFile(data_dir / bundle_name, "r") as zip_ref:
        zip_ref.extract(filename, data_dir)

    if remove_bundle:
        os.remove(data_dir / bundle_name)
    
    return data_dir / filename

def download_huggingface(
    api_token,
    repository_url="https://huggingface.co/datasets/McGill-NLP/statcan-dialogue-dataset",
    branch="main",
    filename='task_data.zip',
    data_dir=None,
    overwrite=False,
):
    """
    This downloads the task data from HuggingFace. It requires an API token to be
    passed in, which can be generated from your HuggingFace account, after you
    have been granted access to the data repository.

    Parameters
    ----------
    api_token : str
        The API token for the account that has been granted access to the file. You can
        create an API token from the account settings page. See the HuggingFace documentation
        for more information: https://huggingface.co/docs/hub/security-tokens
    
    repository_url : str, default "https://huggingface.co/datasets/McGill-NLP/statcan-dialogue-dataset"
        The URL of the HuggingFace repository. By default, this is the URL for the
        StatCan Dialogue Dataset repository.
    
    branch : str, default "main"
        The branch of the repository to download the file from. By default, this is
        the "main" branch.
    
    filename : str, default "task_data.zip"
        The name of the file to download. By default, this is the name of the file
        that contains the task data.
    
    data_dir : str or Path, default None
        The directory to download the file to. By default, this is the directory
        returned by `utils.get_data_dir()`. If the directory does not exist, it will
        be created.
    
    overwrite : bool, default False
        Whether to overwrite the file if it already exists in the directory. By
        default, this is False.
    
    Note
    ----
    Once this is downloaded, you can load the task data using the `extract_task_data_zip` function.
    """
    data_dir = utils.infer_and_create_dir(data_dir, utils.get_data_dir())
    
    if (data_dir / filename).exists() and not overwrite:
        print(f"File {filename} already exists in {data_dir}. Skipping download.")
        return
    
    base_url = f"{repository_url}/resolve/{branch}/{filename}"
    headers = {"Authorization": f"Bearer {api_token}"}

    with urllib.request.urlopen(urllib.request.Request(base_url, headers=headers)) as response:
        data = response.read()
        
        with open(data_dir / filename, "wb") as f:
            f.write(data)
    
    return data_dir / filename


def download_full_tables(
    data_dir=None, lang="en", deposition_id="auto", show_progress=True
):
    """
    Download the full tables from Zenodo.

    Parameters
    ----------
    data_dir : str or Path, default None
        The directory to download the tables to. If None, the default data directory
        is used. If the directory does not exist, it will be created.
    lang : str, default 'en'
        The language of the tables to download. Must be 'en' or 'fr'.
    deposition_id : int or str, default 'auto'
        The Zenodo deposition ID to download the tables from. If 'auto', the
        appropriate deposition ID is used based on the value of `lang`.
    show_progress : bool, default True
        Whether to show a progress bar while downloading the tables. If True,
        the `tqdm` package must be installed.

    Returns
    -------
    Path
        The path to the downloaded tables.
    """
    data_dir = utils.infer_and_create_dir(data_dir, utils.get_data_dir())

    if lang not in ["en", "fr"]:
        raise ValueError(f"lang must be 'en' or 'fr', not '{lang}'")

    if deposition_id == "auto":
        if lang == "en":
            deposition_id = 7765406
        elif lang == "fr":
            deposition_id = 7772918

    filename = f"tables-{lang}.zip"
    url = f"https://zenodo.org/record/{deposition_id}/files/{filename}?download=1"

    if show_progress:
        try:
            from tqdm.auto import tqdm

            # First, get the total size of the file
            with urllib.request.urlopen(url) as response:
                total_size = int(response.info()["Content-Length"])

            t = tqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc="Downloading tables zip",
                total=total_size,
            )

            def reporthook(blocknum, blocksize, total):
                t.update(blocknum * blocksize - t.n)

        except ImportError:
            raise ImportError("Please install tqdm with `pip install tqdm` to show download progress.")

    else:
        reporthook = None

    res = urllib.request.urlretrieve(
        url, filename=data_dir / filename, reporthook=reporthook
    )

    if show_progress:
        t.close()

    return res


def extract_task_data_zip(filename="task_data.zip", data_dir=None, load_dir=None, remove_zip=False):
    """
    Extracts the ZIP file from Huggingface/Dataverse from the given path. The file is extracted
    to the data directory. For instructions on how to download the data from dataverse,
    visit: https://mcgill-nlp.github.io/statcan-dialogue-dataset

    Parameters
    ----------
    filename : str or Path
        The path to the ZIP file to extract.
    data_dir : str or Path, default None
        The path to the data directory. If None, the default data directory is used.
    load_dir : str or Path, default None
        The path to the directory to load the data from. If None, the data directory
        is used. This is useful if load_dir is different from data_dir, e.g. if the
        data is extracted to a different directory.
    remove_zip : bool, default True
        Whether to remove the ZIP file after extraction.
    """
    if data_dir is None:
        data_dir = utils.get_data_dir()
    else:
        data_dir = Path(data_dir)
    
    if load_dir is None:
        load_dir = data_dir
    
    path = Path(load_dir) / filename

    # Extract file and save it to the data directory
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(str(data_dir))

    # Remove the ZIP file
    if remove_zip:
        os.remove(filename)


def load_table(code, data_dir=None, load_from_zip=True, lang="en"):
    """
    Loads a table from the data directory.

    Parameters
    ----------
    code : str
        The PID code of the table to load.
    data_dir : str or Path, default None
        The path to the data directory. If None, the default data directory is used.
    load_from_zip : bool, default True
        Whether to load the table from the ZIP file. If False, it is assumed that the
        tables-<lang> directory exists in the data directory, which contains the
        <code>.csv.zip files.
    lang : str, default "en"
        The language of the table to load. Must be either "en" or "fr".
    
    Returns
    -------
    pd.DataFrame
        The table with the given PID code.
    
    Note
    ----
    The tables must be downloaded first using the `download_full_tables` function.
    """
    if lang not in ["en", "fr"]:
        raise ValueError(f"lang must be 'en' or 'fr', not '{lang}'")

    if data_dir is None:
        data_dir = utils.get_data_dir()
    else:
        data_dir = Path(data_dir)

    if not load_from_zip:
        table = pd.read_csv(
            data_dir / f"tables-{lang}" / f"{code}.csv.zip", index_col=0
        )
    else:
        with zipfile.ZipFile(data_dir / f"tables-{lang}.zip", "r") as zip_ref:
            with zip_ref.open(f"{code}.csv.zip") as f:
                table = pd.read_csv(f, index_col=0, compression="zip")

    return table


def extract_full_tables(data_dir=None, remove_zip=False, lang="en"):
    if lang not in ["en", "fr"]:
        raise ValueError(f"lang must be 'en' or 'fr', not '{lang}'")

    if data_dir is None:
        data_dir = utils.get_data_dir()
    else:
        data_dir = Path(data_dir)

    with zipfile.ZipFile(data_dir / f"tables-{lang}.zip", "r") as zip_ref:
        zip_ref.extractall(str(data_dir / f"tables-{lang}"))

    if remove_zip:
        os.remove(data_dir / f"tables-{lang}.zip")

    return data_dir / f"tables-{lang}"


def load_task_data(
    task="retrieval",
    lang="en",
    split="train",
    with_hn=False,
    with_augmented=False,
    data_dir=None,
):
    """
    Loads the data for a given task, language and split. The data is returned as a
    pandas DataFrame.

    Parameters
    ----------
    task : str, default 'retrieval'
        The task to load the data for. Must be either 'retrieval' or 'generation'.
    lang : str, default 'en'
        The language to load the data for. Must be either 'en' or 'fr'.
    split : str, default 'train'
        The split to load the data for. Must be either 'train', 'valid' or 'test'.
    with_hn : bool, default False
        Whether to include the BM25 hard negatives for the retrieval task.
    with_augmented : bool, default False
        Whether to include the augmented data for the generation task.
    data_dir : str or Path, default None
        The path to the base data directory. If None, the default data directory is used.
        It must contain subdirectories "retrieval" and "generation", which contain the
        data for the retrieval and generation tasks respectively.
    """
    if data_dir is None:
        data_dir = utils.get_data_dir()
    else:
        data_dir = Path(data_dir)

    # Check if the inputs are valid
    if task not in ["retrieval", "generation"]:
        raise ValueError(
            f"task must be either 'retrieval' or 'generation'. Got: {task}"
        )

    if lang not in ["en", "fr"]:
        raise ValueError(f"lang must be either 'en' or 'fr'. Got: {lang}")

    if split not in ["train", "valid", "test"]:
        if split in ["dev", "development", "validation", "val"]:
            split = "valid"
        else:
            raise ValueError(
                f"split must be either 'train', 'valid' or 'test'. Got: {split}"
            )

    file_path = data_dir / task / f"{split}_{lang}.csv"

    # Load the data
    if with_hn:
        if task == "retrieval":
            file_path = data_dir / task / f"{split}_{lang}_bm25.csv"

        else:
            raise ValueError(
                f"with_hn is only supported for the retrieval task. Got: {task}"
            )

    elif with_augmented:
        if task == "generation":
            file_path = data_dir / task / f"{split}_{lang}_augmented.csv.zip"

        else:
            raise ValueError(
                f"with_augmented is only supported for the generation task. Got: {task}"
            )

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


def _pre_verify_file(default_data_dir, data_dir, file_name):
    if data_dir is None:
        data_dir = default_data_dir
    else:
        data_dir = Path(data_dir)

    file_path = data_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return file_path


def load_retrieval_metadata(data_dir=None):
    """
    Loads the metadata for the retrieval task. The metadata is returned as a pandas DataFrame.
    The 'pid' column is set as the index. Note this is not the same thing as sdd.metadata module.
    """
    file_path = _pre_verify_file(
        default_data_dir=utils.get_data_dir() / "retrieval",
        data_dir=data_dir,
        file_name="metadata.csv",
    )

    return pd.read_csv(file_path).set_index("pid")


def load_index_splits(data_dir=None):
    """
    Loads the list of indices for each split (train, valid, test). Those indices are
    used to split the data into the different splits. This is only needed when you are
    using the json form of the original conversation data.
    """
    file_path = _pre_verify_file(
        default_data_dir=utils.get_data_dir(),
        data_dir=data_dir,
        file_name="index_splits.json",
    )

    return json.load(open(file_path, "r"))


def load_task_conversations(data_dir=None):
    """
    Loads the conversations for the retrieval and generation tasks. The conversations
    are returned as a dictionary with the following structure:
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
    """
    file_path = _pre_verify_file(
        default_data_dir=utils.get_data_dir(),
        data_dir=data_dir,
        file_name="task_conversations.json",
    )
    return json.load(open(file_path, "r"))


def find_pid(url):
    """
    Given a URL, finds the pid value (or returns None if not found, or -1 if the PID is invalid).
    """
    parsed = urllib.parse.urlparse(url)
    query = parse_qs(parsed.query)
    if "pid" not in query:
        return None

    pid = query["pid"][0]
    pid = str(pid)

    if len(pid) > 8:
        return pid[:8]

    elif len(pid) < 8:
        return -1

    else:
        return pid
