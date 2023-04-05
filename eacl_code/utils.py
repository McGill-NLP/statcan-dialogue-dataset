import os
from pathlib import Path
import re
import json
from collections import OrderedDict

def find_urls(description):
    """
    1. Find all urls using regex: https://stackoverflow.com/a/840110/13837091
    2. Remove the trailing ")"
    3. Get unique URLs only (set comprehension -> list)
    """
    return list(
        OrderedDict.fromkeys(
            [s for s in re.findall("(?P<url>https?://[^\s)]+)", description)]
        )
    )


def prepare_for_training(
    conv_json,
    replace_pid=True,
    response_sep="[SEP]",
    pid_token="[PID]",
    msg_sep=" ",
    prefix_actor=True,
):
    """
    Converts the conversation JSON into user response separated by a sep token and replacing
    the URLs containing a PID table with a PID token.

    Parameters
    ----------
    conv_json : str or list
        The conversation JSON.
    replace_pid : bool
        Whether to replace URLs linking to PID tables with a PID token.
    response_sep : str
        The separator token to use for separating responses.
    pid_token : str
        The token to use for replacing PID tables if `replace_pid` is True.
    msg_sep : str
        The separator token to use for separating messages within a response.
    prefix_actor : bool
        Whether to prefix a response with "user" or "operator".

    Returns
    -------
    str
        The conversation JSON in a format ready for training.
    """

    def pad(*args):
        return [" " + a + " " for a in args]

    if type(conv_json) is str:
        conv_json = json.loads(conv_json)

    # First, let's add some white space to sep and pid tokens
    response_sep, pid_token = pad(response_sep, pid_token)

    turns = []
    current_actor = None

    for c in conv_json:
        text = c["text"]

        # Replace URLs with the PID token if needed
        if replace_pid:
            for url in c["urls"]:
                if "action?pid" in url:
                    text = text.replace(url, pid_token)

        # Now, if the actor is different from the previous one, we need to add a sep token
        # otherwise, we just concatenate the text. This is because one response might have
        # multiple messages (sent in succession).
        if c["actor"] != current_actor:
            # If we want to keep the actor name, we need to prefix the response with the actor
            if prefix_actor:
                text = f"{c['actor']}: {text}"

            turns.append(text)
            current_actor = c["actor"]
        else:
            turns[-1] += msg_sep + text

    return response_sep.join(turns)

def encode_right_truncated(
    tokenizer,
    text: list,
    padding: str = "max_length",
    max_length: int = 512,
    add_special_tokens: bool = True,
):
    """
    This truncates the text starting from the right side, then tokenizes and converts it to a tensor.
    """
    tokenized = tokenizer.tokenize(
        text,
        padding=padding,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    )

    if not add_special_tokens:
        truncated = tokenized[-max_length:]
    else:
        truncated = tokenized[0:1] + tokenized[-(max_length - 1) :]

    ids = tokenizer.convert_tokens_to_ids(truncated)

    return ids

def encode_right_truncated_old(
    tokenizer, text, max_length, padding="max_length", return_tensors="pt", **kwargs
):
    import torch
    tokenized = tokenizer(text, max_length=max_length, padding=padding, **kwargs)
    input_ids = [ids[-max_length:] for ids in tokenized.input_ids]
    attn_mask = [mask[-max_length:] for mask in tokenized.attention_mask]

    if return_tensors == "pt":
        input_ids = torch.tensor(input_ids)
        attn_mask = torch.tensor(attn_mask)

    return input_ids, attn_mask


def get_ignored_pids():
    return {
        '12100037',
        '12100147',
        '12100148',
        '12100149',
        '12100150',
        '12100151',
        '12100152',
        '12100153',
        '12100154',
        '12100155',
        '12100156',
        '13100157',
        '13100412',
        '13100575',
        '13100598',
        '13100769',
        '17100062',
        '22100102',
        '32100265',
        '36100293',
    }


def get_data_dir():
    return Path(os.getenv("STATCAN_DATA_DIR", Path.home() / ".statcan_dialogue_dataset"))

def get_raw_data_dir():
    return Path(os.getenv("STATCAN_RAW_DATA_DIR", get_data_dir())) / "raw"

def get_large_data_dir():
    return Path(os.getenv("STATCAN_LARGE_DATA_DIR", get_data_dir())) / "large"

def get_checkpoint_dir():
    return Path(os.getenv("STATCAN_CHECKPOINT_DIR", get_data_dir())) / "checkpoints"

def get_temp_dir():
    return Path(os.getenv("STATCAN_TEMP_DIR", get_data_dir())) / "temp"