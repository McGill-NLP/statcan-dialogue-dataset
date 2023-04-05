import os
from pathlib import Path

def get_data_dir():
    """
    Returns the path to the data directory. The data directory is a subdirectory of the user's home directory.
    """
    return Path(os.getenv("STATCAN_DATA_DIR", Path.home() / ".statcan_dialogue_dataset"))

def get_raw_data_dir():
    """
    Returns the path to the raw data directory. By default, it will be a subdirectory of the data directory.
    This will be used to store the raw data files.
    """
    return Path(os.getenv("STATCAN_RAW_DATA_DIR", get_data_dir())) / "raw"

def get_large_data_dir():
    """
    Returns the path to the large data directory. By default, it will be a subdirectory of the data directory.
    This will be used to store the large data files.
    """
    return Path(os.getenv("STATCAN_LARGE_DATA_DIR", get_data_dir())) / "large"

def get_checkpoint_dir():
    """
    Returns the path to the checkpoint directory. By default, it will be a subdirectory of the data directory.
    This will be used to store custom model checkpoints.
    """
    return Path(os.getenv("STATCAN_CHECKPOINT_DIR", get_data_dir())) / "checkpoints"

def get_temp_dir():
    """
    Returns the path to the temp directory. By default, it will be a subdirectory of the data directory.
    This will be used to store the temporary files.
    """
    return Path(os.getenv("STATCAN_TEMP_DIR", get_data_dir())) / "temp"

