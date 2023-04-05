---
permalink: /docs/utils/
title: "Utility Functions"
---
# Reference for `sdd.utils`

You can find here some useful utility functions for the `statcan_dialogue_dataset` package.

To start:

```python
import statcan_dialogue_dataset as sdd
```

## `get_data_dir`

```python
sdd.utils.get_data_dir()
```

#### Description

Returns the path to the data directory. The data directory is a subdirectory of the user's home directory.

## `get_raw_data_dir`

```python
sdd.utils.get_raw_data_dir()
```

#### Description

Returns the path to the raw data directory. By default, it will be a subdirectory of the data directory.
This will be used to store the raw data files.

## `get_large_data_dir`

```python
sdd.utils.get_large_data_dir()
```

#### Description

Returns the path to the large data directory. By default, it will be a subdirectory of the data directory.
This will be used to store the large data files.

## `get_checkpoint_dir`

```python
sdd.utils.get_checkpoint_dir()
```

#### Description

Returns the path to the checkpoint directory. By default, it will be a subdirectory of the data directory.
This will be used to store custom model checkpoints.

## `get_temp_dir`

```python
sdd.utils.get_temp_dir()
```

#### Description

Returns the path to the temp directory. By default, it will be a subdirectory of the data directory.
This will be used to store the temporary files.

