---
permalink: /docs/cli-process/
title: "Metadata Processing (CLI)"
---

# Reference for `sdd.cli.process`

This is the reference for the `sdd.cli.process` module. It contains the functions used to process the raw data files into the metadata files. This is likely not useful for most users, but it is useful for developers who want to understand how the metadata files are generated, in case they want to modify the generation process or use new data files.

To start:

```python
import statcan_dialogue_dataset as sdd
```


## `sdd.cli.process.code_sets`

### `main`

```python
sdd.cli.process.code_sets.main(in_path, out_path)
```

##### Description

Generates the code sets map from the raw code sets.


##### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `in_path` | `str` |  | The path to the raw code sets file. It must be the `code_sets.json` file. The `code_sets.json` can be found on the [GitHub Releases Data page](https://github.com/McGill-NLP/statcan-dialogue-dataset/releases/data-latest). |
| `out_path` | `str` |  | The path to the output code sets map file. |


### `add_args`

```python
sdd.cli.process.code_sets.add_args(parser)
```

## `sdd.cli.process.full_metadata`

### `main`

```python
sdd.cli.process.full_metadata.main(in_path, out_dir)
```

### `add_args`

```python
sdd.cli.process.full_metadata.add_args(parser)
```

## `sdd.cli.process.metadata_hierarchy`

### `generate_metadata_hierarchy`

```python
sdd.cli.process.metadata_hierarchy.generate_metadata_hierarchy(full_metadata)
```

##### Description

Generates the metadata hierarchy from the full metadata.


##### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `full_metadata` | `dict` |  | The full metadata, as a dictionary. It must be loaded from the `full_metadata.zip` file, preferably using the `load_full_metadata` function. |


### `add_args`

```python
sdd.cli.process.metadata_hierarchy.add_args(parser)
```

### `main`

```python
sdd.cli.process.metadata_hierarchy.main(in_path, out_path)
```

##### Description

Generates the metadata hierarchy from the full metadata.


##### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `in_path` | `str` |  | Path to the zip file containing the full metadata. It must be the `full_metadata.zip` file. You can download it from the [GitHub Releases Data page](https://github.com/McGill-NLP/statcan-dialogue-dataset/releases/tag/data-latest) |
| `out_path` | `str` |  | Path to the output file, which is by default `<data_path>/metadata_hierarchy.json`. |


