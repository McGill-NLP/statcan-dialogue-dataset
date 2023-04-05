---
permalink: /docs/metadata/
title: "Metadata Module"
---
# Reference for `sdd.metadata`


## `MetadataBase`

```python
sdd.metadata.MetadataBase(code, title, lang="en")
```

#### Description

Initialize an object from this class from scratch. This gives more control,
but it is better to use the from_code or from_title methods if possible,
as they require fewer arguments.


#### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `code` | `str` |  | Code that uniquely identifies the object. |
| `title` | `str` |  | Title in the language specified by `lang` describing the object. |
| `lang` | `str` | `"en"` | The language of the object. Defaults to "en". |


### `MetadataBase.__repr__`

```python
sdd.metadata.MetadataBase.__repr__(self)
```

### `MetadataBase.__eq__`

```python
sdd.metadata.MetadataBase.__eq__(self, other)
```

### `MetadataBase.from_code`

```python
sdd.metadata.MetadataBase.from_code(cls, code, lang="en")
```

#### Description

Creates an instance of this class given a code that uniquely identifies the object.
For surveys, this is the record number. For tables, this is the product ID (PID).


#### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `code` | `str` |  | The code of the instance to return. |
| `lang` | `str` | `"en"` | The language of the instance to return. |


#### Notes

If the code is not found, this will return an object with title=None.

### `MetadataBase.from_title`

```python
sdd.metadata.MetadataBase.from_title(cls, title, lang="en")
```

#### Description

Creates an instance of this class using the given title.


#### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `title` | `str` |  | The exact title of the instance to return in the specified language. |
| `lang` | `str` | `"en"` | The language of the instance to return. |


#### Notes

If the title is not found, this will return an object with code -1.

### `MetadataBase.is_valid`

```python
sdd.metadata.MetadataBase.is_valid(self)
```

#### Description

Check if the object is valid. This is the case if the code is not -1 and the title is not None.


#### Returns

```
bool
```

True if the object is valid, False otherwise.

### `MetadataBase.to_dict`

```python
sdd.metadata.MetadataBase.to_dict(self)
```

#### Description



#### Returns

```
dict
```

A copy of the dictionary representation of the object.

### `MetadataBase.list_all`

```python
sdd.metadata.MetadataBase.list_all(cls, lang="en", return_as="object", code_length=None)
```

#### Description

List all possible objects belonging to this class. For example, a list of all
Surveys on Statcan.


#### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `lang` | `str` | `"en"` | The language of the instance to return. |
| `return_as` | `str` | `"object"` | The type of object to return. Must be either "object", "title" or "code". |
| `code_length` | `int` | `None` | Filter the list to only include objects with a code of this length. If None, all objects are returned. |


#### Returns

```
list
```

A list of all possible objects belonging to this class. For example, a list of all
Surveys on Statcan.

### `Survey.get_subjects`

```python
sdd.metadata.Survey.get_subjects(self, return_as="object", lang=None, warn_if_not_found=True)
```

#### Description

Get the subjects associated with the data tables created from this survey.


#### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `return_as` | `str` | `"object"` | How to represent the subjects inside the returned list. Must be either "object", "title" or "code". |
| `lang` | `str` | `None` | The language of the subjects to return. If None, the language of the survey will be used. |
| `warn_if_not_found` | `bool` | `True` | Whether to warn if no subjects are found. |


#### Returns

```
list of {Subject, str}
```

A list of Subject objects, or of the title/code in str. If not found, the list will
be empty.

### `Survey.get_tables`

```python
sdd.metadata.Survey.get_tables(self, return_as="object", lang=None, warn_if_not_found=True)
```

#### Description

Get the data tables created from this survey.


#### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `return_as` | `str` | `"object"` | How to represent the tables inside the returned list. Must be either "object", "title" or "code". |
| `lang` | `str` | `None` | The language of the table. Must be either "en" or "fr", or None to use the default language. |
| `warn_if_not_found` | `bool` | `True` | Whether to print a warning if no table are found. |


#### Returns

```
list of {Table, str}
```

A list of Table objects, or of the title/code in str. If not found, the list will be
empty.

### `Survey.get_url`

```python
sdd.metadata.Survey.get_url(self)
```

#### Description

Get the URL on statcan's website containing information about this survey.


#### Returns

```
str
```

The URL on statcan's website containing information about this survey.

### `Subject.get_surveys`

```python
sdd.metadata.Subject.get_surveys(self, return_as="object", lang=None, warn_if_not_found=True)
```

#### Description

Get the surveys that belong to this subject, which are parsed from the full metadata.


#### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `return_as` | `str` | `"object"` | How to represent the surveys inside the returned list. Must be either "object", "title" or "code". |
| `lang` | `str` | `None` | The language of the survey. Must be either "en" or "fr", or None to use the default language. |
| `warn_if_not_found` | `bool` | `True` | Whether to print a warning if no surveys are found. |


#### Returns

```
list of {Survey, str}
```

A list of Survey objects, or of the title/code in str. If not found, the list will
be empty.

### `Subject.get_url`

```python
sdd.metadata.Subject.get_url(self)
```

#### Description

Get the URL on statcan's website containing information about this subject.


#### Returns

```
str
```

The URL on statcan's website containing information about this subject.

## `Table`

```python
sdd.metadata.Table(code, title, subjects, surveys, frequency, start_date, end_date, release_time, archive_info, lang="en", dimensions=None, footnotes=None)
```

#### Description

A StatCan data table. A list of tables can be found here:
https://www150.statcan.gc.ca/n1/en/type/data

This class is meant to help work with a table's metadata, and not to actually download the data.
To do that, use the download_full_tables() function. To load the content of a table as a CSV, use
the sdd.load_table() function.

You can find useful methods to work with the table's metadata in the MetadataBase class. For
example, to get the surveys that belong to this table, use the get_surveys() method. To get
the subjects that belong to this table, use the get_subjects() method.


#### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `code` | `str` |  | The product ID (PID) of the table. Must be 8 or 10 digits (will be truncated to 8). |
| `title` | `str` |  | The title of the table. |
| `subjects` | `list of Subject (list[Subject])` |  | The subjects categorizing this table. |
| `surveys` | `list of Survey (list[Survey])` |  | The surveys used to create this table. |
| `frequency` | `Frequency` |  | A Frequency object representing how often the table is updated. |
| `start_date` | `datetime.date (date)` |  | Reference period start date for the table. |
| `end_date` | `datetime.date (date)` |  | Reference period end date for the table. |
| `release_time` | `datetime.datetime (datetime)` |  | Time the table was released. |
| `archive_info` | `str` |  | Information about whether the table is archived or not. |
| `lang` | `str` | `'en' ("en")` | The language of the table. Must be either "en" or "fr". |
| `dimensions` | `list` | `None` | The dimensions of the table, and possibly contains a `member` key holding information about the member items. |
| `footnotes` | `list` | `None` | The footnotes of the table, optional. |


#### Returns

```
Table
```

A table object.

#### Notes

The dimensions and footnotes are unprocessed JSON. This is the unprocessed JSON but may
    be changed in the future.
More information about Tables here:
    https://www.statcan.gc.ca/en/developers/wds/user-guide

### `Table.__repr__`

```python
sdd.metadata.Table.__repr__(self)
```

### `Table.is_valid`

```python
sdd.metadata.Table.is_valid(self)
```

#### Description

Checks if this table is valid by if the title is not None and the code is not -1.

### `Table.from_code`

```python
sdd.metadata.Table.from_code(cls, code, lang="en", include_members=False, include_footnotes=False)
```

#### Description

Creates a Table object from a table code (PID).


#### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `code` | `str` |  | The product ID (PID) of the table. Must be 8 or 10 digits (which would be truncated to 8). |
| `lang` | `str` | `'en' ("en")` | The language of the table. Must be either "en" or "fr". |
| `include_members` | `bool` | `False` | Whether to include the member items of the table inside the `dimensions` attribute. |
| `include_footnotes` | `bool` | `False` | Whether to include the footnotes of the table. |


#### Returns

```
Table
```

A table object. If the code is not found, the title will be None.

### `Table.from_title`

```python
sdd.metadata.Table.from_title(cls, title, lang="en", include_members=False, include_footnotes=False)
```

#### Description

Creates a Table object from a title.


#### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `title` | `str` |  | The title of the table. |
| `lang` | `str` | `'en' ("en")` | The language of the table. Must be either "en" or "fr". |
| `include_members` | `bool` | `False` | Whether to include the member items of the table inside the `dimensions` attribute. |
| `include_footnotes` | `bool` | `False` | Whether to include the footnotes of the table. |


#### Returns

```
Table
```

A table object. If the title is not found, the code will be -1.

### `Table.get_subjects`

```python
sdd.metadata.Table.get_subjects(self, return_as="object")
```

#### Description

Get the subjects associated with this table.


#### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `return_as` | `str` | `"object"` | The type of object to return. Must be either "object" or "code". |


#### Returns

```
list of Subject
```

The subjects categorizing this table.

### `Table.get_surveys`

```python
sdd.metadata.Table.get_surveys(self, return_as="object")
```

#### Description

Get the surveys associated with this table.


#### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `return_as` | `str` | `"object"` | The type of object to return. Must be either "object" or "code". |


#### Returns

```
list of Survey
```

The surveys used to create this table.

### `Table.get_ignored_codes`

```python
sdd.metadata.Table.get_ignored_codes()
```

#### Description

Get the product IDs (PIDs) that are not included in the dataset.


#### Returns

```
set of str
```

Product IDs (PIDs) that are not included in the dataset.

### `Table.get_url`

```python
sdd.metadata.Table.get_url(self)
```

#### Description

Get the URL on statcan's website containing information about this table.


#### Returns

```
str
```

The URL on statcan's website containing information about this table.

#### Examples

```python
>>> Table.from_code("36100293").get_url()
'https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3610029301'
```



### `Table.to_dict`

```python
sdd.metadata.Table.to_dict(self)
```

### `Table.key_to_prefix`

```python
sdd.metadata.Table.key_to_prefix()
```

#### Description

Get a dictionary mapping the keys of the table's `to_dict` method to their prefixes.


#### Returns

```
dict of str
```

A dictionary mapping the keys of the table's `to_dict` method to their prefixes.

### `Table.to_basic_info_text`

```python
sdd.metadata.Table.to_basic_info_text(self, omitted_attrs=('"code"', '"lang"', '"footnotes"'))
```

#### Description

Get a human-readable textual representation of the table's basic information,
omitting the product ID (PID), member items and footnotes.



#### Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `omitted_attrs` | `tuple of str` | `('code', 'lang', 'footnotes') (('"code"', '"lang"', '"footnotes"'))` | Attributes to omit from the output. |


#### Returns

```
str
```

A human-readable textual representation.

#### Notes

This formats the basic information slightly differently from the paper, as
the order is different, and some extra information is included.

