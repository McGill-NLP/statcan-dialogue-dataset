---
title: "Leaderboard"
permalink: /leaderboard/
---

You can find below the leaderboard.

## Retrieval Task

| Authors | Model | Date | Links | Tags | R@1 | R@10 | R@20 |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| McGill NLP | DPR (B+M) | 2023-05-01 | [Paper](https://arxiv.org/pdf/2304.01412.pdf) | SR | 15.7 | 46.2 | 56.3 |

## Generation Task

| Authors | Model | Date | Links | Tags | METEOR | ROUGE-L | MoverScore | BERTScore | Title Acc. |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| McGill NLP | T5 | 2023-05-01 | [Paper](https://arxiv.org/pdf/2304.01412.pdf) | SR | 23.35 | 30.65 | 59.82 | 86.04 | 6.96 |


## Information

Information about the information in the leaderboard can be found below.

## Adding results

To add your results, please fork the [repository](https://github.com/McGill-NLP/statcan-dialogue-dataset), add your results to the [leaderboard.md](https://github.com/McGill-NLP/statcan-dialogue-dataset/blob/main/docs/_pages/leaderboard.md) file and submit a pull request to the main branch with the subject line "Add results for [your team name]".


## Date

Please use the `YYYY-MM-DD` format. If you are submitting multiple results, please use the date of the most recent result.

## Links

In this column, you can provide links to your code, paper, and/or blog post. Rather than giving the full URL, please use the following format:

```
[Code](https://github.com); [Paper](https://arxiv.org/abs/1234.5678); [Blog](https://medium.com)
```

Please do not exceed 5 links.

## Tags

The following tags are allowed:

* `SR`: Self-reported
* `EE`: Externally evaluated (must provide link to evaluation script)
* `RI`: Inference was reproduced by a third party following given instructions (must provide link to instructions and third party's results)
* `TR`: Training process was independently reproduced by a third party following given instructions (must provide link to training scripts and third party's results).

Since `TR` supersedes `RI`, `RI` superseded `EE`, and `EE` supersedes `SR`, please only use the most appropriate tag (or update results if an existing tag exists). If there are multiple tags, please separate them with a comma:

```
[EE](https://github.com/some-repo/eval.py), [TR](https://github.com/another-repo/train.py)
```