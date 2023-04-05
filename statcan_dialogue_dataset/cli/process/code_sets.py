"""
This script generates the code sets map from the raw code sets.

The `code_sets.json` can be found on the 
[GitHub Releases Data page](https://github.com/McGill-NLP/statcan-dialogue-dataset/releases/data-latest).
"""
import argparse
import json

from statcan_dialogue_dataset._data import data_path


def main(in_path, out_path):
    """
    Generates the code sets map from the raw code sets.

    Parameters
    ----------
    in_path : str
        The path to the raw code sets file. It must be the `code_sets.json` file.
        The `code_sets.json` can be found on the
        [GitHub Releases Data page](https://github.com/McGill-NLP/statcan-dialogue-dataset/releases/data-latest).

    out_path : str
        The path to the output code sets map file.
    """
    with open(in_path, "r") as f:
        code_sets_raw = json.load(f)

    code_sets_map = {
        lang: {
            x: {"code_to_title": {}, "title_to_code": {}}
            for x in ["survey", "subject", "frequency"]
        }
        for lang in ["en", "fr"]
    }

    for code_type, desc in [
        ("survey", "survey"),
        ("subject", "subject"),
        ("frequency", "frequencyDesc"),
    ]:
        for x in code_sets_raw["object"][code_type]:
            title_en = x[f"{desc}En"]
            title_fr = x[f"{desc}Fr"]
            code = x[f"{code_type}Code"]

            code_sets_map["fr"][code_type]["code_to_title"][code] = title_fr
            code_sets_map["fr"][code_type]["title_to_code"][title_fr] = code
            code_sets_map["en"][code_type]["code_to_title"][code] = title_en
            code_sets_map["en"][code_type]["title_to_code"][title_en] = code

    with open(out_path, "w") as f:
        json.dump(code_sets_map, f)


def add_args(parser):
    parser.add_argument(
        "--in_path",
        help="Path to the zip file containing the full metadata.",
        default="raw_data/code_sets.json",
    )
    parser.add_argument(
        "--out_path",
        help="Path to the output file.",
        default=data_path / "code_sets_map.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the code sets map from the raw code sets."
    )
    add_args(parser)
    args = parser.parse_args()

    main(args.in_path, args.out_path)
