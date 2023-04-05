"""
This script generates the metadata hierarchy from the full metadata.

The `full_metadata.zip` can be found on the 
[GitHub Releases Data page](https://github.com/McGill-NLP/statcan-dialogue-dataset/releases/data-latest).
"""
import argparse
import json
import zipfile

from statcan_dialogue_dataset._data import data_path


def generate_metadata_hierarchy(full_metadata: dict) -> dict:
    """
    Generates the metadata hierarchy from the full metadata.

    Parameters
    ----------
    full_metadata : dict
        The full metadata, as a dictionary. It must be loaded from the `full_metadata.zip` file,
        preferably using the `load_full_metadata` function.
    """
    survey_to_subject = {}
    survey_to_pid = {}
    subject_to_survey = {}

    for pid, data in full_metadata.items():
        if data["surveyCode"] is None:
            continue

        subject_code = pid[:2]

        if subject_code not in subject_to_survey:
            subject_to_survey[subject_code] = set()

        for survey_code in data["surveyCode"]:
            if survey_code not in survey_to_subject:
                survey_to_subject[survey_code] = set()

            if survey_code not in survey_to_pid:
                survey_to_pid[survey_code] = set()

            survey_to_subject[survey_code].add(subject_code)
            subject_to_survey[subject_code].add(survey_code)
            survey_to_pid[survey_code].add(pid)

    for di in [survey_to_pid, subject_to_survey, survey_to_subject]:
        for k, v in di.items():
            di[k] = list(sorted(v))

    metadata_hierarchy = dict(
        survey_to_subject=survey_to_subject,
        survey_to_pid=survey_to_pid,
        subject_to_survey=subject_to_survey,
    )

    return metadata_hierarchy


def add_args(parser):
    parser.add_argument(
        "--in_path",
        help="Path to the zip file containing the full metadata.",
        default="raw_data/full_metadata.zip",
    )
    parser.add_argument(
        "--out_path",
        help="Path to the output file.",
        default=data_path / "metadata_hierarchy.json",
    )


def main(in_path, out_path):
    """
    Generates the metadata hierarchy from the full metadata.

    Parameters
    ----------
    in_path : str
        Path to the zip file containing the full metadata. It must be the `full_metadata.zip` file.
        You can download it from the
        [GitHub Releases Data page](https://github.com/McGill-NLP/statcan-dialogue-dataset/releases/tag/data-latest)
    out_path : str
        Path to the output file, which is by default `<data_path>/metadata_hierarchy.json`.
    """
    with zipfile.ZipFile(in_path, "r") as zf:
        full_metadata = json.load(zf.open("full_metadata.json"))

    metadata_hierarchy = generate_metadata_hierarchy(full_metadata)
    # Save
    with open(out_path, "w") as f:
        json.dump(metadata_hierarchy, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the metadata hierarchy from the full metadata."
    )
    add_args(parser)
    args = parser.parse_args()

    main(args.in_path, args.out_path)
