"""
This loads the json from full_metadata.zip and break it down into 4 data files and save them into _data:
    1. footnotes.json: A map from PID to a list of footnotes.
    2. dimensions.json: A map from PID to a list of dimensions and member items.
    3. basic_info.json: A map from PID to a list of basic metadata information.
    4. title_to_pid.json: A map from table title to PID.

The `full_metadata.zip` can be found on the 
[GitHub Releases Data page](https://github.com/McGill-NLP/statcan-dialogue-dataset/releases/data-latest).
"""
import argparse
from copy import deepcopy
import json
from zipfile import ZipFile, ZIP_DEFLATED

from statcan_dialogue_dataset._data import data_path


def main(in_path, out_dir):
    with ZipFile(in_path, "r") as zf:
        full_metadata = json.load(zf.open("full_metadata.json"))

    basic_info = {}
    members = {}
    footnotes = {}
    title_to_pid = {"fr": {}, "en": {}}

    for pid, data in full_metadata.items():
        data["member"] = deepcopy(data["dimension"])
        data["dimension"] = [
            {k: v for k, v in dim.items() if k != "member"} for dim in data["dimension"]
        ]

        basic_info[pid] = {
            "pid": data["productId"],
            "title_en": data["cubeTitleEn"],
            "title_fr": data["cubeTitleFr"],
            "start_date": data["cubeStartDate"],
            "end_date": data["cubeEndDate"],
            "survey_code": data["surveyCode"],
            "subject_code": data["subjectCode"],
            "frequency_code": data["frequencyCode"],
            "release_time": data["releaseTime"],
            "archive_en": data["archiveStatusEn"],
            "archive_fr": data["archiveStatusFr"],
            "dimensions": data["dimension"],
        }
        members[pid] = data["member"]
        footnotes[pid] = data["footnote"]
        title_to_pid["en"][data["cubeTitleEn"]] = pid
        title_to_pid["fr"][data["cubeTitleFr"]] = pid

    with open(out_dir / "basic_info.json", "w") as f:
        json.dump(basic_info, f)

    with open(out_dir / "footnotes.json", "w") as f:
        json.dump(footnotes, f)

    with open(out_dir / "title_to_pid.json", "w") as f:
        json.dump(title_to_pid, f)

    with ZipFile(out_dir / "members.zip", "w", compression=ZIP_DEFLATED) as zf:
        for pid, data in members.items():
            zf.writestr(f"{pid}.json", json.dumps(data))

        zf.writestr("pids.json", json.dumps(list(members.keys())))


def add_args(parser):
    in_path = "raw_data/full_metadata.zip"
    out_dir = data_path

    parser.add_argument(
        "--in_path",
        help="Path to the zip file containing the full metadata.",
        default=in_path,
    )
    parser.add_argument(
        "--out_dir", help="Path to the output directory.", default=out_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse the full metadata into 4 data files."
    )
    add_args(parser)
    args = parser.parse_args()

    main(args.in_path, args.out_dir)
