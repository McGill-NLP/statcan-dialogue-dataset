"""
This script uses the cube_list.json to create a basic dataframe containing data for the 
retrieval task.
"""
import os
import json
from textwrap import dedent
import zipfile

import pandas as pd

from eacl_code.utils import get_data_dir, get_raw_data_dir


def get_basic_info(c, frequency_codes, subject_codes, survey_codes, postfix="En"):
    freq_desc = frequency_codes.loc[int(c["frequencyCode"]), f"frequencyDesc{postfix}"]
    subject_info = [
        subject_codes.loc[int(code), f"subject{postfix}"] for code in c["subjectCode"]
    ]

    survey_info = []
    for code in c["surveyCode"]:
        code = int(code)
        if code in survey_codes.index:
            survey_info.append(survey_codes.loc[code, f"survey{postfix}"])
        else:
            surveys_not_found.append([code, c["productId"]])
            survey_info.append(f"Survey {code}")

    if postfix == "Fr":
        # We use different spec for french
        titles_mapping = {
            "Frequency": "Fréquence",
            "Subject": "Sujet",
            "Survey": "Enquête",
            "Dimensions": "Dimensions",
            "Footnotes": "Notes",
            "Title": "Titre",
            "Date range": "Période",
        }
    else:
        titles_mapping = {
            "Frequency": "Frequency",
            "Subject": "Subject",
            "Survey": "Survey",
            "Dimensions": "Dimensions",
            "Footnotes": "Footnotes",
            "Title": "Title",
            "Date range": "Date range",
        }


    basic_info = dedent(
        f"""
        {titles_mapping['Title']}: {c[f'cubeTitle{postfix}']}
        {titles_mapping['Date range']}: {c['cubeStartDate']} to {c['cubeEndDate']}
        {titles_mapping['Dimensions']}: {", ".join([dim[f'dimensionName{postfix}'] for dim in c['dimensions']])}
        {titles_mapping['Subject']}: {", ".join(subject_info)}
        {titles_mapping['Survey']}: {", ".join(survey_info)}
        {titles_mapping['Frequency']}: {freq_desc}

        """
    ).strip()

    return basic_info


def get_member_info(c, postfix='En'):
    dim_info = []

    for dim in c["dimension"]:
        dim_name = dim[f"dimensionName{postfix}"]

        formatted = f"{dim_name}:\n" + "\n".join(
            [
                f"ID: {m['memberId']}, Parent: {m['parentMemberId']}, Name: {m[f'memberName{postfix}']}"
                for m in dim["member"]
            ]
        )

        dim_info.append(formatted)

    footnotes = "\n".join(
        [
            f"ID: {f['link']['memberId']}, Note: {f[f'footnotes{postfix}']}"
            for f in c["footnote"]
        ]
    )

    if footnotes is None:
        footnotes = "No footnote."

    member_info = "\n".join(dim_info)

    return member_info, footnotes

if __name__ == "__main__":
    raw_data_dir = get_raw_data_dir()
    data_dir = get_data_dir()
    
    with zipfile.ZipFile(raw_data_dir / "full_metadata.zip", "r") as zf:
        full_metadata = json.load(zf.open("full_metadata.json"))

    cube_list = json.load(open(raw_data_dir / "cube_list.json"))

    survey_codes = pd.read_html(raw_data_dir / "code_sets/survey_code.html")[0].rename(columns={"SurveyFr": "surveyFr"})
    survey_codes.set_index("surveyCode", inplace=True)

    subject_codes = pd.read_html(raw_data_dir / "code_sets/subject_code.html")[0]
    subject_codes.set_index("subjectCode", inplace=True)

    frequency_codes = pd.read_html(raw_data_dir / "code_sets/frequency_code.html")[0]
    frequency_codes.set_index("frequencyCode", inplace=True)

    surveys_not_found = []
    records = []
    ignored_pids = {
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

    for cube in cube_list:
        pid = str(cube['productId'])

        if pid in ignored_pids:
            print(f"Ignoring table {pid} as specified in `create_retrieval_metadata.py`")
            continue
        
        meta = full_metadata[pid]

        if cube["surveyCode"] is None:
            cube["surveyCode"] = []

        record = {}

        record["pid"] = cube["productId"]
        record["title"] = cube["cubeTitleEn"]
        record["basic_info"] = get_basic_info(
            cube, frequency_codes, subject_codes, survey_codes
        )
        record["member_info"], record["footnote_info"] = get_member_info(meta)

        record["title_fr"] = cube["cubeTitleFr"]
        record["basic_info_fr"] = get_basic_info(
            cube, frequency_codes, subject_codes, survey_codes, postfix="Fr"
        )
        record["full_info"] = (
            record["basic_info"]
            + "\n\n"
            + record["member_info"]
            + "\n\n"
            + record["footnote_info"]
        )
        record['title_and_footnote'] = record['title'] + '\n\n' + record['footnote_info']
        record['basic_and_footnote'] = record['basic_info'] + '\n\n' + record['footnote_info']

        record['title_and_member'] = record['title'] + '\n\n' + record['member_info']
        record['basic_and_member'] = record['basic_info'] + '\n\n' + record['member_info']

        # We do everything for french
        record["member_info_fr"], record["footnote_info_fr"] = get_member_info(meta, postfix="Fr")

        record["full_info_fr"] = (
            record["basic_info_fr"]
            + "\n\n"
            + record["member_info_fr"]
            + "\n\n"
            + record["footnote_info_fr"]
        )
        record['title_and_footnote_fr'] = record['title_fr'] + '\n\n' + record['footnote_info_fr']
        record['basic_and_footnote_fr'] = record['basic_info_fr'] + '\n\n' + record['footnote_info_fr']

        record['title_and_member_fr'] = record['title_fr'] + '\n\n' + record['member_info_fr']
        record['basic_and_member_fr'] = record['basic_info_fr'] + '\n\n' + record['member_info_fr']
        
        records.append(record)


    df = pd.DataFrame(records)

    metadata_df = pd.DataFrame.from_records(records)
    surveys_not_found_df = pd.DataFrame(surveys_not_found, columns=["survey_code", "pid"])


    print("The following surveys were not found")
    print(surveys_not_found_df)

    # Fill the footnotes_info member with "No footnote"
    metadata_df["footnote_info"] = metadata_df["footnote_info"].fillna("No footnotes")

    metadata_df.to_csv(data_dir / "retrieval" / "metadata.csv.zip", index=False)
