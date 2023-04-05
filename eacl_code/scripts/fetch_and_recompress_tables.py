import argparse
import os
import json
import urllib
import urllib.request
import time
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED, is_zipfile
import os

from tqdm import tqdm
import requests

from eacl_code.utils import get_raw_data_dir, get_large_data_dir, get_temp_dir


def main(args):
    lang = args.lang
    raw_data_dir = args.raw_data_dir
    load_dir = args.load_dir
    meta_save_dir = args.meta_save_dir
    tempdir_dir = args.tempdir_dir
    # Special case for saving tables: Use language to determine name, i.e. /path/to/tables/tables-en or tables-fr
    table_save_dir_by_lang = os.path.join(args.table_save_dir, f"tables-{lang[:2]}")

    os.makedirs(load_dir, exist_ok=True)

    # ######################### PART 1: FETCH TABLES FROM WEBSITE #########################
    cached_pids = {
        int(x.split("-", 1)[0]) 
        for x in os.listdir(load_dir) 
        if x.endswith(f"{lang}.zip")
    }
    failed = json.load(open(os.path.join(raw_data_dir, "failed_fetch.json")))
    cube_list = json.load(open(os.path.join(raw_data_dir, "cube_list.json")))

    if "tables" not in failed:
        failed["tables"] = []

    pids = [cube["productId"] for cube in cube_list]

    for pid in tqdm(pids):
        if pid in cached_pids or pid in failed['tables']:
            continue

        r = requests.get(
            f"https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/{pid}/{lang[:2]}"
        )
        if r.status_code != 200:
            print(f"Failed to find {pid}")
            if pid not in failed["tables"]:
                failed["tables"].append(pid)
                json.dump(failed, open(os.path.join(raw_data_dir, "failed_fetch.json"), "w"))
            continue

        url = json.loads(r.text)["object"]
        name = url.split("/")[-1]

        try:
            res = urllib.request.urlretrieve(url, os.path.join(raw_data_dir, "tables", name))
        except Exception as e:
            print(f"Failed to download {pid}")
            if pid not in failed["tables"]:
                failed["tables"].append(pid)
                json.dump(failed, open(os.path.join(raw_data_dir, "failed_fetch.json"), "w"))

        cached_pids.add(pid)

        time.sleep(2)

    # ######################### PART 2: UNZIP AND RECOMPRESS TABLES #########################

    os.makedirs(table_save_dir_by_lang, exist_ok=True)
    os.makedirs(meta_save_dir, exist_ok=True)
    os.makedirs(tempdir_dir, exist_ok=True)

    pids = {int(x.split("-", 1)[0]) for x in os.listdir(load_dir)}
    cached_pids = {int(x.split(".", 1)[0]) for x in os.listdir(table_save_dir_by_lang)}

    with TemporaryDirectory(dir=tempdir_dir) as tmpdirname:
        print("Temp directory created:", tmpdirname)

        for pid in tqdm(pids):
            fname = f'{pid}-{lang}.zip'

            if pid in cached_pids or pid in failed['tables']:
                continue
            
            if not is_zipfile(os.path.join(load_dir, fname)):
                print(f"Table {pid} is not a correct zipfile")
                failed['tables'].append(pid)
                json.dump(failed, open(os.path.join(raw_data_dir, "failed_fetch.json"), "w"))
                continue

            with ZipFile(os.path.join(load_dir, fname), 'r') as zip_ref:
                table_path = zip_ref.extract(f'{pid}.csv', tmpdirname)
                meta_path = zip_ref.extract(f'{pid}_MetaData.csv', tmpdirname)

            with ZipFile(os.path.join(table_save_dir_by_lang, f'{pid}.csv.zip'), 'w', compression=ZIP_DEFLATED) as zip_ref:
                zip_ref.write(table_path, f'{pid}.csv')

            with ZipFile(os.path.join(meta_save_dir, f'{pid}.csv.zip'), 'w', compression=ZIP_DEFLATED) as zip_ref:
                zip_ref.write(meta_path, f'{pid}.csv')

            # cleanup
            os.remove(table_path)
            os.remove(meta_path)

if __name__ == "__main__":
    raw_data_dir = get_raw_data_dir()
    large_data_dir = get_large_data_dir()


    parser = argparse.ArgumentParser(
        description="Downloads the tables from statcan.gc.ca and recompresses them into ZIP format to save space",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--lang",
        type=str,
        default="eng",
        choices=["eng", "fra"],
        help="Language of the tables to download",
    )
    parser.add_argument(
        '--raw_data_dir',
        type=str,
        default=raw_data_dir,
        help='Directory where raw data are stored (tables and cube_list.json)'
    )
    parser.add_argument(
        '--load_dir',
        type=str,
        default=raw_data_dir / 'tables',
        help='Directory where the tables are stored'
    )

    parser.add_argument(
        '--table_save_dir',
        type=str,
        default=large_data_dir,
        help='Directory where two folders, tables-en and tables-fr, will be created and the tables zip files will be saved.'
    )
    parser.add_argument(
        '--meta_save_dir',
        type=str,
        default=large_data_dir / 'metadata_tables',
        help='Directory where the metadata tables are stored'
    )

    parser.add_argument(
        '--tempdir_dir',
        type=str,
        default=get_temp_dir(),
        help='Directory where temporary files will be stored'
    )

    args = parser.parse_args()

    main(args)