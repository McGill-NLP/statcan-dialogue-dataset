import os
import json
import urllib.request
import time
from collections import OrderedDict
from zipfile import ZipFile, ZIP_DEFLATED

from tqdm import tqdm

from eacl_code.utils import get_raw_data_dir

if __name__ == "__main__":
    raw_data_dir = get_raw_data_dir()
    os.makedirs(raw_data_dir / "metadata", exist_ok=True)
    # First, let's fetch the code sets

    urllib.request.urlretrieve('https://www150.statcan.gc.ca/t1/wds/rest/getCodeSets', raw_data_dir / "code_sets.json")

    # Now, let's fetch the rest of metadata
    cached = {int(x.replace(".json", "")) for x in os.listdir(raw_data_dir / "metadata")}
    failed = json.load(open(raw_data_dir / "failed_fetch.json"))
    cube_list = json.load(open(raw_data_dir / "cube_list.json"))
    pids = [c["productId"] for c in cube_list]

    for pid in tqdm(pids):
        if pid in cached:
            continue

        post_url = "https://www150.statcan.gc.ca/t1/wds/rest/getCubeMetadata"
        body = json.dumps([{"productId": pid}]).encode()
        hdrs = {"Content-Type": "application/json"}

        req = urllib.request.Request(post_url, data=body, headers=hdrs)

        try:
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as e:
            print(f"{pid=} failed with http error: {e}")
            if pid not in failed["metadata"]:
                failed["metadata"].append(pid)

            with open(raw_data_dir / "failed_fetch.json", "w") as f:
                json.dump(failed, f)

            time.sleep(2)

            continue

        if len(data) != 1:
            raise Exception(f"{pid} has {len(data)} results")

        if data[0]["status"] == "SUCCESS":
            with open(raw_data_dir / "metadata" / "{}.json".format(pid), "w") as f:
                json.dump(data[0]["object"], f)
        else:
            print(f"{pid=} failed with status {data[0]['status']}")
            if pid not in failed["metadata"]:
                failed["metadata"].append(pid)
                json.dump(failed, open(raw_data_dir / "failed_fetch.json", "w"))

        cached.add(pid)

        time.sleep(2)

    full_metadata = OrderedDict()

    for pid in tqdm(pids):
        full_metadata[pid] = json.load(open(raw_data_dir / "metadata" / f"{pid}.json"))

    with ZipFile(raw_data_dir / "full_metadata.zip", "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("full_metadata.json", json.dumps(full_metadata).encode("utf-8"))
