import json
import zipfile

from . import data_path

in_path = data_path / 'members.zip'

members = {}

# Re-hydrate the members from the zip file
with zipfile.ZipFile(in_path, "r") as zf:
    pids = json.load(zf.open("pids.json"))
    for pid in pids:
        members[pid] = json.load(zf.open(f"{pid}.json"))
