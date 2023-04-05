import json
from pathlib import Path

data_path = Path(__file__).parent

def load_json(filename):
    with open(data_path / filename, 'r') as f:
        return json.load(f)
