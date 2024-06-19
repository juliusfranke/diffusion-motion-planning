import json
import sys
from icecream import ic
from typing import List, Any, Dict
import pathlib
import torch
import pandas as pd
from tabulate import tabulate
import uuid

JSON_DEFAULT = {
    "structure": {
        "uuid": "UUID",
        "type": "str",
        "n_hidden": "int",
        "s_hidden": "int",
        "dim_action": "int",
        "action_in": "bool",
        "action_out": "bool",
        "state_in": "bool",
        "state_out": "bool",
        "mse": "float",
    },
    "data": {},
}


def isFile(string: str):
    path = pathlib.Path(string)
    if path.is_file() and path.suffix == ".json":
        return True
    else:
        return False


class Database:
    def __init__(self, filename) -> None:
        self.filename = pathlib.Path(filename)
        if not isFile(filename):
            self._create()

        self.data = self._read()
        self.df = pd.DataFrame.from_dict(self.data["data"])
        self.check = []

    def _create(self) -> None:
        json_object = json.dumps(JSON_DEFAULT)
        with open(self.filename, "w") as file:
            file.write(json_object)

    def _read(self) -> dict:
        with open(self.filename, "r") as file:
            json_object = json.load(file)
        return json_object

    def addEntry(self, data: Dict, uuid: uuid.UUID):
        data["uuid"] = uuid
        if uuid in self.check:
            if data["mse"] > self.df[self.df["uuid"] == uuid]:
                print("mse is greater than existing model")
                print("model will not be safed")
                return None
            else:
                self.data["data"].pop(data)
        self.data["data"].append(data)

        json_object = json.dumps(self.data)
        with open(self.filename, "w") as file:
            file.write(json_object)

    def getUUID(self, data: Dict):
        check = None
        if self.data["data"]:
            print(self.data["data"])
            check = [
                item["uuid"]
                for item in self.data["data"]
                if data.items() <= item.items()
            ][0]
        if check:
            print("Entry already exists, train again? [y, n]")
            if input() in ["y", "yes"]:
                id = uuid.UUID(check)
                self.check.append(id)
                return id
            else:
                sys.exit()
        else:
            return uuid.uuid4()

    def tabulate(self, keys: List):
        if not set(keys) <= set(self.data["structure"].keys()):
            raise NameError
        rows = [
            [value for (key, value) in x.items() if key in keys]
            for x in self.data["data"]
        ]
        return tabulate(rows, keys)


def main():
    db = Database(filename="data.json")
    test = {
        "type": "car",
        "n_hidden": 4,
        "s_hidden": 256,
        "dim_action": 2,
        "dim_state": 3,
        "action_in": True,
        "action_out": True,
        "state_in": True,
        "state_out": True,
    }
    uuid = db.getUUID(test)
    test = {
        "type": "car",
        "n_hidden": 4,
        "s_hidden": 256,
        "dim_action": 2,
        "dim_state": 3,
        "action_in": True,
        "action_out": True,
        "state_in": True,
        "state_out": True,
        "mse": 0.1,
    }
    db.addEntry(test, uuid)


if __name__ == "__main__":
    main()
