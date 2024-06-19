import json
import sys
from icecream import ic
from typing import List, Any, Dict
import pathlib
import torch
import pandas as pd
import numpy as np
import uuid

STRUCTURE = {
    "uuid": "UUID",
    "type": "str",
    "n_hidden": "int",
    "s_hidden": "int",
    "dim_action": "int",
    "dim_state": "int",
    "action_in": "bool",
    "action_out": "bool",
    "state_in": "bool",
    "state_out": "bool",
    "mse": "float",
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
        self.check = []

    def _create(self) -> None:
        data = pd.DataFrame(columns=list(STRUCTURE.keys()))
        data.to_json(self.filename)
        # json_object = json.dumps(data)
        # with open(self.filename, "w") as file:
        #     file.write(json_object)

    def _write(self) -> None:
        self.data.to_json(self.filename, orient="records", default_handler=str)

    def _read(self) -> pd.DataFrame:
        df = pd.read_json(self.filename)
        return df

    def addEntry(self, data: Dict, uuid: uuid.UUID):
        data["uuid"] = uuid
        if uuid in self.check:
            # breakpoint()
            if data["mse"] > self.data[self.data["uuid"] == uuid]["mse"].tolist()[0]:
                print("mse is greater than existing model")
                print("model will not be safed")
                return None
            # else:
            # id = self.data[self.data["uuid"] == uuid].index.values.astype(int)[0]
            # self.data["data"].drop(id)
        # self.data["data"].append(data)
        if len(self.data.index) > 0:
            self.data.loc[max(self.data.index) + 1] = data
        else:
            self.data.loc[0] = data

        self._write()
        # json_object = json.dumps(self.data)
        # with open(self.filename, "w") as file:
        #     file.write(json_object)

    def getUUID(self, data: Dict):
        # breakpoint()
        if self.data.empty:
            return uuid.uuid4()
        id = self.data[
            (
                self.data.loc[
                    :,
                    np.logical_and(
                        self.data.columns != "uuid", self.data.columns != "mse"
                    ),
                ]
                == data
            ).all(1)
        ]["uuid"].tolist()[0]
        # breakpoint()
        # check = [
        #     item["uuid"] for item in self.data["data"] if data.items() <= item.items()
        # ][0]
        if id:
            print("Entry already exists, train again? [y, n]")
            if input() in ["y", "yes"]:
                self.check.append(id)
                return id
            else:
                sys.exit()
        else:
            return uuid.uuid4()

    def tabulate(self, keys: List):
        if not set(keys) <= set(self.data.columns):
            raise NameError
        # rows = [
        #     [value for (key, value) in x.items() if key in keys]
        #     for x in self.data["data"]
        # ]

        return self.data[keys]


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
    print(test)
    _uuid = db.getUUID(test)
    test["mse"] = 0.1
    db.addEntry(test, _uuid)
    test["s_hidden"] = 128
    test["mse"] = 0.2
    db.addEntry(test, uuid.uuid4())
    test.pop("mse")
    test.pop("uuid")
    print(test)
    # breakpoint()
    _uuid = db.getUUID(test)
    test["mse"] = 0.15
    db.addEntry(test, _uuid)
    print(type(uuid))
    # test_1 = {
    #     "type": "car",
    #     "n_hidden": 4,
    #     "s_hidden": 256,
    #     "dim_action": 2,
    #     "dim_state": 3,
    #     "action_in": True,
    #     "action_out": True,
    #     "state_in": True,
    #     "state_out": True,
    #     "mse": 0.1,
    # }
    # test_2 = {
    #     "type": "car",
    #     "n_hidden": 4,
    #     "s_hidden": 128,
    #     "dim_action": 2,
    #     "dim_state": 3,
    #     "action_in": True,
    #     "action_out": True,
    #     "state_in": True,
    #     "state_out": True,
    #     "mse": 0.2,
    # }
    # db.addEntry(test_1, uuid)
    # db.addEntry(test_2, uuid.uuid4())


if __name__ == "__main__":
    main()
