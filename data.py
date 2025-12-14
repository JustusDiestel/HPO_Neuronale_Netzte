import kagglehub
import pandas as pd
from pathlib import Path
from pandas import Series
# ich lade hier die Daten aus dem Datensatz rein.
# die Daten sind bereits bereinigt un in train und data unterteilt - keine standardisierung und so notwendig

path = kagglehub.dataset_download("uciml/human-activity-recognition-with-smartphones")

print("Path to dataset files:", path)



base_path = Path(path)
train_data = pd.read_csv(base_path / "train.csv")
test_data = pd.read_csv(base_path / "test.csv")

test_result = test_data["Activity"]
train_result = train_data["Activity"]
test_data = test_data.drop(columns=["Activity", "subject"])
train_data = train_data.drop(columns=["Activity", "subject"])

print(test_data.head())


def get_test_data() -> pd.DataFrame:
    return test_data

def get_test_result() -> Series:
    return test_result

def get_train_data() -> pd.DataFrame:
    return train_data

def get_train_result() -> Series:
    return train_result