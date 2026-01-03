import kagglehub
import pandas as pd
from pathlib import Path


path = kagglehub.dataset_download("uciml/human-activity-recognition-with-smartphones")
base_path = Path(path)

train_df = pd.read_csv(base_path / "train.csv")
test_df  = pd.read_csv(base_path / "test.csv")

y_train_raw = train_df["Activity"]
y_test_raw  = test_df["Activity"]

X_train_df = train_df.drop(columns=["Activity", "subject"])
X_test_df  = test_df.drop(columns=["Activity", "subject"])

# müssen wir noch in ganze zahlen umwandeln für tensor
y_all = pd.concat([y_train_raw, y_test_raw], axis=0).astype("category")
codes = y_all.cat.codes
K = len(y_all.cat.categories)

class_names = y_all.cat.categories.tolist()

y_train = codes.iloc[:len(y_train_raw)]
y_test  = codes.iloc[len(y_train_raw):]


