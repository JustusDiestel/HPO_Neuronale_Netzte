import kagglehub
import pandas as pd
from pathlib import Path


path = kagglehub.dataset_download("uciml/human-activity-recognition-with-smartphones")
base_path = Path(path)

train_df = pd.read_csv(base_path / "train.csv")
test_df  = pd.read_csv(base_path / "test.csv")

from sklearn.model_selection import train_test_split

# --- Original labels ---
y_trainval_raw = train_df["Activity"]
y_test_raw     = test_df["Activity"]

X_trainval_df = train_df.drop(columns=["Activity", "subject"])
X_test_df     = test_df.drop(columns=["Activity", "subject"])

# --- Train / Validation split ---
X_train_df, X_val_df, y_train_raw, y_val_raw = train_test_split(
    X_trainval_df,
    y_trainval_raw,
    test_size=0.2,
    stratify=y_trainval_raw,
    random_state=42
)

y_all = pd.concat([y_train_raw, y_val_raw, y_test_raw], axis=0).astype("category")
codes = y_all.cat.codes

K = len(y_all.cat.categories)
class_names = y_all.cat.categories.tolist()

y_train = codes.iloc[:len(y_train_raw)]
y_val   = codes.iloc[len(y_train_raw):len(y_train_raw) + len(y_val_raw)]
y_test  = codes.iloc[len(y_train_raw) + len(y_val_raw):]


