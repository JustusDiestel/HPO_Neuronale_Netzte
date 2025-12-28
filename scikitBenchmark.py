import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from config import SEED

RandomForestClassifier(random_state=SEED)
GradientBoostingClassifier(random_state=SEED)
MLPClassifier(random_state=SEED)

class SklearnBenchmark:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(),
            "GradientBoosting": GradientBoostingClassifier(),
            "MLP(Default)": MLPClassifier(max_iter=300),
        }

        self.results = {}

    def run(self):
        for name, model in self.models.items():
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])

            start = time.perf_counter()
            pipeline.fit(self.X_train, self.y_train)
            runtime = time.perf_counter() - start

            acc = accuracy_score(self.y_test, pipeline.predict(self.X_test))

            self.results[name] = {
                "accuracy": acc,
                "runtime": runtime
            }

            print(f"[SK] {name} | Acc={acc:.4f} | Time={runtime:.2f}s")

        return self.results