import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_best_accuracy(
    generations,
    best_acc,
    title="Best Accuracy per Generation",
    xlabel="Generation",
    ylabel="Accuracy",
    save_dir=SAVE_DIR,
    filename="best_accuracy.png"
):
    plt.figure()
    plt.plot(generations, best_acc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if save_dir is not None:
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_avg_accuracy(
    generations,
    avg_acc,
    title="Average Accuracy per Generation",
    xlabel="Generation",
    ylabel="Accuracy",
    save_dir=SAVE_DIR,
    filename="avg_accuracy.png",
):
    plt.figure()
    plt.plot(generations, avg_acc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if save_dir is not None:
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_accuracy_comparison(
    methods,
    accuracies,
    title="Accuracy Comparison",
    ylabel="Accuracy",
    save_dir=SAVE_DIR,
    filename="accuracy_comparison.png",
):
    fig, ax = plt.subplots()

    ax.bar(methods, accuracies)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")

    # Y-Achse als Prozent formatieren
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y * 100:.1f}%")
    )

    fig.tight_layout()

    if save_dir is not None:
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_accuracy_vs_runtime(
    runtimes,
    accuracies,
    labels,
    title="Accuracy vs Runtime",
    xlabel="Runtime (seconds)",
    ylabel="Accuracy",
    save_dir=SAVE_DIR,
    filename="accuracy_vs_runtime.png",
):
    plt.figure()
    plt.scatter(runtimes, accuracies)

    for x, y, label in zip(runtimes, accuracies, labels):
        plt.text(x, y, label, fontsize=8, ha="right", va="bottom")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    if save_dir is not None:
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_convergence(
    visualize_data,
    title="Population Convergence",
    xlabel="Generation",
    ylabel="Convergence",
    save_dir=SAVE_DIR,
    filename="convergence.png"
):
    generations = [g for g, _, _, _ in visualize_data]
    convergence  = [c for _, _, _, c in visualize_data]

    plt.figure()
    plt.plot(generations, convergence)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if save_dir is not None:
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()

def export_hpo_comparison_csv(
        ga_individual,
        ga_fitness,
        optuna_trial,
        param_names,
        path="results/hpo_best_comparison.csv"
):
    # GA-Parameter als Dict
    ga_params = dict(zip(param_names, ga_individual))

    # Optuna-Parameter
    optuna_params = optuna_trial.params

    # Gemeinsame Spaltenmenge (Union)
    all_params = sorted(set(ga_params.keys()) | set(optuna_params.keys()))

    rows = []

    # --- GA ---
    ga_row = {
        "method": "GA",
        "accuracy": ga_fitness,
    }
    for p in all_params:
        ga_row[p] = ga_params.get(p, None)
    rows.append(ga_row)

    # --- Optuna ---
    optuna_row = {
        "method": "Optuna",
        "accuracy": optuna_trial.value,
    }
    for p in all_params:
        optuna_row[p] = optuna_params.get(p, None)
    rows.append(optuna_row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df

def create_confusion_matrix_plot(
    y_true,
    y_pred,
    class_names,
    title="Confusion Matrix (row-normalized)",
    save_dir=SAVE_DIR,
    filename="confusion_matrix.png"
):
    cm = confusion_matrix(y_true, y_pred)

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_normalized = cm.astype(float) / row_sums

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    for i, j in np.ndindex(cm.shape):
        plt.text(
            j, i,
            f"{cm_normalized[i, j]:.2f}\n({cm[i, j]})",
            ha="center",
            va="center",
            color="white" if cm_normalized[i, j] > 0.5 else "black"
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    if save_dir is not None:
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()
    return cm, cm_normalized
