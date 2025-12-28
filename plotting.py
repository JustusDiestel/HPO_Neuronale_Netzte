import matplotlib.pyplot as plt
import os

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
    plt.figure()
    plt.bar(methods, accuracies)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")

    if save_dir is not None:
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
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
        plt.text(x, y, label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    if save_dir is not None:
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

