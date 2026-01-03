import math
import random
import time
import optuna
import torch
import pandas as pd
import numpy as np

from data import X_train_df, X_test_df, y_train, y_test, K
from nn import do_nn_training
from plotting import (
    plot_best_accuracy,
    plot_avg_accuracy,
    plot_accuracy_comparison,
    plot_accuracy_vs_runtime,
)
from config import SEED, PARAM_NAMES
from optunaBenchmark import optuna_objective
from scikitBenchmark import SklearnBenchmark


# =====================
# Reproduzierbarkeit
# =====================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

fitness_cache = {}


# =====================
# Suchraum
# =====================
search_space = {
    'num_layers': {'type': 'int', 'bounds': [1, 4]},
    'base_units': {'type': 'int', 'choices': [32, 64, 128]},
    'width_pattern': {'type': 'categorical', 'choices': ['constant', 'increasing', 'decreasing']},
    'activation': {'type': 'categorical', 'choices': ['relu', 'tanh', 'elu']},
    'learning_rate': {'type': 'float', 'bounds': [1e-5, 1e-1], 'scale': 'log'},
    'optimizer': {'type': 'categorical', 'choices': ['adam', 'rmsprop']},
    'batch_size': {'type': 'int', 'choices': [16, 32, 64, 128, 256]},
    'dropout_rate': {'type': 'float', 'bounds': [0.0, 0.5]},
    'l2_weight_decay': {'type': 'float', 'bounds': [0.0, 1e-2]},
    'scaler': {'type': 'categorical', 'choices': ['standard', 'minmax', 'none']},
    'epochs': {'type': 'int', 'bounds': [10, 100]},
}


# =====================
# GA-Hilfsfunktionen
# =====================
def create_individual():
    a, b = search_space['learning_rate']['bounds']
    return [
        random.randint(*search_space['num_layers']['bounds']),
        random.choice(search_space['base_units']['choices']),
        random.choice(search_space['width_pattern']['choices']),
        random.choice(search_space['activation']['choices']),
        10 ** random.uniform(math.log10(a), math.log10(b)),
        random.choice(search_space['optimizer']['choices']),
        random.choice(search_space['batch_size']['choices']),
        random.uniform(*search_space['dropout_rate']['bounds']),
        random.uniform(*search_space['l2_weight_decay']['bounds']),
        random.choice(search_space['scaler']['choices']),
        random.randint(*search_space['epochs']['bounds']),
    ]


def evaluate_fitness(individual, runs=3):
    key = tuple(individual)
    if key in fitness_cache:
        return fitness_cache[key]

    scores = []
    for i in range(runs):
        torch.manual_seed(SEED + i)
        random.seed(SEED + i)
        acc = do_nn_training(individual)
        scores.append(acc)

    fitness = sum(scores) / len(scores)
    fitness_cache[key] = fitness
    return fitness


def create_population(pop_size):
    return [(ind := create_individual(), evaluate_fitness(ind)) for _ in range(pop_size)]


def tournament_selection(population, tournament_size):
    return max(random.sample(population, tournament_size), key=lambda x: x[1])


def crossover(parent1, parent2):
    p1, p2 = parent1[0], parent2[0]
    c1, c2 = sorted(random.sample(range(1, len(p1)), 2))
    child1 = p1[:c1] + p2[c1:c2] + p1[c2:]
    child2 = p2[:c1] + p1[c1:c2] + p2[c2:]
    return child1, child2


def mutate(individual, mutation_rate):
    child = individual.copy()
    for i in range(len(child)):
        if random.random() < mutation_rate:
            if i == 0:
                child[i] = random.randint(*search_space['num_layers']['bounds'])
            elif i == 1:
                child[i] = random.choice(search_space['base_units']['choices'])
            elif i == 2:
                child[i] = random.choice(search_space['width_pattern']['choices'])
            elif i == 3:
                child[i] = random.choice(search_space['activation']['choices'])
            elif i == 4:
                a, b = search_space['learning_rate']['bounds']
                child[i] = 10 ** random.uniform(math.log10(a), math.log10(b))
            elif i == 5:
                child[i] = random.choice(search_space['optimizer']['choices'])
            elif i == 6:
                child[i] = random.choice(search_space['batch_size']['choices'])
            elif i == 7:
                child[i] = random.uniform(*search_space['dropout_rate']['bounds'])
            elif i == 8:
                child[i] = random.uniform(*search_space['l2_weight_decay']['bounds'])
            elif i == 9:
                child[i] = random.choice(search_space['scaler']['choices'])
            elif i == 10:
                child[i] = random.randint(*search_space['epochs']['bounds'])
    return child


def population_convergence(population):
    individuals = [ind for ind, _ in population]
    pop_size = len(individuals)
    num_genes = len(individuals[0])

    freqs = []
    for g in range(num_genes):
        alleles = [ind[g] for ind in individuals]
        most_common = max(set(alleles), key=alleles.count)
        freqs.append(alleles.count(most_common) / pop_size)

    return sum(freqs) / num_genes


# =====================
# Genetischer Algorithmus
# =====================
def genetic_algorithm(
    visualize_data,
    pop_size=30,
    generations=10,
    tournament_size=3,
    mutation_rate=0.1,
    elitism=1,
    convergence_threshold=0.90,
):
    population = create_population(pop_size)

    for gen in range(generations):
        population.sort(key=lambda x: x[1], reverse=True)
        new_population = population[:elitism]

        while len(new_population) < pop_size:
            p1 = tournament_selection(population, tournament_size)
            p2 = tournament_selection(population, tournament_size)

            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)

            new_population.append((c1, evaluate_fitness(c1)))
            if len(new_population) < pop_size:
                new_population.append((c2, evaluate_fitness(c2)))

        population = new_population

        best = population[0]
        avg_fitness = sum(f for _, f in population) / pop_size
        convergence = population_convergence(population)

        visualize_data.append((gen + 1, best[1], avg_fitness, convergence))

        print(
            f"Generation {gen + 1}: "
            f"Best = {best[1]:.4f}, "
            f"Avg = {avg_fitness:.4f}, "
            f"Conv = {convergence:.3f}"
        )

        if convergence >= convergence_threshold:
            print(f"Population converged at generation {gen + 1}")
            break

    return population[0]


# =====================
# Main
# =====================
if __name__ == "__main__":
    visualize_data = []
    benchmark_results = []
    best_params = []

    # ----- GA -----
    start_time = time.perf_counter()
    best_individual, best_fitness = genetic_algorithm(visualize_data)
    elapsed_time = time.perf_counter() - start_time

    benchmark_results.append({
        "method": "GA (NN)",
        "accuracy": best_fitness,
        "runtime": elapsed_time
    })

    ga_best_params = dict(zip(PARAM_NAMES, best_individual))
    best_params.append(ga_best_params)

    # ----- Optuna -----
    start_time = time.perf_counter()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(optuna_objective, n_trials=30, show_progress_bar=True)
    elapsed_time = time.perf_counter() - start_time

    benchmark_results.append({
        "method": "Optuna (NN)",
        "accuracy": study.best_value,
        "runtime": elapsed_time
    })


    best_params.append(study.best_params)

    # ----- Sklearn -----
    sk_benchmark = SklearnBenchmark(
        X_train_df,
        y_train,
        X_test_df,
        y_test
    )

    sk_results = sk_benchmark.run()

    for name, res in sk_results.items():
        benchmark_results.append({
            "method": name,
            "accuracy": res["accuracy"],
            "runtime": res["runtime"]
        })

    # =====================
    # Ergebnisse & Plots
    # =====================
    df_results = pd.DataFrame(benchmark_results)
    print(df_results)
    df_results.to_csv("results/benchmark_results.csv", index=False)

    generations = [g for g, _, _, _ in visualize_data]
    best_acc    = [b for _, b, _, _ in visualize_data]
    avg_acc     = [a for _, _, a, _ in visualize_data]

    plot_best_accuracy(
        generations,
        best_acc,
        title="Best Accuracy per Generation (GA NN)",
        filename="best_accuracy_ga_nn.png"
    )

    plot_avg_accuracy(
        generations,
        avg_acc,
        title="Average Accuracy per Generation (GA NN)",
        filename="avg_accuracy_ga_nn.png"
    )

    plot_accuracy_comparison(
        df_results["method"].tolist(),
        df_results["accuracy"].tolist(),
        title="Accuracy Comparison",
        filename="accuracy_comparison.png"
    )

    plot_accuracy_vs_runtime(
        df_results["runtime"].tolist(),
        df_results["accuracy"].tolist(),
        df_results["method"].tolist(),
        title="Accuracy vs Runtime",
        filename="accuracy_vs_runtime.png"
    )