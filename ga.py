import math
import random
import time

import torch
import pandas as pd

from data import X_train_df, X_test_df, y_train, K, y_test
from nn import do_nn_training
from plotting import plot_best_accuracy, plot_avg_accuracy, plot_accuracy_comparison, plot_accuracy_vs_runtime
import numpy as np
from config import SEED

from scikitBenchmark import SklearnBenchmark


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

fitness_cache = {}



search_space = {
    'num_layers': {'type':'int','bounds':[1,4]},
    'base_units': {'type':'int','choices':[32,64,128]},
    'width_pattern': {'type':'categorical','choices':['constant','increasing','decreasing']},
    'activation': {'type':'categorical','choices':['relu','tanh','elu']},
    'learning_rate': {'type':'float','bounds':[1e-5,1e-1],'scale':'log'},
    'optimizer': {'type':'categorical','choices':['adam','rmsprop']},
    'batch_size': {'type':'int','choices':[16,32,64,128,256]},
    'dropout_rate': {'type':'float','bounds':[0.0,0.5]},
    'l2_weight_decay': {'type':'float','bounds':[0.0,1e-2]},
    'scaler': {'type':'categorical','choices':['standard','minmax','none']},
    'epochs': {'type':'int','bounds':[10,100]},
}


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
    """
    Fitness = Mittelwert über mehrere Trainingsläufe
    + Caching identischer Individuen
    """
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
    population = []
    for _ in range(pop_size):
        ind = create_individual()
        fit = evaluate_fitness(ind)
        population.append((ind, fit))
    return population


def tournament_selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x[1])  # (ind, fit)


def crossover(parent1, parent2):
    p1, p2 = parent1[0], parent2[0]
    point = random.randint(1, len(p1) - 1)
    child1 = p1[:point] + p2[point:]
    child2 = p2[:point] + p1[point:]
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


def genetic_algorithm(
    visualize_data,
    pop_size=30,
    generations=10,
    tournament_size=3,
    mutation_rate=0.1,
    elitism=1,
):
    population = create_population(pop_size)

    for gen in range(generations):
        population = sorted(population, key=lambda x: x[1], reverse=True)

        # Elitismus
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
        avg_fitness = sum(f for _, f in population) / len(population)

        print(
            f"Generation {gen+1}: "
            f"Best = {best[1]:.4f}, "
            f"Avg = {avg_fitness:.4f}"
        )

        visualize_data.append((gen + 1, best[1], avg_fitness))

    return population[0]



if __name__=="__main__":
    visualize_data = []
    benchmark_results = []
    start_time = time.perf_counter()

    best_individual, best_fitness = genetic_algorithm(visualize_data)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: %.2f seconds" % elapsed_time)

    benchmark_results.append({
        "method": "GA (NN)",
        "accuracy": best_fitness,
        "runtime": elapsed_time
    })

    generations = [x for x, _, _ in visualize_data]
    best_acc    = [y for _, y, _ in visualize_data]
    avg_acc     = [z for _, _, z in visualize_data]


    print("\nBest Individual:", best_individual)
    print("Best Fitness:", best_fitness)

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

    df_results = pd.DataFrame(benchmark_results)
    print(df_results)
    df_results.to_csv("benchmark_results.csv", index=False)

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
        title="Accuracy vs. Runtime",
        filename="accuracy_vs_runtime.png"
    )
