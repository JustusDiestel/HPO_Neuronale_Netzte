import math
import random
import time
import optuna
import torch
import pandas as pd
import numpy as np

from data import X_train_df, X_val_df, X_test_df, y_train, y_val, y_test, K, class_names
from nn import do_nn_training
from plotting import (
    plot_best_accuracy,
    plot_avg_accuracy,
    plot_accuracy_comparison,
    plot_accuracy_vs_runtime,
    export_hpo_comparison_csv,
    create_confusion_matrix_plot,
    plot_convergence,
)
from config import SEED, PARAM_NAMES
from optunaBenchmark import optuna_objective
from scikitBenchmark import SklearnBenchmark


# Reproduzierbarkeit
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)




# Suchraum
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
    'scaler': {'type': 'categorical', 'choices': ['none']},  # Datensatz ist bereits skaliert ansonsten mit  ['standard', 'minmax', 'none']
    'epochs': {'type': 'int', 'bounds': [10, 100]},
}



# GA-Hilfsfunktionen
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


 # Fitness is defined as validation accuracy returned by do_nn_training
def evaluate_fitness(individual, runs):
    scores = []
    for i in range(runs):
        torch.manual_seed(SEED + i)
        random.seed(SEED + i)
        acc = do_nn_training(individual)
        scores.append(acc)

    fitness = sum(scores) / len(scores)
    return fitness


def create_population(pop_size):
    return [(ind := create_individual(), evaluate_fitness(ind, runs)) for _ in range(pop_size)]


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
    """
    Konvergenzmaß:
    - Diskrete Gene: Anteil des häufigsten Allels
    - Kontinuierliche Gene: 1 - (normierte Varianz)
    """
    individuals = [ind for ind, _ in population]
    pop_size = len(individuals)

    # Indizes der Gene
    discrete_genes = {0, 1, 2, 3, 5, 6, 9, 10}
    continuous_genes = {4, 7, 8}

    scores = []

    # Diskrete Gene
    for g in discrete_genes:
        alleles = [ind[g] for ind in individuals]
        most_common_freq = max(alleles.count(a) for a in set(alleles)) / pop_size
        scores.append(most_common_freq)

    # Kontinuierliche Gene
    for g in continuous_genes:
        values = np.array([ind[g] for ind in individuals])
        var = np.var(values)
        norm_var = var / (var + 1e-8)  # Stabilisierung
        scores.append(1.0 - norm_var)

    return float(np.mean(scores))



# Genetischer Algorithmus

def genetic_algorithm(
    visualize_data,
    start_time,
    pop_size=30,
    generations=10,
    tournament_size=3,
    mutation_rate=0.1,
    elitism=1,
    convergence_threshold=0.90,
):
    convergence_time = None
    conv_best_ind = None
    convergence_reached = False
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

            new_population.append((c1, evaluate_fitness(c1, runs)))
            if len(new_population) < pop_size:
                new_population.append((c2, evaluate_fitness(c2, runs)))

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

        if convergence >= convergence_threshold and not convergence_reached:
            convergence_time = time.perf_counter() - start_time
            print(
                f"Konvergenz erreicht: {convergence:.3f} >= {convergence_threshold} "
                f"nach {convergence_time:.2f}s"
            )
            conv_best_ind = best
            convergence_reached = True

    return population[0],conv_best_ind, convergence_time


# Main

if __name__ == "__main__":
    for runs in [1, 3]:
        visualize_data = []
        benchmark_results = []
        best_params = []

        # ----- GA -----
        start_time = time.perf_counter()
        convergence_time = None
        (best_individual, best_fitness),(best_individual_conv, conv_fitness), convergence_time = genetic_algorithm(visualize_data, start_time)
        elapsed_time = time.perf_counter() - start_time
        if best_individual_conv is None:
            conv_fitness = None
        ga_best_ind_per_generation = []

        benchmark_results.append({
            "method": "GA Convergence Stop",
            "accuracy": conv_fitness,
            "runtime": convergence_time
        })



        for i in range(len(visualize_data)):
            gen, best_acc, avg_accuracy, convergence = visualize_data[i]
            ga_best_ind_per_generation.append({
                "generation": gen,
                "best_accuracy": best_acc,
                "avg_accuracy": avg_accuracy,
                "convergence": convergence
            })
        df_ga_best_ind_per_generation = pd.DataFrame(ga_best_ind_per_generation)
        df_ga_best_ind_per_generation.to_csv(f"results/best_individuals_per_generation_runs_{runs}.csv", index=False)


        benchmark_results.append({
            "method": "GA (NN)",
            "accuracy": best_fitness,
            "runtime": elapsed_time
        })

        ga_best_params = dict(zip(PARAM_NAMES, best_individual))
        ga_conv_best_params = dict(zip(PARAM_NAMES, best_individual_conv))
        best_params.append(ga_best_params)
        best_params.append(ga_conv_best_params)

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


        # Ergebnisse & Plots

        create_confusion_matrix_plot(
            y_true=y_test,
            y_pred=do_nn_training(
                best_individual,
                return_predictions=True,
                train_final_model=True
            ),
            class_names= class_names,
        )

        plot_convergence(
            visualize_data=visualize_data,
            title=f"GA Convergence (runs={runs})",
            filename=f"convergence_runs_{runs}.png"
        )


        df_hpo = export_hpo_comparison_csv(
            ga_individual=best_individual,
            ga_fitness=best_fitness,
            ga_conv_individual=best_individual_conv,
            ga_conv_fitness=conv_fitness,
            optuna_trial=study.best_trial,
            param_names=PARAM_NAMES,
            path=f"results/hpo_best_comparison_runs_{runs}.csv"
        )

        df_comp_best_params = pd.DataFrame(best_params)

        df_results = pd.DataFrame(benchmark_results)
        print(df_results)
        df_results.to_csv(
            f"results/benchmark_results_runs_{runs}.csv",
            index=False
        )
        generations = [g for g, _, _, _ in visualize_data]
        best_acc    = [b for _, b, _, _ in visualize_data]
        avg_acc     = [a for _, _, a, _ in visualize_data]

        plot_best_accuracy(
            generations,
            best_acc,
            title=f"Best Accuracy per Generation (GA NN, runs={runs})",
            filename=f"best_accuracy_ga_nn_runs_{runs}.png"

        )

        plot_avg_accuracy(
            generations,
            avg_acc,
            title=f"Average Accuracy per Generation (GA NN, runs={runs})",
            filename=f"avg_accuracy_ga_nn_runs_{runs}.png"
        )

        plot_accuracy_comparison(
            df_results["method"].tolist(),
            df_results["accuracy"].tolist(),
            title=f"Accuracy Comparison (runs={runs})",
            filename=f"accuracy_comparison_runs_{runs}.png"
        )

        plot_accuracy_vs_runtime(
            df_results["runtime"].tolist(),
            df_results["accuracy"].tolist(),
            df_results["method"].tolist(),
            title=f"Accuracy vs Runtime (runs={runs})",
            filename=f"accuracy_vs_runtime_runs_{runs}.png"
        )