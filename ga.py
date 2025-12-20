import math
import random
import torch

from nn import do_nn_training
from plotting import plot_data

torch.manual_seed(42)



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
    'scaler': {'type':'categorical','choices':['standard','minmax','none']}
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
        random.choice(search_space['scaler']['choices'])
    ]


def evaluate_fitness(individual):
    accuracy = do_nn_training(individual)
    return accuracy


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
    return child


def genetic_algorithm(
    visualize_data,
    pop_size=30,
    generations=20,
    tournament_size=3,
    mutation_rate=0.1,
    elitism=1,

):
    population = create_population(pop_size)

    for gen in range(generations):
        new_population = sorted(population, key=lambda x: x[1], reverse=True)[:elitism]

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
        best = max(population, key=lambda x: x[1])
        print(f"Generation {gen+1}: Best Fitness = {best[1]:.4f}")
        visualize_data.append((gen+1, best[1]))
    return max(population, key=lambda x: x[1])



# Start
# =========================
visualize_data = []
best_individual, best_fitness = genetic_algorithm(visualize_data)

generations = [x for x, _ in visualize_data]
accuracies  = [y for _, y in visualize_data]

plot_data(
    generations,
    accuracies,
    title="GA NN Hyperparameter Optimization",
    xlabel="Generation",
    ylabel="Accuracy"
)
print("\nBest Individual:", best_individual)
print("Best Fitness:", best_fitness)