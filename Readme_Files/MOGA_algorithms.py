from deap.benchmarks import dtlz1
from deap.tools import crossover, mutation
import copy
import time
from operator import itemgetter
import random
import dask
import numpy as numpy
import matplotlib.pyplot as plt

class NSGAStructure:
    def __init__(self, p_mutation, objects, total_numbers, max_iterations, object_evaluations, generate_start_population, mutation_operator,
                 target_functions, crossover_operator, m_distribution_index, c_distribution_index, combined_method, total_population):

        if total_population % 2 != 0:
            raise ValueError

        if objects != len(target_functions):
            raise ValueError

        self.target_functions = target_functions
        self.n = total_numbers
        self.max_iterations = max_iterations
        self.fronts = numpy.zeros(total_population)
        self.combined_total_population = 2 * total_population
        self.total_population = total_population
        self.combined_fronts = numpy.zeros(total_population*2)
        self.objects = objects
        self.object_evaluations = object_evaluations
        self.generate_start_population = generate_start_population
        self.m_distribution_index = m_distribution_index
        self.crossover_operator = crossover_operator
        self.c_distribution_index = c_distribution_index
        self.points_per_front = []
        self.combined_method = combined_method
        self.p_mutation = p_mutation
        self.mutation_operator = mutation_operator

    def call_algo_nsgaii(self):
        
        self.current_algorithm = NSGAII()
        
        # Creating sample population
        self.population_representation = self.generate_start_population(self.total_population)

        # Calling objective function with given population for evaluation

        self.evaluations = self.object_evaluations(self.population_representation)
        #  Calling function for pareto fronts and its objects
        self.fronts, self.points_per_front = non_dominated_sorting(self.evaluations, self.target_functions)

        # Generating offspring population and evaluation  after getting basic evaluations
        
        self.offspring_population_representation, self.offspring_evaluations = self.combined_method(
            self.population_representation, self.n, self.fronts, numpy.empty(0),self.p_mutation, self.object_evaluations, compare_binary_selection, self.crossover_operator, self.mutation_operator, self.c_distribution_index, self.m_distribution_index)


        for current_iteration in numpy.arange(self.max_iterations):

            print('Current Iteration: ', current_iteration)

            # For further operation , we are combining the basic and off spring evaluations
            
            self.merged_population_representation = numpy.concatenate(
                (self.population_representation, self.offspring_population_representation), axis=0)
            
            self.merged_evaluations = numpy.concatenate((self.evaluations, self.offspring_evaluations), axis=0)

            # Based on pareto fronts i.e non-domination sorting merged population

            self.fronts, self.points_per_front = non_dominated_sorting(self.merged_evaluations, self.target_functions)

            #  filtering the elements of function
            self.population_representation,self.evaluations, self.crowding_assignment = self.current_algorithm.selection_of_population(self.points_per_front,
                                                                                                                                 self.total_population, self.merged_population_representation, self.merged_evaluations)
            # For MOGA algorithm , 3 real valued variables are used which are
            # 1. Crossover
            # 2. Polynomial Mutation
            # 3. Selection
            # Based on that , the offspring population will be created
        
            self.offspring_population_representation, self.offspring_evaluations = self.combined_method(
                self.population_representation, self.n, self.fronts, self.crowding_assignment, self.p_mutation, self.object_evaluations,
                crowded_binary_selection, self.crossover_operator, self.mutation_operator, self.c_distribution_index, self.m_distribution_index)

        # After every iteration , updating the generation count
        self.fronts, self.points_per_front = non_dominated_sorting(self.evaluations, self.target_functions)

    def call_algo_nsgaiii(self):

        self.current_algorithm = NSGAIII()

        self.reference_points = get_all_reference_points(self.objects, 12)
        self.num_reference_points = len(self.reference_points)

        self.min_value = numpy.full((self.objects), numpy.Inf)
        self.max_value = numpy.full((self.objects, self.objects), numpy.NINF)

        # Creating dynamic population for evaluation
        self.population_representation = self.generate_start_population(self.total_population)
        
        # based on created population getting the evaluation
        
        self.evaluations = self.object_evaluations(self.population_representation)
        self.fronts, self.points_per_front = non_dominated_sorting(self.evaluations, self.target_functions)

        # As we have calculated the evaluations, now creating offspring population
        
        self.offspring_population_representation, self.offspring_evaluations = self.combined_method(
            self.population_representation, self.n, self.fronts, numpy.empty(0), self.p_mutation, self.object_evaluations,
            compare_binary_selection, self.crossover_operator, self.mutation_operator, self.c_distribution_index, self.m_distribution_index)

        for current_iteration in numpy.arange(self.max_iterations):
            print('Current Iteration: ', current_iteration)

            # For further operation , we are combining the basic and off spring evaluations
            self.merged_population_representation = numpy.concatenate(
                (self.population_representation, self.offspring_population_representation), axis=0)
            self.merged_evaluations = numpy.concatenate((self.evaluations, self.offspring_evaluations), axis=0)

            # Based on pareto fronts i.e non-domination sorting merged population

            self.fronts, self.points_per_front = non_dominated_sorting(self.merged_evaluations,
                                                                         self.target_functions)

            #  filtering the elements of function
            self.population_representation, self.evaluations = self.current_algorithm.selection_of_population(self.points_per_front,
                                                                                              self.total_population,
                                                                                              self.merged_population_representation,
                                                                                              self.merged_evaluations,
                                                                                              self.target_functions,
                                                                                              self.min_value, self.max_value,
                                                                                              self.reference_points)

            # For MOGA algorithm , 3 real valued variables are used which are
            # 1. Crossover
            # 2. Polynomial Mutation
            # 3. Selection
            # Based on that , the offspring population will be created
            self.offspring_population_representation, self.offspring_evaluations = self.combined_method(
                self.population_representation, self.n, self.fronts, numpy.empty(0), self.p_mutation,
                self.object_evaluations,
                compare_binary_selection, self.crossover_operator, self.mutation_operator, self.c_distribution_index, self.m_distribution_index)

        # After every iteration , updating the generation count
        self.fronts, self.points_per_front = non_dominated_sorting(self.evaluations, self.target_functions)


def compare_operators(first_indicator, second_indicator, fronts):
    return fronts[first_indicator] < fronts[second_indicator]

def compare_crowded_operator(first_indicator, second_indicator, fronts, crowding_assignment):
    return fronts[first_indicator] < fronts[second_indicator] or (fronts[first_indicator] == fronts[second_indicator] 
                                                                  and crowding_assignment[first_indicator] > 
                                                                  crowding_assignment[second_indicator])

def crowded_binary_selection(first_indicator, second_indicator, fronts, crowding_assignment):

    if compare_crowded_operator(first_indicator, second_indicator, fronts, crowding_assignment):
        return first_indicator
    else:
        return second_indicator

def compare_binary_selection(first_indicator, second_indicator, fronts, crowding_assignment):

    if compare_operators(first_indicator, second_indicator, fronts):
        return first_indicator
    else:
        return second_indicator
    
# Below function creates reference points evenly with hyper
# plane on axis = 1. To combine the layers of reference points scaling factors are used.

def get_all_reference_points(current_object, p, scaling=None):
  
    def recursively_get_reference_points(reference_points, current_object, left, total, level):
        points = []
        if level == current_object - 1:
            reference_points[level] = left / total
            points.append(reference_points)
        else:
            for i in range(left + 1):
                reference_points[level] = i / total
                points.extend(recursively_get_reference_points(reference_points.copy(), current_object, left - i, total, level + 1))
        return points

    updated_reference_points = numpy.array(recursively_get_reference_points(numpy.zeros(current_object), current_object, p, p, 0))
    if scaling is not None:
        updated_reference_points *= scaling
        updated_reference_points += (1 - scaling) / current_object

    return updated_reference_points

def plot_of_calculated_evaluations(evaluations, problem):

    x = evaluations[:, 0]
    y = evaluations[:, 1]

    plt.scatter(x, y, c='blue')
    plt.title(problem.get_suite_name())
    plt.xlabel('function1')
    plt.ylabel('function2')

    plt.show()


def plot_3d_of_calculated_evaluations(evaluations):
    fig = plt.figure(figsize=(7,7))
    axis = fig.add_subplot(111, projection='3d')

    x = evaluations[:, 0]
    y = evaluations[:, 1]
    z = evaluations[:, 2]

    axis.scatter(x, y, z, color='navy', marker='o')

    # final figure details
    axis.set_xlabel('$f_1()$', fontsize=15)
    axis.set_ylabel('$f_2()$', fontsize=15)
    axis.set_zlabel('$f_3()$', fontsize=15)
    axis.view_init(elev=11, azim=-21)
    plt.autoscale(tight=True)
    plt.savefig("test.svg", format="svg")
    plt.show()


def plot_given_reference_points(reference_points):
    x = []
    y = []
    for point in reference_points:
        x.append(point[0])
        y.append(point[1])

    plt.scatter(x, y)

    plt.show()

def plot_3d_reference_points(reference_points):
    fig = plt.figure(figsize=(7, 7))
    axis = fig.add_subplot(111, projection='3d')

    x = []
    y = []
    z = []

    for point in reference_points:
        x.append(point[0])
        y.append(point[1])
        z.append(point[2])

    axis.scatter(x,y,z, color='navy', marker='o')

    # final figure details
    axis.set_xlabel('$f_1()$', fontsize=15)
    axis.set_ylabel('$f_2()$', fontsize=15)
    axis.set_zlabel('$f_3()$', fontsize=15)
    axis.view_init(elev=11, azim=-21)
    plt.autoscale(tight=True)
    plt.show()

# Updating ideal points
def ideal_points_update(costs,current_min_value, j):
    new_ideal = numpy.min(costs[:, j])
    current_min_value[j] = min(new_ideal, current_min_value[j])

# Use Normalization on structured points of objective functions
def normalize(combined_evaluations, structured_points, objects, current_min_value, current_max_value):
    # Merged evaluation
    merged_evaluations = combined_evaluations[structured_points]

    for M in numpy.arange(objects):
        ideal_points_update(merged_evaluations, current_min_value, M)
        perform_scalarization(merged_evaluations, current_max_value, M)

    translated_objectives = merged_evaluations - current_min_value

    a = get_intercepts_of_hyperplane(merged_evaluations, current_max_value)

    # Getting normalized evaluation
    normalized_evaluations = numpy.zeros((len(structured_points), objects))

    normalized_evaluations = translated_objectives / a

    return normalized_evaluations

def perform_scalarization(evaluations, current_max_value, j):
    ind = get_extreme_values_obj_function(evaluations, j)

    if evaluations[ind][j] > current_max_value[j][j]:
        current_max_value[j] = evaluations[ind]

# It collects the extreme values for objective functions

def get_extreme_values_obj_function(evaluations_values, j):
    return numpy.argmax(evaluations_values[:, j])


def get_intercepts_of_hyperplane(evaluations_values, current_max_value):
    # Handle duplicate scenarios
    objects = evaluations_values.shape[1]
    hyperplane_intercepts = numpy.zeros(objects)

    if numpy.all(evaluations_values == numpy.unique(evaluations_values, axis=0)):
        for j in range(objects):
            hyperplane_intercepts[j] = current_max_value[j][j]
    else:
        b = numpy.ones(objects)
        try:
            x = numpy.linalg.solve(current_max_value,b)
        except numpy.linalg.LinAlgError:
            for j in range(objects):
                hyperplane_intercepts[j] = current_max_value[j][j]
        else:
            hyperplane_intercepts = 1/x

    # Returning all hyperplane intercepts
    return hyperplane_intercepts

# Checking if the operator is dominant
def is_operator_dominant(u, v):
    objects = u.shape[0]

    for j in numpy.arange(objects):
        if u[j] > v[j]:
            return False

    return not numpy.array_equal(u, v)


# Sorting non_dominant solutions

def non_dominated_sorting(evaluations_values, target_functions):

    current_object = len(target_functions)
    normalized_evaluations = numpy.copy(evaluations_values)
    total_population = evaluations_values.shape[0]

    dominated_solutions = numpy.empty(total_population, dtype=object)
    domination_counts = numpy.zeros(total_population)
    fronts = numpy.full(total_population, numpy.inf)
    points_per_front = []
    first_front_points = []

    for j in numpy.arange(current_object):
        if target_functions[j] == 1:
            normalized_evaluations[:, j] *= -1
    # Checking if p is dominating q , if then q to solutions dominant by p
    for p in numpy.arange(total_population):

        dominated_solutions[p] = set()

        for q in numpy.arange(total_population):

            if is_operator_dominant(normalized_evaluations[p], normalized_evaluations[q]):
                dominated_solutions[p].add(q)
            elif is_operator_dominant(normalized_evaluations[q], normalized_evaluations[p]):
                domination_counts[p] += 1

        if domination_counts[p] == 0:
            fronts[p] = 0
            first_front_points.append(p)

    points_per_front.append(first_front_points)
    # Initializing the first counter
    current_front_counter = 0
    while not len(points_per_front[current_front_counter]) == 0:
        # To save the next front members
        Q = set()

        for p in points_per_front[current_front_counter]:
            for q in dominated_solutions[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    fronts[q] = current_front_counter + 1
                    Q.add(q)
        current_front_counter += 1
        points_per_front.append(list(Q))

    return fronts, points_per_front[:-1]

def get_perpendicular_distance(direction, point):
    k = numpy.dot(direction, point) / numpy.sum(numpy.power(direction, 2))
    d = numpy.sum(numpy.power(numpy.subtract(numpy.multiply(direction, [k] * len(direction)), point), 2))

    return numpy.sqrt(d)

def associate(structured_points, normalized_evaluations, reference_points, next_generation):

    reference_points_assignment = numpy.zeros(len(structured_points))
    perpendicular_distances_of_reference_points = numpy.zeros(len(structured_points))
    association_counts_next_generation = numpy.zeros(len(reference_points))
    association_counts_structured_points = numpy.zeros(len(reference_points))
    # Calculating distance between all reference points and ith point in st
    # Selecting closest ref points and assign them accordingly
    for i in range(len(structured_points)):

        reference_points_distances = numpy.fromiter(
            [get_perpendicular_distance(normalized_evaluations[i], reference_point) for reference_point in
             reference_points], float)

        nearest_reference_point = numpy.argmin(reference_points_distances)
        perpendicular_distances_of_reference_points[i] = reference_points_distances[nearest_reference_point]
        reference_points_assignment[i] = nearest_reference_point
        association_counts_structured_points[nearest_reference_point] += 1
        if structured_points[i] in next_generation:
            association_counts_next_generation[nearest_reference_point] += 1

    return reference_points_assignment, association_counts_structured_points, association_counts_next_generation, perpendicular_distances_of_reference_points


def get_closest_reference_point(point, reference_points):
    # Calculating distance between all reference points and ith point in st
    # Selecting closest ref points and assign them accordingly
    reference_points_distances = [get_perpendicular_distance(point, reference_point) for reference_point in reference_points]
    nearest_reference_point = numpy.argmin(reference_points_distances)

    return nearest_reference_point, reference_points_distances[nearest_reference_point]


# Dask association

def associate_dask(structured_points, normalized_evaluations, reference_points, next_generation):
    association_counts_next_generation = numpy.zeros(len(reference_points))
    association_counts_structured_points = numpy.zeros(len(reference_points))

    delayed_reference_points = dask.delayed(reference_points)

    delayed_results = [
        dask.delayed(get_closest_reference_point)(normalized_evaluations[i], delayed_reference_points) for i in
        range(len(structured_points))]

    results = dask.compute(*delayed_results)
    results = numpy.asarray(results)

    reference_points_assignment = results[:, 0].astype('int')
    perpendicular_distances_of_reference_points = results[:, 1]

    unique_counts, counts_elements = numpy.unique(reference_points_assignment, return_counts=True)
    association_counts_structured_points[unique_counts] = counts_elements

    unique_counts, counts_elements = numpy.unique(
        reference_points_assignment[numpy.where(structured_points == next_generation)], return_counts=True)
    association_counts_next_generation[unique_counts] = counts_elements

    return reference_points_assignment, association_counts_structured_points, association_counts_next_generation, perpendicular_distances_of_reference_points


def associate_to_niche(structured_points, association_counts_next_generation, reference_points_assignment, perpendicular_distances_of_reference_points, last_front_points, last_front_index,selected_number):
    selected_last_front_points = set()
    # Getting the lowest niche count reference points
    while len(selected_last_front_points) < selected_number:
        min_reference_point = numpy.argsort(association_counts_next_generation)[0]

        if association_counts_next_generation[min_reference_point] == 0:

            points_associated_minimum_niche = [i for i in range(last_front_index, len(structured_points)) if reference_points_assignment[i] == min_reference_point]

            if len(points_associated_minimum_niche) == 0:
                association_counts_next_generation[min_reference_point] = numpy.Inf
            else:
                perpendicular_distances = [(point, perpendicular_distances_of_reference_points[point]) for point in
                                           points_associated_minimum_niche]
                min_perpendicular_distance_point = sorted(perpendicular_distances, key=itemgetter(1))[0]

                selected_last_front_points.add(structured_points[min_perpendicular_distance_point[0]])
                association_counts_next_generation[min_reference_point] += 1
        else:
            random_last_front_member = random.randrange(last_front_points.size)

            if reference_points_assignment[
                last_front_index + random_last_front_member] == min_reference_point:
                selected_last_front_points.add(structured_points[last_front_index + random_last_front_member])

            association_counts_next_generation[min_reference_point] += 1

    return list(selected_last_front_points)

def polynomial_mutation(ind, p_mutation, distribution_index):
    individual = copy.deepcopy(ind)
    return numpy.asarray(mutation.mutPolynomialBounded(individual,distribution_index,0.0,1.0,p_mutation))

# Set mutation for feature's subsets
def set_mutation(individual, p_mutation, n):

    Mutation = set()
    for i in individual:
        if random.uniform(0.0, 1.0) <= p_mutation:
            Mutation.add(i)

    Mutations = set()
    for m in Mutation:
        if random.uniform(0.0, 1.0) <= 0.5:
            Mutations.add(m)

    Mutationr = Mutation - Mutations
    Ns = set()
    while len(Ns) < len(Mutations):
        rand_item = random.randint(0, n - 1)
        if not rand_item in individual:
            Ns.add(rand_item)
    try:
        Na = {random.choice(list((individual-Mutation) - Ns))}
    except IndexError:
        Na = set()
    mutated_individual = (individual - Mutation) | Ns | Na

    if len(mutated_individual) == 0:
        return {(list(individual)[0]+1) % n}

    return mutated_individual

def combine_population_real_representation(population_representation, n, fronts, crowding_assignment, p_mutation, object_evaluations,
                                           binary_tournament, crossover_operator, mutation_operator, c_distribution_index, m_distribution_index):

    total_population = len(population_representation)
    offspring_population_representation = numpy.zeros((total_population, n))

    current_population_count = 0
    # Among two random parents, selecting first efficient one and then the next efficient
    while current_population_count < total_population:
        random_individuals = random.sample(range(0, total_population), 2)
        first_indicator = binary_tournament(random_individuals[0], random_individuals[1], fronts, crowding_assignment)
        random_individuals = random.sample(range(0, total_population), 2)
        second_indicator = binary_tournament(random_individuals[0], random_individuals[1], fronts, crowding_assignment)

        offspring1, offspring2 = crossover_operator(population_representation[first_indicator], population_representation[second_indicator], c_distribution_index)

        offspring_population_representation[current_population_count] = mutation_operator(offspring1, p_mutation, m_distribution_index)
        offspring_population_representation[current_population_count + 1] = mutation_operator(offspring2, p_mutation, m_distribution_index)
        current_population_count += 2

    offspring_evaluations = object_evaluations(offspring_population_representation)

    return offspring_population_representation, offspring_evaluations


def assigning_crowding_distance(evaluations_values):
    total_population = evaluations_values.shape[0]
    objects = evaluations_values.shape[1]
    crowding_assignment = numpy.zeros(total_population)
    # Sorting for each given value of object making sure that for every other point the boundary point is selected
    for m in numpy.arange(objects):
        sorted_evaluations = numpy.argsort(evaluations_values[:, m])

        crowding_assignment[
            sorted_evaluations[0]] = numpy.inf
        crowding_assignment[sorted_evaluations[-1]] = numpy.inf

        f_min = evaluations_values[sorted_evaluations[0]][m]
        f_max = evaluations_values[sorted_evaluations[-1]][m]

        for i in numpy.arange(1, total_population - 1):
            crowding_assignment[sorted_evaluations[i]] = crowding_assignment[sorted_evaluations[i]] + (
                        evaluations_values[sorted_evaluations[i + 1]][m] - evaluations_values[sorted_evaluations[i - 1]][m]) / (
                                                                     f_max - f_min)

    return crowding_assignment

def initialize_operator_sbx(first_indicator, second_indicator, distribution_index):

    start_indicator = copy.deepcopy(first_indicator)
    last_indicator = copy.deepcopy(second_indicator)

    return crossover.cxSimulatedBinaryBounded(start_indicator, last_indicator, distribution_index, 0.0, 1.0)

class NSGAII:

    def get_structured_points(self, points_per_front, total_population, evaluations_values):
        crowding_per_front = []
        current_front_counter = 0
        current_structured_points = []
        crowding_assignment = numpy.empty(0)

        while len(current_structured_points) < total_population:

            for point in points_per_front[current_front_counter]:
                current_structured_points.append(point)

            crowding_per_front.append(assigning_crowding_distance(evaluations_values[points_per_front[current_front_counter]]))
            crowding_assignment = numpy.append(crowding_assignment, crowding_per_front[current_front_counter])

            current_front_counter += 1

        last_front = current_front_counter - 1
        last_front_points = points_per_front[last_front]

        return numpy.asarray(current_structured_points, dtype=int), last_front, numpy.asarray(last_front_points, dtype=int), crowding_assignment, crowding_per_front

    def selection_of_population(self, points_per_front, total_population, merged_population_representation, combined_evaluations):

        current_structured_points, last_front, last_front_points, crowding_assignment, crowding_per_front = self.get_structured_points(points_per_front, total_population, combined_evaluations)
        
        # if both populations are not equal , then it will be ordered with crowding distance of last_fronts's solutions.
        if len(current_structured_points) == total_population:
            population_representation = merged_population_representation[current_structured_points]
            evaluations_values = combined_evaluations[current_structured_points]

            return population_representation, evaluations_values, crowding_assignment

        else:
            num_elements = len(current_structured_points) - len(last_front_points)
            next_generation = current_structured_points[:num_elements]
            selected_number = total_population - len(next_generation)

            ordered_crowded = numpy.argsort(crowding_per_front[last_front]*-1)
            selected_points = last_front_points[ordered_crowded[:selected_number]]
            selected_crowds = crowding_per_front[last_front][ordered_crowded[:selected_number]]

            population_representation = numpy.concatenate((merged_population_representation[next_generation],
                                                        merged_population_representation[selected_points]), axis=0)
            evaluations_values = numpy.concatenate((combined_evaluations[next_generation], combined_evaluations[selected_points]),
                                         axis=0)

            crowding_assignment = numpy.concatenate((crowding_assignment[:len(next_generation)], selected_crowds), axis=0)

            return population_representation, evaluations_values, crowding_assignment

class NSGAIII:

    def get_structured_points(self, points_per_front, total_population):
        current_structured_points = []
        current_front_counter = 0
        while len(current_structured_points) < total_population:

            for point in points_per_front[current_front_counter]:
                current_structured_points.append(point)
            current_front_counter += 1

        last_front = current_front_counter - 1
        last_front_points = points_per_front[last_front]

        return numpy.asarray(current_structured_points, dtype=int), numpy.asarray(last_front_points, dtype=int)

    def selection_of_population(self, points_per_front, total_population, merged_population_representation, combined_evaluations, target_functions, current_min_value, current_max_value, reference_points):

        current_structured_points, last_front_points = self.get_structured_points(points_per_front, total_population)
        objects = len(target_functions)

        if len(current_structured_points) == total_population:
            population_representation = merged_population_representation[current_structured_points]
            evaluations_values = combined_evaluations[current_structured_points]

            return population_representation, evaluations_values

        else:
            num_elements = len(current_structured_points) - len(last_front_points)
            next_generation = current_structured_points[:num_elements]
            last_front_index = len(next_generation)
            selected_number = total_population - len(next_generation)
            normalized_evaluations = normalize(combined_evaluations, current_structured_points, objects,
                                               current_min_value, current_max_value)

            reference_points_assignment, association_counts_structured_points, association_counts_next_generation, perpendicular_distances_of_reference_points = \
                associate_dask(current_structured_points, normalized_evaluations, reference_points, next_generation)

            selected_points = \
                associate_to_niche(current_structured_points, association_counts_next_generation,
                        reference_points_assignment, perpendicular_distances_of_reference_points, last_front_points,
                        last_front_index, selected_number)

            population_representation = numpy.concatenate((merged_population_representation[next_generation],
                                                        merged_population_representation[selected_points]), axis=0)
            evaluations_values = numpy.concatenate((combined_evaluations[next_generation], combined_evaluations[selected_points]),
                                         axis=0)

            return population_representation, evaluations_values


# Test Suite for comparing NSGA2 and NSGA3
class DTLZ1:
    def __init__(self, objects):
        self.objects = objects
        self.target_functions = numpy.zeros(objects, dtype=numpy.int)
        self.k = 5
        self.n = objects + self.k - 1
        
    def generate_start_population(self, total_population):
        return numpy.random.rand(total_population, self.n)

    def object_evaluations(self, representation):
        separate_numbers = len(representation)
        evaluations_values = numpy.zeros((separate_numbers, self.objects))

        for i in range(separate_numbers):
            evaluations_values[i] = list(dtlz1(representation[i], self.objects))

        return evaluations_values

    def get_suite_name(self):
        return 'DTLZ1'

# displaying the hyper volume values of algorithm
def display(evaluations, algorithm, test_suite, no_iteration):
    print(algorithm)
    print("   ", test_suite)
    print("       ", "HyperVolume", ":")
    hypervolume_lists_after_normalization = []
    count = 0
    for evaluation in evaluations:
        if count < no_iteration:
            hypervolume_lists_after_normalization.append(evaluation[0])
            count += 1
        else:
            break
    print("       ", hypervolume_lists_after_normalization)


def main(event, context):

    current_problem = DTLZ1(3)
    total_population = 120
    poly_mutation = 1.0 / current_problem.n
    max_iterations = 50
    c_distribution_index = 20.0
    m_distribution_index = 20.0

    nsga = NSGAStructure(
        poly_mutation,
        current_problem.objects,
        current_problem.n,
        max_iterations,
        current_problem.object_evaluations,
        current_problem.generate_start_population,
        polynomial_mutation,
        current_problem.target_functions,
        initialize_operator_sbx,
        m_distribution_index,
        c_distribution_index,
        combine_population_real_representation,
        total_population
    )
    nsga.call_algo_nsgaii()
    pareto_front_points = nsga.points_per_front[0]
    pareto_front_evaluations = nsga.evaluations[pareto_front_points]
    pareto_front = nsga.fronts[pareto_front_points]
    evaluations = nsga.evaluations
    display(evaluations, "NSGAII", "DTLZ1", max_iterations)
    print(evaluations)
    plot_of_calculated_evaluations(evaluations, current_problem)
    plot_of_calculated_evaluations(pareto_front_evaluations, current_problem)
    plot_3d_of_calculated_evaluations(evaluations)

    nsga.call_algo_nsgaiii()
    pareto_front_points = nsga.points_per_front[0]
    pareto_front_evaluations = nsga.evaluations[pareto_front_points]
    evaluations = nsga.evaluations
    display(evaluations, "NSGAIII", "DTLZ1", max_iterations)

    print(evaluations)
    plot_of_calculated_evaluations(evaluations, current_problem)
    plot_of_calculated_evaluations(pareto_front_evaluations, current_problem)
    plot_3d_of_calculated_evaluations(evaluations)


if __name__ == '__main__':
    main('', '')