import random
import sys
import time
import copy
import numpy as np

""" TSP """


# Problem representation and holds genetic algorithm parameters
class Tsp:
    def __init__(self, populationSize, distanceMatrix, k, crossOverRate, mutationRate,
                 localSearchRateInit, localSearchOperator, localSearchRate,
                 initializationFunction, selectionFunction, crossOverFunction, mutationFunction, eliminationFunction):
        self.distanceMatrix = distanceMatrix
        self.populationSize = populationSize
        self.numberOfCities = np.size(self.distanceMatrix, 0)
        self.k = k
        self.crossOverRate = crossOverRate
        self.mutationRate = mutationRate

        self.localSearchRateInit = localSearchRateInit
        self.localSearchOperator = localSearchOperator
        self.localSearchRate = localSearchRate

        self.initializationFunction = initializationFunction
        self.selectionFunction = selectionFunction
        self.crossOverFunction = crossOverFunction
        self.mutationFunction = mutationFunction
        self.eliminationFunction = eliminationFunction


# Candidate solution representation
class Individual:
    def __init__(self, tsp, listOfCities=None, fitness=None):
        self.tsp = tsp
        if listOfCities is None:
            self.listOfCities = np.random.permutation(self.tsp.numberOfCities)
        else:
            self.listOfCities = listOfCities
        self.fitness = fitness
        if fitness is None:
            self.fitness = 0
        else:
            self.fitness = fitness
        self.mutationRate = tsp.mutationRate + np.random.normal(0, 0.01)
        self.crossOverRate = tsp.crossOverRate + np.random.normal(0, 0.04)

    def printIndividual(self):
        print("list of cities: " + str(self.listOfCities) + "| fitness: " + str(self.fitness))


def calculateFitness(tsp, listOfCities):
    fitness = 0
    for i in range(-1, len(listOfCities) - 1):
        fitness += tsp.distanceMatrix.item(listOfCities[i], listOfCities[i + 1])
    return fitness


# Print the given population
def printPopulation(tsp, pop):
    for i in range(tsp.populationSize):
        pop[i].printIndividual()


# Calculate the total fitness of a given population
def calculateTotalFitness(pop):
    totalFitness = 0
    for i in range(0, len(pop)): totalFitness += pop[i].fitness
    return totalFitness


# Calculate the best individual from a population according to its fitness value
def getBestIndividual(tsp, pop):
    bestIndividual = pop[0]
    minFitness = bestIndividual.fitness
    for i in range(0, tsp.populationSize):
        if pop[i].fitness < minFitness:
            minFitness = pop[i].fitness
            bestIndividual = pop[i]
    return bestIndividual


# Calculate the best individual's index from a population according to its fitness value
def getBestIndividualIndex(tsp, pop):
    bestIndividualIndex = 0
    minFitness = pop[0].fitness
    for i in range(0, tsp.populationSize):
        if pop[i].fitness < minFitness:
            minFitness = pop[i].fitness
            bestIndividualIndex = i
    return bestIndividualIndex


# Calculate the best individual from a population according to its fitness value
def getWorstIndividualIndex(pop):
    minFitness = pop[0].fitness
    worstIndividualIndex = 0
    for i in range(0, len(pop)):
        if pop[i].fitness > minFitness:
            minFitness = pop[i].fitness
            worstIndividualIndex = i
    return worstIndividualIndex


# Check whether there a duplicate cities in one of the individuals in a population (for testing purposes)
def duplicatesInPopulation(population):
    for i in range(len(population)):
        if len(population[i].listOfCities) != len(set(population[i].listOfCities)):
            return True
    return False


"""-------------------"""
""" Genetic Algorithm """
"""-------------------"""

""" Initialization """


# Initialize a population randomly
def initPopulationRandomly(tsp):
    population = np.empty((tsp.populationSize,), dtype=type(Individual(tsp)))
    for i in range(tsp.populationSize):
        population[i] = Individual(tsp)
        population[i].fitness = calculateFitness(tsp, population[i].listOfCities)
    return population


# Initialize a population using KNN clustering
def initPopulationKNNclusters(tsp):
    # setup
    population = np.empty((tsp.populationSize,), dtype=type(Individual(tsp)))
    numberOfClusters = round(tsp.numberOfCities ** (1 / 1.5))
    clusterSizes = [tsp.numberOfCities // numberOfClusters + (1 if x < tsp.numberOfCities % numberOfClusters else 0) for
                    x in range(numberOfClusters)]

    # perform KNN clustering on tsp.localSearchRateInit-percent of the starting population.
    for j in range(tsp.populationSize):
        if random.random() < tsp.localSearchRateInit:
            availableCities = list(range(0, tsp.numberOfCities))
            citiesTaken = np.full((tsp.numberOfCities,), -1, dtype='int16')
            elIndex = 0

            # add first cluster center
            clusterCenter = random.choice(availableCities)
            availableCities.remove(clusterCenter)
            citiesTaken[elIndex] = clusterCenter
            elIndex += 1

            for i in range(numberOfClusters - 1):
                # find nearest neighbours
                listOfKNeighbours = findKNN(tsp, clusterCenter, clusterSizes[i] - 1, availableCities)

                # remove the KNN cities from the available cities and add them to taken cities
                availableCities = [ele for ele in availableCities if ele not in listOfKNeighbours]
                for el in range(clusterSizes[i] - 1): citiesTaken[elIndex + el] = listOfKNeighbours[el]
                elIndex += (el + 1)

                # get closest next cluster center
                clusterCenter = findKNN(tsp, clusterCenter, 1, availableCities)[0]
                availableCities.remove(clusterCenter)
                citiesTaken[elIndex] = clusterCenter
                elIndex += 1

            # find its K nearest neighbours
            listOfKNeighbours = findKNN(tsp, clusterCenter, clusterSizes[numberOfClusters - 1] - 1, availableCities)
            # remove the KNN cities from the available cities and add them to taken cities
            for el in range(clusterSizes[numberOfClusters - 1] - 1): citiesTaken[elIndex + el] = listOfKNeighbours[el]
            population[j] = Individual(tsp, citiesTaken, calculateFitness(tsp, citiesTaken))
        else:
            population[j] = Individual(tsp)
            population[j].fitness = calculateFitness(tsp, population[j].listOfCities)
    return population


# Find the k nearest neighbours of the center point in the available cities.
def findKNN(tsp, center, k, availableCities):
    kNearestNeighbours = []

    # make tuple array with city distance to center and number and sort it on distance to center
    distancesFromCenterArray = tsp.distanceMatrix[center]
    tuplesArray = []
    for i in range(tsp.numberOfCities):
        tuplesArray.append((distancesFromCenterArray[i], i))
    tuplesArray.sort(key=lambda tup: tup[0])

    # insert closest city to center, which is not already taken in the new K nearest neighbours array
    i = 0
    insertions = 0
    while insertions < k:
        potentialCityInCluster = tuplesArray[i + 1][1]
        if potentialCityInCluster in availableCities:
            kNearestNeighbours.append(potentialCityInCluster)
            insertions += 1
        i += 1
    return kNearestNeighbours


""" Selection """


# Perform the given selection scheme of the island on the population
def selectParents(tsp, population):
    selectedParents = np.empty((tsp.populationSize,), dtype=type(Individual(tsp)))
    for j in range(tsp.populationSize):
        selectedParents[j] = tsp.selectionFunction(tsp, population)
    return selectedParents


# Perform a k-tournament selection on a population given a value for k
def kTournamentSelection(tsp, population):
    randomIndices = np.random.randint(0, tsp.populationSize, tsp.k)
    minFitness = population[0].fitness
    bestIndividual = population[0]
    for i in range(tsp.k):
        if population[randomIndices[i]].fitness < minFitness:
            bestIndividual = population[randomIndices[i]]
            minFitness = population[randomIndices[i]].fitness
    return bestIndividual


""" Crossover """


# Perform order crossover on the given mating pool of parents
def crossoverPop(tsp, parents):
    offspring = np.empty((tsp.populationSize,), dtype=type(Individual(tsp)))
    for j in range(round(tsp.populationSize * 0.50)):
        if random.uniform(0, 1) < (parents[2 * j].crossOverRate + parents[
            2 * j + 1].crossOverRate) * 0.5:  # and np.not_equal(parents[2*j,0].listOfCities, parents[2*j + 1,0].listOfCities).all():
            offspring[2 * j] = tsp.crossOverFunction(tsp, parents[2 * j],
                                                     parents[2 * j + 1])  # Perform crossover on parent 1 and parent 2
            offspring[2 * j + 1] = tsp.crossOverFunction(tsp, parents[2 * j + 1],
                                                         parents[2 * j])  # Perform crossover on parent 2 and parent 1
        else:
            offspring[2 * j] = inversionMutation(tsp, copy.deepcopy(parents[2 * j]))
            offspring[2 * j + 1] = inversionMutation(tsp, copy.deepcopy(parents[2 * j + 1]))
    return offspring


# Perform an order crossover given two parents
def orderCrossover(tsp, parent1, parent2):
    randomIndices = random.sample(range(0, tsp.numberOfCities), 2)  # Take 2 random indices to make slices
    randomIndices.sort()
    individual = Individual(tsp)
    individual.listOfCities = copy.deepcopy(parent1.listOfCities)  # parent1  # Keep the middle slice of parent 1

    allRemaining = np.setdiff1d(parent2.listOfCities, parent1.listOfCities[randomIndices[0]:randomIndices[1]],
                                assume_unique=True)
    allRemaining = allRemaining.astype(int)

    splitIndex = tsp.numberOfCities - randomIndices[1]
    thirdSlice = allRemaining[:splitIndex]
    firstSlice = allRemaining[splitIndex:]

    individual.listOfCities[:randomIndices[0]] = firstSlice
    individual.listOfCities[randomIndices[1]:] = thirdSlice

    individual.fitness = calculateFitness(tsp, individual.listOfCities)  # Calculate the fitness for the new individual
    return individual


""" Mutation """


# perform the local search method given to the tsp object on the population.
def mutatePop(tsp, offspring):
    for j in range(len(offspring)):
        individual = offspring[j]
        if random.uniform(0, 1) < individual.mutationRate:
            offspring[j] = tsp.mutationFunction(tsp, individual)
    return offspring


# Perform an inversion mutation on an individual
def inversionMutation(tsp, individual):
    randomIndices = random.sample(range(0, tsp.numberOfCities), 2)  # Take 2 random indices to make slices
    randomIndices.sort()
    # Flip the cities between the 2 indices
    individual.listOfCities[randomIndices[0]: randomIndices[1]] = np.flip(
        individual.listOfCities[randomIndices[0]: randomIndices[1]])
    # Calculate the fitness for the new individual
    individual.fitness = calculateFitness(tsp, individual.listOfCities)
    return individual


# Perform a random number of swaps of cities with their neighbour
def swapWithNeighbourMutation(tsp, individual):
    for i in range(tsp.numberOfCities - 1):
        if random.uniform(0, 1) < individual.mutationRate:
            individual.listOfCities[i], individual.listOfCities[i + 1] = individual.listOfCities[i + 1], \
                                                                         individual.listOfCities[i]
    individual.fitness = calculateFitness(tsp, individual.listOfCities)
    return individual


# Perform a scramble on the cities of an individual between two indices
def scrambleMutation(tsp, individual):
    randomIndices = random.sample(range(0, len(individual.listOfCities)), 2)  # Take 2 random indices to make slices
    randomIndices.sort()
    # Scramble the cities between the 2 indices
    np.random.shuffle(individual.listOfCities[randomIndices[0]: randomIndices[1]])
    individual.fitness = calculateFitness(tsp, individual.listOfCities)  # Calculate the fitness for the new individual
    return individual


""" Local Search Operators """


# perform the local search method given to the tsp object on the population.
def localSearchPop(tsp, offspring):
    for j in range(len(offspring)):
        individual = offspring[j]
        if random.uniform(0, 1) < tsp.localSearchRate:
            offspring[j] = tsp.localSearchOperator(tsp, individual)
    return offspring


# Perform a random number of swaps of cities with their neighbour
def oneOptLocalSearch(tsp, individual):
    randomIndex = random.randint(0, tsp.numberOfCities)  # Take a random index
    bestListOfCities = copy.deepcopy(individual.listOfCities)
    minFitness = individual.fitness
    listOfCities = np.array(individual.listOfCities)
    for i in range(randomIndex, tsp.numberOfCities - 1):
        listOfCities[i], listOfCities[i + 1] = listOfCities[i + 1], listOfCities[i]
        fitness = calculateFitness(tsp, individual.listOfCities)
        if fitness < minFitness:
            minFitness = fitness
            bestListOfCities = listOfCities
            break
    listOfCities[tsp.numberOfCities - 1], listOfCities[0] = listOfCities[0], listOfCities[tsp.numberOfCities - 1]
    for i in range(0, randomIndex - 1):
        listOfCities[i], listOfCities[i + 1] = listOfCities[i + 1], listOfCities[i]
        fitness = calculateFitness(tsp, individual.listOfCities)
        if fitness < minFitness:
            minFitness = fitness
            bestListOfCities = listOfCities
            break
    individual.listOfCities = bestListOfCities
    individual.fitness = minFitness  # Calculate the fitness for the new individual
    return individual


# Perform K-opt heuristic mutation with for the time being k = 1
def kOptHeuristicLocalSearch(tsp, individual):
    # 1. take random city and its predecessor/successor
    randomCityIndex = random.randrange(tsp.numberOfCities)
    randomCity = individual.listOfCities[randomCityIndex]

    # 2. find 1-nn of the randomly chosen city
    nearestNeighbour = findKNN(tsp, randomCity, 1, individual.listOfCities)[0]

    # 3. if chosen random city is to the left of knn:
    #          sequence to remove => right neighbour of KNN -> left neighbour of random city
    #    else: sequence to remove => right neighbour of random city -> right neighbour of KNN
    indexOfNearestNeighbour = np.where(individual.listOfCities == nearestNeighbour)[0][0]
    # indexOfNearestNeighbour = individual.listOfCities.index(nearestNeighbour)
    if randomCityIndex < indexOfNearestNeighbour:
        sequence = individual.listOfCities[randomCityIndex + 1: indexOfNearestNeighbour]
    else:
        sequence = individual.listOfCities[indexOfNearestNeighbour + 1: randomCityIndex]

    # 4. remove selected sequence
    for i in range(len(sequence)):
        indexToRemove = np.where(individual.listOfCities == sequence[i])
        individual.listOfCities = np.delete(individual.listOfCities, indexToRemove)

    # 5. insert sequence (right/left neighbour of point, right/left neighbour of knn)
    #    in list of cities where the resulting fitness of the whole individual is the lowest.
    minFitness = sys.maxsize
    copyOfIndividual = Individual(tsp)
    bestInsert = 0
    for i in range(tsp.numberOfCities - len(sequence)):  # is bottleneck
        newListOfCities = np.hstack((individual.listOfCities[0:i], sequence, individual.listOfCities[i:]))
        copyOfIndividual.listOfCities = newListOfCities
        copyOfIndividual.fitness = calculateFitness(tsp, copyOfIndividual.listOfCities)
        if copyOfIndividual.fitness < minFitness:
            minFitness = copyOfIndividual.fitness
            bestInsert = i
    individual.listOfCities = np.hstack(
        (individual.listOfCities[0:bestInsert], sequence, individual.listOfCities[bestInsert:]))
    individual.fitness = minFitness
    return individual


# Perform knn on a random city of the given individual and swap them with cities next to the chosen city
def knnLocalSearch(tsp, individual):
    randomCity = np.random.randint(0, tsp.numberOfCities, 2)  # Take a random city
    oldFitness = individual.fitness
    individualCopy = copy.deepcopy(individual)
    for k in range(0, len(randomCity)):
        listOfKNeighbours = findKNN(tsp, randomCity[k], 3, list(range(0, tsp.numberOfCities)))
        indexCenter = np.where(individualCopy.listOfCities == randomCity[k])[0][0]
        i = 0
        j = 0
        while i + j < len(listOfKNeighbours):
            indexNeighbour = np.where(individualCopy.listOfCities == listOfKNeighbours[i + j])[0][0]
            if indexCenter + i + 1 < tsp.numberOfCities:
                tmp = individualCopy.listOfCities[indexCenter + i + 1]
                individualCopy.listOfCities[indexCenter + i + 1] = individualCopy.listOfCities[indexNeighbour]
                individualCopy.listOfCities[indexNeighbour] = tmp
                i += 1
            else:
                tmp = individualCopy.listOfCities[indexCenter - i - 1]
                individualCopy.listOfCities[indexCenter - i - 1] = individualCopy.listOfCities[indexNeighbour]
                individualCopy.listOfCities[indexNeighbour] = tmp
                j += 1
    newFitness = calculateFitness(tsp, individualCopy.listOfCities)
    if oldFitness > newFitness:
        return individualCopy
    return individual


""" Elimination """


# Perform the given selection scheme of the island on the population
def eliminateIndividuals(tsp, population):
    selectedParents = np.empty((tsp.populationSize,), dtype=type(Individual(tsp)))
    for j in range(tsp.populationSize):
        selectedParents[j] = tsp.eliminationFunction(tsp, population)
    return selectedParents


# Perform k-tournament elimination
def kTournamentElimination(tsp, population):
    randomIndices = np.random.randint(0, tsp.populationSize, tsp.k)
    minFitness = population[0].fitness
    bestIndividual = population[0]
    for i in range(tsp.k):
        if population[randomIndices[i]].fitness < minFitness:
            bestIndividual = population[randomIndices[i]]
            minFitness = population[randomIndices[i]].fitness
    return bestIndividual


""" Island Model """


# Migrate individuals from one population to another
def migrate(tsp1, tsp2, population1, population2):
    numberOfMigrants = 7
    randomIndices = np.random.randint(0, tsp1.populationSize - numberOfMigrants, numberOfMigrants)
    for i in randomIndices:
        population2.append(population1[i])
        population1.remove(population1[i])
    return


# Perform island recombination
def islandRecombination(tsp, populations, pop1Index, pop2Index):
    parent1 = tsp.selectionFunction(tsps[pop1Index], populations[pop1Index])  # Selection for parent 1
    parent2 = tsp.selectionFunction(tsps[pop2Index], populations[pop2Index])  # Selection for parent 2
    bestIndivIndex1 = getBestIndividualIndex(tsp, populations[pop1Index])
    bestIndivIndex2 = getBestIndividualIndex(tsp, populations[pop2Index])
    r = np.hstack((np.arange(1, bestIndivIndex1), np.arange(bestIndivIndex1 + 1, tsp.populationSize)))
    randomReplacement1 = random.choice(r)
    r = np.hstack((np.arange(1, bestIndivIndex2), np.arange(bestIndivIndex2 + 1, tsp.populationSize)))
    randomReplacement2 = random.choice(r)
    populations[pop1Index][randomReplacement1] = tsp.crossOverFunction(tsps[pop1Index], parent1, parent2)  # Crossover
    populations[pop2Index][randomReplacement2] = tsp.crossOverFunction(tsps[pop2Index], parent1, parent2)  # Crossover
    return populations[pop1Index], populations[pop2Index]


""" Genetic algorithm """


# Print status of population (for testing purposes)
def printStatus(tsp, it, pop, i):
    print("---Island " + str(i+1) +": iteration: " + str(it) + "-----------------------------------------------")
    # mean = np.mean([pop[i].fitness for i in range(tsp.populationSize)])
    # print("Mean Fitness: " + str(mean))
    # print("Diversity: " + str(mean - getBestIndividual(tsp, pop).fitness))
    print("best individual: ")
    print(getBestIndividual(tsp, pop).fitness)
    print("-----------------------------------------------")


# Initialisation step
def initialize(distanceMatrix):
    print("initializing populations...")
    global tsps
    start = time.time()

    # Chose the specifics for each island
    tsps = [Tsp(populationSize=100, distanceMatrix=distanceMatrix,
                k=7, crossOverRate=0.95, mutationRate=0.05, localSearchRateInit=0.95,
                localSearchOperator=knnLocalSearch, localSearchRate=0.05,
                initializationFunction=initPopulationKNNclusters,
                selectionFunction=kTournamentSelection,
                crossOverFunction=orderCrossover,
                mutationFunction=inversionMutation,
                eliminationFunction=kTournamentElimination),

            Tsp(populationSize=100, distanceMatrix=distanceMatrix,
                k=4, crossOverRate=0.98, mutationRate=0.05, localSearchRateInit=0.90,
                localSearchOperator=knnLocalSearch, localSearchRate=0.05,
                initializationFunction=initPopulationKNNclusters,
                selectionFunction=kTournamentSelection,
                crossOverFunction=orderCrossover,
                mutationFunction=scrambleMutation,
                eliminationFunction=kTournamentElimination),

            Tsp(populationSize=100, distanceMatrix=distanceMatrix,
                k=3, crossOverRate=0.95, mutationRate=0.20, localSearchRateInit=0.90,
                localSearchOperator=knnLocalSearch, localSearchRate=0.05,
                initializationFunction=initPopulationKNNclusters,
                selectionFunction=kTournamentSelection,
                crossOverFunction=orderCrossover,
                mutationFunction=inversionMutation,
                eliminationFunction=kTournamentElimination)]

    # Initialize each island
    populations = np.empty((3,), dtype=type(Individual(tsps[0])))
    for i in range(len(tsps)):
        populations[i] = initPopulationKNNclusters(tsps[i])  # Initialisation
        # printStatus(tsps[i], 0, populations[i])
    end = time.time()
    print("Elapsed time: " + str(end - start))
    return populations


def oneIteration(tsp, population):
    # Elitism
    bestIndividual = kOptHeuristicLocalSearch(tsp, getBestIndividual(tsp, population))

    # Selection
    parents = selectParents(tsp, population)

    # Crossover
    offspring = crossoverPop(tsp, parents)

    # Mutation
    offspring = mutatePop(tsp, offspring)  # not here

    # Local Search
    offspring = localSearchPop(tsp, offspring)  # not here

    # Elimination
    population = eliminateIndividuals(tsp, offspring)

    # Replace worst individual from offspring with best individual from parents
    population[getWorstIndividualIndex(offspring)] = bestIndividual
    return population

def printOutputAlgorithm():
    bestIndividInAlg = 0
    bestFitInAlg = float('inf')
    for i in range(len(tsps)):
        printStatus(tsps[i], it, populations[i], i)
        bestIndividInPop = getBestIndividual(tsps[i], populations[i])
        if bestIndividInPop.fitness < bestFitInAlg:
            bestIndividInAlg = bestIndividInPop
            bestFitInAlg = bestIndividInPop.fitness

    print("-----------------------------------------------")
    print("-----------------------------------------------")
    print("Result: ")
    print("Route: " + str(bestIndividInAlg.listOfCities))
    print("Fitness: " + str(bestIndividInAlg.fitness))
    end = time.time()
    print("Elapsed time: " + str(end - start))
    print("-----------------------------------------------")
    print("-----------------------------------------------")


# Don't forget to remove this, is only for testing purposes
if __name__ == '__main__':
    args = sys.argv[1:]
    filename = args[0]
    # Read distance matrix from file.
    file = open(filename)
    distanceMatrix = np.loadtxt(filename, delimiter=",")
    file.close()

    populations = initialize(distanceMatrix)
    for i in range(len(tsps)): printStatus(tsps[i], 0, populations[i], i)

    start = time.time()
    it = 1
    notImprovedCounter = 0
    prevBestFitnesses = [] + [float('inf')] * len(tsps)
    # averageFitness = calculateTotalFitness(populations[0]) / tsp.populationSize
    while notImprovedCounter < 2*tsps[0].numberOfCities:  # getBestIndividual(population).fitness / averageFitness < 0.993:
        for i in range(len(tsps)):
            populations[i] = oneIteration(tsps[i], populations[i])

        if it % 50 == 0:
            end = time.time()
            print("Elapsed time: " + str(end - start))
            for i in range(len(tsps)): printStatus(tsps[i], it, populations[i], i)
            print('##############################')
            for _ in range(round(tsps[0].populationSize * 0.10)):
                randomIndices = np.random.permutation(len(tsps))
                for i in range(len(tsps) - 1):
                    populations[randomIndices[i]], populations[randomIndices[i + 1]] = \
                        islandRecombination(tsps[0], populations, randomIndices[i], randomIndices[i + 1])
                populations[randomIndices[len(tsps) - 1]], populations[randomIndices[0]] = \
                    islandRecombination(tsps[0], populations, randomIndices[len(tsps) - 1], randomIndices[0])
        # stop-condition
        for i in range(len(tsps)):
            newFit = calculateFitness(tsps[i], getBestIndividual(tsps[i], populations[i]).listOfCities)
            if newFit < prevBestFitnesses[i]:
                prevBestFitnesses[i] = newFit
            else:
                notImprovedCounter += 1
        # print(notImprovedCounter)
        it += 1

    # output of Algorithm
    printOutputAlgorithm()
