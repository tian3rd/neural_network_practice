# COMP4660/8420 Lab 8.2 - Solving an equation through Genetic Algorithm
# derived from https://morvanzhou.github.io/tutorials/machine-learning/evolutionary-algorithm/2-01-genetic-algorithm/

# import required libraries
import numpy as np
import matplotlib.pyplot as plt

# ## Step 2: Define settings
# 1. DNA size: the number of bits in DNA
# 2. Population size
# 3. Crossover rate
# 4. Mutation rate
# 5. Number of generations

# define GA settings
DNA_SIZE = 10             # number of bits in DNA
POP_SIZE = 100            # population size
CROSS_RATE = 0.8          # DNA crossover probability
MUTATION_RATE = 0.002     # mutation probability
N_GENERATIONS = 100       # generation size


# define data settings
X_BOUND = [0, 5]          # data upper and lower bounds 


# ## Step 3: Define fitness, select, crossover, mutate functions

# define target function
def F(x): 
    return np.sin(10*x)*x + np.cos(2*x)*x


# define non-zero fitness function for selection
def get_fitness(prediction):
    return prediction + 1e-3 - np.min(prediction)


# covert binary DNA to meaningful decimals, and normalise it to 0-5
def translateDNA(pop):
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]


# define population select function based on fitness value
# population with higher fitness value has higher chance to be selected
def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]


# define gene crossover function
def crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        # randomly select another individual from population
        i = np.random.randint(0, POP_SIZE, size=1)    
        # choose crossover points(bits)
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)
        # produce one child
        parent[cross_points] = pop[i, cross_points]  
    return parent


# define mutation function
def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


# ## Step 4: Start training GA
# 1. randomly initialise population
# 2. determine fitness of population
# 3. repeat
#     1. select parents from population
#     2. perform crossover on parents creting population
#     3. perform mutation of population


# initialise population DNA
pop = np.random.randint(0, 2, (1, DNA_SIZE)).repeat(POP_SIZE, axis=0) #note there is a mistake here that needs correcting


# here are some commands for plotting to show learning process;
# please comment this line if you would not like to see a plot
plt.ion()

for t in range(N_GENERATIONS):
    # convert binary DNA to decimals between 0-5
    pop_DNA = translateDNA(pop)
    # compute target function based on extracted DNA
    F_values = F(pop_DNA)
    
    # here are some commands for plotting
    # please comment this code if you would not like to see a plot
    if 'sca' in globals():
        sca.remove()
    # plot best population so far
    sca = plt.scatter(pop_DNA, F_values, s=200, lw=0, c='red', alpha=0.5)
    # plot target function: y = sin(10*x)*x + cos(2*x)*x
    x = np.linspace(*X_BOUND, 200)
    plt.plot(x, F(x))
    plt.show()
    plt.pause(0.05)
    
    # train GA
    # calculate fitness value 
    fitness = get_fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    
    # select better population as parent 1
    pop = select(pop, fitness)
    # make another copy as parent 2
    pop_copy = pop.copy()
    
    for parent in pop:
        # produce a child by crossover operation
        child = crossover(parent, pop_copy)
        # mutate child
        child = mutate(child)
        # replace parent with its child
        parent[:] = child  
        
# turn plotting off
plt.ioff()

