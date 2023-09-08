import random
import time
import copy

import numpy as np
from deap import tools
import dso.gp.utils as U

from dso.program import from_tokens, _finish_tokens, Program

def _eval_step(tokens):
    """
    Take in an action which is a numpy array/string and return the reward
    """
    p = from_tokens(tokens, on_policy=False, finish_tokens=False, skip_cache=True)
    r = p.r # This calls the reward computation 
    return p

class RunOneStepAlgorithm:
    """
    Top level class which runs the GP one generation at a time, replacing
    classes like eaSimple since we need more control over how it runs.
    """

    def __init__(self, toolbox, p_crossover, verbose=__debug__):

        self.toolbox = toolbox
        self.verbose = verbose

        self.cxpb = p_crossover  # TODO: use toolbox.mate.keywords['indpb']

        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'iter', 'pop_size', 'nevals',
                               'uncached_size', 'best_val', 'timer']

        self.population = None # Must be explicitly set
        self.gen = 0

    def _eval(self, population):
        # Evaluate the individuals with an invalid fitness
        # This way we do not evaluate individuals that we have already seen.
        tokens_list     = []
        uncached_ind    = [] 
        invalid_ind     = [ ind for ind in population if not ind.fitness.valid ]
        for ind in invalid_ind:
            # Get from deap to numpy array
            # Make sure token is equiv with program finished token used for cache key
            tokens = _finish_tokens(ind.tokenized_repr)

            try:
                # we've seen this one before, copy back the reward.
                # We use a try loop to avoid double look-ups in the dict
                p = Program.cache[tokens.tostring()]
                ind.fitness.values = (-p.r,)
            except (KeyError, TypeError):
                # We have not seen this one, we need to compute reward
                tokens_list.append(tokens)
                # Keep track of uncached inds for now
                uncached_ind.append(ind)
                # Deap demands this is a value and not None. positive inf should not 
                # be viable since rewards are always negative (are penalties) in Deap.
                ind.fitness.values = (np.inf,)

        # Calls either map or pool.map
        programs    = list(self.toolbox.cmap(_eval_step, tokens_list))
        for i, ind in enumerate(population):
            if np.isposinf(ind.fitness.values[0]):
                p = programs.pop(0)
                ind.fitness.values = (-p.r,)
                Program.cache[p.str] = p 

        return population, invalid_ind, uncached_ind

    # Would this benefit from using process pooling?
    def _var_and(self, population):

        offspring = [self.toolbox.clone(ind) for ind in population]

        # Apply crossover on the offspring
        for i in range(1, len(offspring), 2):
            if random.random() < self.cxpb:
                offspring[i - 1], offspring[i] = self.toolbox.mate(offspring[i - 1], offspring[i])

                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        # Apply mutation on the offspring
        for i in range(len(offspring)):
            offspring[i], = self.toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

        return offspring

    def __call__(self, hof=None, iter=None):

        t1 = time.perf_counter()

        # Select the next generation individuals
        offspring = self.toolbox.select(self.population, len(self.population))
  
        # Vary the pool of individuals
        offspring = self._var_and(offspring)

        # Evaluate the individuals with an invalid fitness
        population, invalid_ind, uncached_ind = self._eval(offspring)
        
        pop_size        = len(population)       # Total population size 
        nevals          = len(invalid_ind)      # New individuals which are Deap not valid (Not yet evaluated)
        uncached_size   = len(uncached_ind)     # Individuals which are not in cache 

        # Replace the current population by the offspring
        self.population[:] = offspring

        # Update hall of fame
        if hof is not None:
            hof.update(self.population)
            best_val = best_val=hof[0].fitness.values
        else:
            best_val = None

        timer = time.perf_counter() - t1

        self.logbook.record(gen=self.gen, iter=iter, pop_size=pop_size,
                            nevals=nevals, uncached_size=uncached_size,
                            best_val=best_val, timer=timer)

        if self.verbose:
            print(self.logbook.stream)

        self.gen += 1

        return nevals

    def set_population(self, population):

        self.population = population
