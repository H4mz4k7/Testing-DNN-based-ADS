import os
import time
import deap.algorithms
import numpy as np
import math
import logging as log
from random import randint, gauss, random
import matplotlib.pyplot as plt
import pandas as pd
from deap import base, creator, tools


from code_pipeline.tests_generation import RoadTestFactory


class GATestGenerator():
    """
        Generates a set of tests using GA (based on the Deap library; https://github.com/deap/deap).

    """

    

    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size
        self.valid_roads = {}
        self.time_elapsed_list = []
        self.test_count = 0
        self.redundant_count = 0
        self.best_solution = []
        self.every_road = []
        self.every_test_outcome = []
        self.every_description = []
        self.every_fitness = []
        self.every_best = []
        self.every_time = []

    def init_attribute(self):
        
        attribute = (randint(10, self.map_size-10), randint(10, self.map_size-10))
        return attribute
    

    def is_redundant(self, all_roads, road_points):
        all_road_points = all_roads
        is_redundant_flag = False
        if road_points in all_road_points:
            is_redundant_flag = True
            return is_redundant_flag
        
      
        for i in range(len(all_road_points)):
            road_point_one = road_points
            road_point_two = all_road_points[i]
            dx = road_point_one[0][0] - road_point_two[0][0]
            dy = road_point_one[0][1] - road_point_two[0][1]

        
            #change road 2 to start at same point as road 1
            road_point_two = [(point[0]+dx, point[1]+dy) for point in road_point_two]

            
            #mid point of each respective road
            middle_one = road_point_one[1]
            middle_two = road_point_two[1]

            #end point of each respective road
            last_one = road_point_one[2]
            last_two = road_point_two[2]

            if abs(middle_one[0] - middle_two[0]) <= 5 and abs(middle_one[1] - middle_two[1]) <= 5 and \
                abs(last_one[0] - last_two[0]) <= 5 and abs(last_one[1] - last_two[1]) <=5:
                    is_redundant_flag = True
                    return is_redundant_flag
            
        return is_redundant_flag


    def eaSimple(self,population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
        """(modified from deap.algorithms.eaSimple)
        This algorithm reproduce the simplest evolutionary algorithm as
        presented in chapter 7 of [Back2000]_.

        :param population: A list of individuals.
        :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                        operators.
        :param cxpb: The probability of mating two individuals.
        :param mutpb: The probability of mutating an individual.
        :param ngen: The number of generation.
        :param stats: A :class:`~deap.tools.Statistics` object that is updated
                    inplace, optional.
        :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                        contain the best individuals, optional.
        :param verbose: Whether or not to log the statistics.
        :returns: The final population
        :returns: A class:`~deap.tools.Logbook` with the statistics of the
                evolution

        The algorithm takes in a population and evolves it in place using the
        :meth:`varAnd` method. It returns the optimized population and a
        :class:`~deap.tools.Logbook` with the statistics of the evolution. The
        logbook will contain the generation number, the number of evaluations for
        each generation and the statistics if a :class:`~deap.tools.Statistics` is
        given as argument. The *cxpb* and *mutpb* arguments are passed to the
        :func:`varAnd` function. The pseudocode goes as follow ::

            evaluate(population)
            for g in range(ngen):
                population = select(population, len(population))
                offspring = varAnd(population, toolbox, cxpb, mutpb)
                evaluate(offspring)
                population = offspring

        As stated in the pseudocode above, the algorithm goes as follow. First, it
        evaluates the individuals with an invalid fitness. Second, it enters the
        generational loop where the selection procedure is applied to entirely
        replace the parental population. The 1:1 replacement ratio of this
        algorithm **requires** the selection procedure to be stochastic and to
        select multiple times the same individual, for example,
        :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
        Third, it applies the :func:`varAnd` function to produce the next
        generation population. Fourth, it evaluates the new individuals and
        compute the statistics on this population. Finally, when *ngen*
        generations are done, the algorithm returns a tuple with the final
        population and a :class:`~deap.tools.Logbook` of the evolution.

        .. note::

            Using a non-stochastic selection method will result in no selection as
            the operator selects *n* individuals from a pool of *n*.

        This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
        :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
        registered in the toolbox.

        .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
        Basic Algorithms and Operators", 2000.
        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = deap.algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook

    def mutate_tuple(self, individual, mu, sigma, indpb):
        """(modified from deap.tools.mutGaussian)
        This function applies a gaussian mutation of mean *mu* and standard
        deviation *sigma* on the input individual. This mutation expects a
        :term:`sequence` individual composed of real valued 2-dimensional tuples.
        The *indpb* argument is the probability of each attribute to be mutated.

        :param individual: Individual to be mutated
        :param mu: Mean or :term:`python:sequence` of means for the
                   gaussian addition mutation
        :param sigma: Standard deviation or :term:`python:sequence` of
                      standard deviations for the gaussian addition mutation
        :param indpb: Independent probability for each attribute to be mutated
        :returns: A tuple of one individual.
        """

        for i in range(0, len(individual)):
            if random() < indpb:
                # convert tuple into list to update values
                point = list(individual[i])

                # update the first value (x-pos)
                point[0] += int(gauss(mu, sigma))
                if point[0] < 10:
                    point[0] = 10
                if point[0] > self.map_size - 10:
                    point[0] = self.map_size - 10

                # update the second value (y-pos)
                point[1] += int(gauss(mu, sigma))
                if point[1] < 10:
                    point[1] = 10
                if point[1] > self.map_size - 10:
                    point[1] = self.map_size - 10

                # update the attribute (tuple) in the individual
                individual[i] = tuple(point)

        return individual,




    def evaluate(self, individual, start_time):
        
        
        # Creating the RoadTest from the points
        road_points = list(individual)


        if self.is_redundant(self.every_road, road_points):
            log.info("Test_outcome: REDUNDANT")
            self.test_count += 1
            current_time = time.time() - start_time
            self.redundant_count += 1
            self.every_road.append(road_points)
            self.every_test_outcome.append("REDUNDANT")
            self.every_description.append("Redundant test")
            self.every_fitness.append("Redundant test")
            self.every_time.append(current_time)
            if self.best_solution != []:
                previous_best = self.every_best[-1]
                self.every_best.append(previous_best)
            else:
                self.every_best.append(("None", "None"))
            
            return 0.0,
        
        the_test = RoadTestFactory.create_road_test(road_points)

        
        # Send the test for execution
        test_outcome, description, execution_data = self.executor.execute_test(the_test)

        # Print test outcome
        # log.info("test_outcome %s", test_outcome)
        # log.info("description %s", description)

        # Collect the oob_percentage values
        oob_percentages = [state.oob_percentage for state in execution_data]
        # log.info("Collected %d states information. Max is %.3f", len(oob_percentages), max(oob_percentages))

        # Compute the fitness
        if len(oob_percentages) == 0:
            fitness = 0.0
        else:
            # fitness = sum(oob_percentages) / len(oob_percentages)
            fitness = max(oob_percentages)  # TODO: change this to a better fitness function
        
        current_time = time.time() - start_time

        if fitness != 0:
            if fitness > 0.15:
                test_outcome = "PASS"
                description = "Car left the lane"
            else:
                test_outcome = "FAIL"
                description = "Car did not leave the lane"

        if test_outcome == "PASS" or test_outcome == "FAIL":
            self.valid_roads[fitness] = road_points
            time_elapsed = time.time() - start_time
            self.time_elapsed_list.append(time_elapsed)

            if self.best_solution == []:
                        self.best_solution.append((fitness, road_points))
            else:
                if fitness > self.best_solution[0][0]:
                    self.best_solution[0] = (fitness,road_points)
        
        # initialize test_count if it's not already defined
        try:
            self.test_count += 1
        except NameError:
            self.test_count = 1
        
        if self.best_solution != []:
                    best_fitness = round(self.best_solution[0][0], 5)
                    best_road = self.best_solution[0][1]

        log.info("-------------------------------------------------")
        log.info(f"TEST NUMBER: {self.test_count}")
        log.info(f"Road_points: {road_points}")
        self.every_road.append(road_points)
        log.info(f"Test_outcome: {test_outcome}")
        self.every_test_outcome.append(test_outcome)
        log.info(f"Description: {description}")
        self.every_description.append(description)
        log.info(f"Fitness: {fitness:.5f}")  
        self.every_fitness.append(fitness)
        self.every_time.append(current_time)
        if self.best_solution != []:
            log.info(f"Current best solution: {best_road}, fitness: {best_fitness}") 
            self.every_best.append((best_fitness,best_road))
        else:
            log.info(f"Current best solution: {self.best_solution}") 
            self.every_best.append(("None", "None"))
        log.info("-------------------------------------------------")


        
    

        return fitness,  # important to return a tuple since deap considers multiple objectives




    

    


    def start(self):
        log.info("Starting test generation")

        

        start_time = time.time()
        
        # Define the problem type
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        num_road_points = 3  # number of points in the road; by default, we want to generate 3 points to make one curve
        toolbox = base.Toolbox()
        # an attribute is a point in the road
        toolbox.register("attribute", self.init_attribute)
        # an individual is road_points (list)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=num_road_points)
        # a population is a list of individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register the crossover and mutation operators' hyperparameters
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate_tuple, mu=0, sigma=self.map_size/10, indpb=1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Register the fitness evaluation function
        toolbox.register("evaluate", self.evaluate, start_time = start_time)

        # Run a simple ready-made GA
        pop_size = 10  # population size
        num_generations = 5  # number of generations
        hof = tools.HallOfFame(1)  # save the best one individual during the whole search
        pop, deap_log = self.eaSimple(population=toolbox.population(n=pop_size),
                                                 toolbox=toolbox,
                                                 halloffame=hof,
                                                 cxpb=0.85,
                                                 mutpb=0.5,
                                                 ngen=num_generations,
                                                 verbose=True)

        log.info("TEST HAS BEEN COMPLETED")

        if self.best_solution != []:
            best_fitness = round(self.best_solution[0][0], 5)
            best_road = self.best_solution[0][1]
        

            log.info(f"Best individual: {best_road}, fitness: {best_fitness}")
        
        
        experiment = {
            'fitness': self.every_fitness,
            'road points' : self.every_road,
            'test outcome' : self.every_test_outcome,
            'description' : self.every_description,
            'current best' : self.every_best,
            'time elapsed' : self.every_time
        }

        df_main = pd.DataFrame(experiment)

        df_main = df_main.round({'fitness' : 5, 'time elapsed' : 2})

        df_main.insert(0, 'generation number', range(1, len(df_main)+1))

            # Create a dictionary with the desired data
        data = {
            'fitness': list(self.valid_roads.keys()),
            'road points': list(self.valid_roads.values()),
            'time elapsed': self.time_elapsed_list
        }

        # Create a new DataFrame from the dictionary
        df = pd.DataFrame(data)

        df = df.round({'fitness' : 5, 'time elapsed' : 2})

        df = df.sort_values('time elapsed', ascending=True)
        
        df['lane violation'] = np.where(df['fitness'] >= 0.15, 1, 0)

        df_lane_violation = df[df['lane violation'] == 1]

        df_lane_violation = df_lane_violation.drop('lane violation', axis=1)
        df = df.drop('lane violation', axis=1)

        df_lane_violation.insert(0, 'Lane Violation', range(1, len(df_lane_violation)+1))

        df.insert(0, 'Unique road', range(1, len(df)+1))


        base_filename = 'results/GA_{}.csv'
        i = 1

        while i <= 20:
            filename = base_filename.format(i)
            if not os.path.isfile(filename):
                break
            i += 1

        # Save the DataFrame to a CSV file
        df_main.to_csv(filename, index=False, mode = 'w')

        df.to_csv(filename, index=False, mode = 'a')    

        df_lane_violation.to_csv(filename, index=False, mode = 'a')


        
        

        with open(filename, 'a') as f:
            f.write('\n')
            f.write(f'Total number of roads generated: {self.test_count}')
            f.write('\n')
            f.write(f'Number of invalid roads: {self.test_count - len(self.valid_roads.keys())}')
            f.write('\n')
            f.write(f'Number of redundant roads: {self.redundant_count}')
            f.write('\n')
            if self.best_solution != []:
                f.write(f"Best solution: {best_road}, fitness: {best_fitness}")



         