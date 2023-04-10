import time
import deap.algorithms
import numpy as np
import math
import logging as log
from random import randint, gauss, random
import matplotlib.pyplot as plt
import pandas as pd

from code_pipeline.tests_generation import RoadTestFactory


class GATestGenerator():
    """
        Generates a set of tests using GA (based on the Deap library; https://github.com/deap/deap).

    """

    candidate_solutions = {}
    time_elapsed_list = []
    test_count = 0
    lane_violations = 0

    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size

    def init_attribute(self):
        
        attribute = (randint(0, self.map_size), randint(0, self.map_size))
        return attribute
    

    def check_redundancy(self, all_roads, time_elapsed_list):
        
        removed_copies = {}


        keys_list = list(all_roads.keys())
        #checks for any duplicate roads and removes the one with the smaller key
        for key in all_roads:
            # If the value of the current key is already in the new dictionary,
            # compare the keys and only keep the larger one
            if all_roads[key] in removed_copies.values():
                for k, v in removed_copies.items():
                    if v == all_roads[key] and key > k:
                        index = keys_list.index(k)
                        del time_elapsed_list[index]
                        del removed_copies[k]
                        removed_copies[key] = all_roads[key]
            # If the value of the current key is not in the new dictionary,
            # add the key/value pair to the new dictionary
            else:
                removed_copies[key] = all_roads[key]

        

        
        #sort the time list in the same order as the dictionary after its been sorted
        keys_without_copies = list(removed_copies)
        sorted_indices = sorted(range(len(keys_without_copies)), key=lambda k: keys_without_copies[k])

        sorted_keys = [keys_without_copies[i] for i in sorted_indices]
        sorted_time = [time_elapsed_list[i] for i in sorted_indices]
        
        road_points = [all_roads[key] for key in sorted_keys]
        values_to_remove = []
        time_to_remove = []
        
        #iterate through roads, comparing the "first" and "second" roads depending on where counter i is. 
        #normalise roads by changing road two to start at the same point as road one
        #if mid points and end points of the two roads are between +/- 5 then remove the one with smaller fitness (first road)
        for i in range(len(road_points)-1):
            road_point_one = road_points[i]
            for j in range(i+1, len(road_points)):
                road_point_two = road_points[j]

                

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
                        values_to_remove.append(road_point_one)


        

        
        for key, value in list(removed_copies.items()):
            if value in values_to_remove:
                index = sorted_keys.index(key)
                time_to_remove.append(sorted_time[index]) #add the time that needs to be removed to a list
                del removed_copies[key]
                
        
        # remove specific times from list (if the key has been removed from dict)
        sorted_time = [x for x in sorted_time if x not in time_to_remove]


        removed_redundant = removed_copies
        removed_redundant_time = sorted_time

        return removed_redundant, removed_redundant_time


         

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
                if point[0] < 0:
                    point[0] = 0
                if point[0] > self.map_size:
                    point[0] = self.map_size

                # update the second value (y-pos)
                point[1] += int(gauss(mu, sigma))
                if point[1] < 0:
                    point[1] = 0
                if point[1] > self.map_size:
                    point[1] = self.map_size

                # update the attribute (tuple) in the individual
                individual[i] = tuple(point)

        return individual,




    def evaluate(self, individual, start_time):
        global candidate_solutions
        global time_elapsed_list
        global test_count
        global lane_violations

        # Creating the RoadTest from the points
        road_points = list(individual)
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
        

        if fitness >= 0.15:
            try:
                lane_violations += 1
            except NameError:
                lane_violations = 1
        # initialize test_count if it's not already defined
        try:
            test_count += 1
        except NameError:
            test_count = 1
            
        log.info("-------------------------------------------------")
        log.info(f"TEST NUMBER: {test_count}")
        log.info(f"Road_points: {road_points}")
        log.info(f"Test_outcome: {test_outcome}")
        log.info(f"Description: {description}")
        log.info(f"Fitness: {fitness:.5f}")  
        log.info("-------------------------------------------------")


        if test_outcome == "PASS":
            self.candidate_solutions[fitness] = road_points
            time_elapsed = time.time() - start_time
            self.time_elapsed_list.append(time_elapsed)
    

        return fitness,  # important to return a tuple since deap considers multiple objectives




    

    


    def start(self):
        log.info("Starting test generation")

        from deap import base
        from deap import creator
        from deap import tools

        start_time = time.time()
        time_elapsed_list = []

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
        pop, deap_log = deap.algorithms.eaSimple(population=toolbox.population(n=pop_size),
                                                 toolbox=toolbox,
                                                 halloffame=hof,
                                                 cxpb=0.85,
                                                 mutpb=0.1,
                                                 ngen=num_generations,
                                                 verbose=True)

        # Print the best individual from the hall of fame
        best_individual = tools.selBest(hof, 1)[0]
        log.info(f"Best individual: {best_individual}, fitness: {best_individual.fitness}")
        removed_redundant, removed_redundant_time = self.check_redundancy(self.candidate_solutions, self.time_elapsed_list)
        


            # Create a dictionary with the desired data
        data = {
            'fitness': list(removed_redundant.keys()),
            'road points': list(removed_redundant.values()),
            'time elapsed': removed_redundant_time
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

        # Save the DataFrame to a CSV file
        df.to_csv('results/SHC.csv', index=False, mode = 'w')

        with open('results/SHC.csv', 'a') as f:
            f.write('\n')
            f.write('\n')
            f.write('\n')
            f.write('\n')


        df_lane_violation.to_csv('results/SHC.csv', index=False, mode = 'a')


        with open('results/SHC.csv', 'a') as f:
            f.write('\n')
            f.write('\n')
            f.write('\n')
            f.write('\n')


        total_roads = pop_size * num_generations + pop_size
        number_redundant = len(self.candidate_solutions.keys()) - len(removed_redundant.keys())

        with open('results/SHC.csv', 'a') as f:
            f.write('\n')
            f.write(f'Total number of roads generated: {total_roads}')
            f.write('\n')
            f.write(f'Number of valid roads: {len(self.candidate_solutions.keys())}')
            f.write('\n')
            f.write(f'Number of invalid roads: {total_roads - len(self.candidate_solutions.keys())}')
            f.write('\n')
            f.write(f'Number of redundant roads: {number_redundant}')
            f.write('\n')
            f.write(f"Candidate solution: {best_individual}, fitness: {best_individual.fitness}")



         