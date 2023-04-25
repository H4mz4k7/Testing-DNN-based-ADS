import os
import time
import deap.algorithms
import numpy as np
import math
import logging as log
from random import randint, gauss, random
import matplotlib.pyplot as plt
import pandas as pd

from code_pipeline.tests_generation import RoadTestFactory


class CMATestGenerator():
    """
        Generates a set of tests using GA (based on the Deap library; https://github.com/deap/deap).

    """

    

    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size
        self.valid_roads = {}
        self.time_elapsed_list = []
        self.test_count = 0
        self.candidate_solution = []
        self.every_road = []
        self.every_test_outcome = []
        self.every_description = []
        self.every_fitness = []
        self.every_candidate = []
        self.every_time = []

    def init_attribute(self):
        
        attribute = randint(10, self.map_size-10)
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




    

    def evaluate(self, individual, start_time):
        

        road_points = list([(individual[i], individual[i+1]) for i in range(0, len(individual), 2)])
        # Creating the RoadTest from the points
        
       
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

        if fitness > 0:
            test_outcome = "PASS"

        if test_outcome == "PASS":
            self.valid_roads[fitness] = road_points
            time_elapsed = time.time() - start_time
            self.time_elapsed_list.append(time_elapsed)

            if self.candidate_solution == []:
                        self.candidate_solution.append((fitness, road_points))
            else:
                if fitness > self.candidate_solution[0][0]:
                    self.candidate_solution[0] = (fitness,road_points)
        
        # initialize test_count if it's not already defined
        try:
            self.test_count += 1
        except NameError:
            self.test_count = 1
        
        if self.candidate_solution != []:
                    candidate_fitness = round(self.candidate_solution[0][0], 5)
                    candidate_road = self.candidate_solution[0][1]

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
        if self.candidate_solution != []:
            log.info(f"Current candidate solution: {candidate_road}, fitness: {candidate_fitness}") 
            self.every_candidate.append((candidate_fitness,candidate_road))
        else:
            log.info(f"Current candidate solution: {self.candidate_solution}") 
            self.every_candidate.append(("None", "None"))
        log.info("-------------------------------------------------")


        
    

        return fitness,  # important to return a tuple since deap considers multiple objectives

    
    def num_random_for_centroid(self, num_road_points):
        num_coord_ind = num_road_points * 2
        coord_list = []
        for i in range(num_coord_ind):
            coord_list.append(randint(10,self.map_size - 10))

        return coord_list
            
        
   

    def start(self):
        log.info("Starting test generation")

        from deap import base
        from deap import creator
        from deap import tools
        from deap import cma
        

        start_time = time.time()
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        num_road_points = 3  # number of points in the road; by default, we want to generate 3 points to make one curve
        toolbox = base.Toolbox()
        # an attribute is a point in the road
        toolbox.register("attribute", self.init_attribute)
        # an individual is road_points (list)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=num_road_points * 2)

        
        
        
        # Define the CMA-ES parameters
        pop_per_gen = 10      
        strategy = cma.Strategy(centroid= self.num_random_for_centroid(num_road_points), sigma=self.map_size/10, lambda_ = pop_per_gen)
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        # Register the fitness evaluation function
        toolbox.register("evaluate", self.evaluate, start_time = start_time)

        # Run CMA-ES
        num_generations = 6   # number of generations
        hof = tools.HallOfFame(1)  # save the best one individual during the whole search
        pop, logbook = deap.algorithms.eaGenerateUpdate(toolbox, ngen=num_generations, halloffame=hof, verbose=True)
            
        # Print the best individual from the hall of fame
        best_individual = tools.selBest(hof, 1)[0]


        print("TEST HAS BEEN COMPLETED")

        if self.candidate_solution != []:
            candidate_fitness = round(self.candidate_solution[0][0], 5)
            candidate_road = self.candidate_solution[0][1]
        
            print(f"Best individual: {candidate_road}, fitness: {candidate_fitness}")

        
        removed_redundant, removed_redundant_time = self.check_redundancy(self.valid_roads, self.time_elapsed_list)
        
        
        experiment = {
            'fitness': self.every_fitness,
            'road points' : self.every_road,
            'test outcome' : self.every_test_outcome,
            'description' : self.every_description,
            'current candidate' : self.every_candidate,
            'time elapsed' : self.every_time
        }

        df_main = pd.DataFrame(experiment)

        df_main = df_main.round({'fitness' : 5, 'time elapsed' : 2})

        df_main.insert(0, 'generation number', range(1, len(df_main)+1))

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


        base_filename = 'results/CMA-ES_{}.csv'
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


        total_roads = pop_per_gen * num_generations
        number_redundant = len(self.valid_roads.keys()) - len(removed_redundant.keys())

        
        

        with open(filename, 'a') as f:
            f.write('\n')
            f.write(f'Total number of roads generated: {total_roads}')
            f.write('\n')
            f.write(f'Number of valid roads: {len(self.valid_roads.keys())}')
            f.write('\n')
            f.write(f'Number of invalid roads: {total_roads - len(self.valid_roads.keys())}')
            f.write('\n')
            f.write(f'Number of redundant roads: {number_redundant}')
            f.write('\n')
            if self.candidate_solution != []:
                f.write(f"Candidate solution: {candidate_road}, fitness: {candidate_fitness}")



         