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
        self.best_solution = []
        self.every_road = []
        self.redundant_count = 0
        self.every_test_outcome = []
        self.every_description = []
        self.every_fitness = []
        self.every_best = []
        self.every_time = []

    def init_attribute(self):
        
        attribute = randint(10, self.map_size-10)
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




    

    def evaluate(self, individual, start_time):
        

        road_points = list([(individual[i], individual[i+1]) for i in range(0, len(individual), 2)])
        # Creating the RoadTest from the points
        
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

        if self.best_solution != []:
            best_fitness = round(self.best_solution[0][0], 5)
            best_road = self.best_solution[0][1]
        
            print(f"Best individual: {best_road}, fitness: {best_fitness}")

                
        
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



         