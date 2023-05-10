from random import randint, gauss, random

import deap.algorithms


import numpy as np
from code_pipeline.tests_generation import RoadTestFactory
from time import  sleep

import os
import time
import logging as log
import pandas as pd

class HillClimbingGenerator():
    """
        This simple (naive) test generator creates roads using 4 points randomly placed on the map.
        We expect that this generator quickly creates plenty of tests, but many of them will be invalid as roads
        will likely self-intersect.
    """

    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size



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
                if point[0] > self.map_size - 10:    # capping at 10 - mapsize-10 because of the way roads are interpolated (want the road to stay inside map boundaries)
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


         


    def start(self):


        from deap import base
        from deap import creator
        from deap import tools

        all_roads = {
            # 0.1245: [(50, 50), (75, 75), (150, 150)], #should be removed
            # 0.3857: [(52, 52), (76, 73), (154, 153)], #should stay
            # 0.2857: [(50, 50), (76, 73), (150, 150)], #should be removed
            # 0.3002: [(50, 50), (72, 75), (150, 150)], #should be removed 
            # 0.3111: [(50, 50), (73, 76), (150, 150)], # should be removed
            # 0.9573: [(50, 50), (90, 95), (150, 150)], # should stay
            # 0.2111: [(50, 50), (73, 76), (150, 150)], # should be removed
        }

     

        best_solution = []

        test_count = 0

        invalid_tests = 0

        time_elapsed_list = [
            # 1,2,3,4,5,6,7
            ]

        start_time = time.time()

        iteration = 0


        redundant_count = 0
        every_road = []
        every_test_outcome = []
        every_description = []
        every_fitness = []
        every_best = []
        every_time = []


        
        
        while test_count < 60:
            try:
                # Some debugging

                iteration += 1
                log.info("-------------------------------------------------")
                log.info(f"TEST NUMBER: {iteration}")

                # Simulate the time to generate a new test
                sleep(0.5)
                # Pick up random points from the map. They will be interpolated anyway to generate the road
                road_points = []
                
                
                

                if best_solution == []:
                    for i in range(0, 3):
                        road_points.append((randint(10, self.map_size - 10), randint(10, self.map_size - 10)))

                else:
                    mutant = best_solution[0][1]
                    ind2, = self.mutate_tuple(mutant, mu=0.0, sigma=self.map_size/10, indpb=1)
                    road_points = ind2
                

                if self.is_redundant(every_road, road_points):
                    log.info("Test_outcome: REDUNDANT")
                    test_count += 1
                    current_time = time.time() - start_time
                    redundant_count += 1
                    every_road.append(list(road_points))
                    every_test_outcome.append("REDUNDANT")
                    every_description.append("Redundant test")
                    every_fitness.append("Redundant test")
                    every_time.append(current_time)
                    if best_solution != []:
                        previous_best = every_best[-1]
                        every_best.append(previous_best)
                    else:
                        every_best.append(("None", "None"))
                    continue

                # Some more debugging
                # log.info("Generated test using: %s", road_points)
                # Decorate the_test object with the id attribute
                the_test = RoadTestFactory.create_road_test(road_points)

                # Try to execute the test
                test_outcome, description, execution_data = self.executor.execute_test(the_test)

                oob_percentages = [state.oob_percentage for state in execution_data]
                if len(oob_percentages) == 0:
                    fitness = 0.0
                else:
                    fitness = max(oob_percentages)  
                

          
                current_time = time.time() - start_time

                if fitness != 0:
                    if fitness > 0.15:
                        test_outcome = "PASS"
                        description = "Car left the lane"
                    else:
                        test_outcome = "FAIL"
                        description = "Car did not leave the lane"

                if test_outcome == "PASS" or test_outcome == "FAIL":
                    
                    all_roads[fitness] = list(road_points)
                    time_elapsed = time.time() - start_time
                    time_elapsed_list.append(time_elapsed)

                    if best_solution == []:
                        best_solution.append((fitness, road_points))
                    else:
                        if fitness > best_solution[0][0]:
                            best_solution[0] = (fitness,road_points)

                else:
                    invalid_tests += 1
                    


                test_count += 1

                
                if best_solution != []:
                    best_fitness = round(best_solution[0][0], 5)
                    best_road = best_solution[0][1]


                
                # Print the result from the test and continue
                log.info(f"Road_points: {road_points}")
                every_road.append(list(road_points))
                log.info(f"Test_outcome: {test_outcome}")
                every_test_outcome.append(test_outcome)
                log.info(f"Description: {description}")
                every_description.append(description)
                log.info(f"Fitness: {fitness:.5f}")  
                every_fitness.append(fitness)
                every_time.append(current_time)
                if best_solution != []:
                    log.info(f"Current best solution: {best_road}, fitness: {best_fitness}") 
                    every_best.append((best_fitness,best_road))
                else:
                    log.info(f"Current best solution: {best_solution}") 
                    every_best.append(("None", "None"))
                log.info("-------------------------------------------------")   
                
            except KeyboardInterrupt:
                break
            
           
        
        log.info("TEST HAS BEEN COMPLETED")



        experiment = {
            'fitness': every_fitness,
            'road points' : every_road,
            'test outcome' : every_test_outcome,
            'description' : every_description,
            'current best' : every_best,
            'time elapsed' : every_time
        }

        df_main = pd.DataFrame(experiment)

        df_main = df_main.round({'fitness' : 5, 'time elapsed' : 2})

        df_main.insert(0, 'generation number', range(1, len(df_main)+1))

        
            # Create a dictionary with the desired data
        data = {
            'fitness': list(all_roads.keys()),
            'road points': list(all_roads.values()),
            'time elapsed': time_elapsed_list
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



        
        base_filename = 'results/HC_{}.csv'
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

        if best_solution != []:
            best_fitness = round(best_solution[0][0], 5)
            best_road = best_solution[0][1]
            
        with open(filename, 'a') as f:
            f.write('\n')
            f.write(f'Total number of roads generated: {test_count}')
            f.write('\n')
            f.write(f'Number of invalid roads: {invalid_tests}')
            f.write('\n')
            f.write(f'Number of redundant roads: {redundant_count}')
            f.write('\n')
            if best_solution != []:
                f.write(f"Best solution: {best_road}, fitness: {best_fitness}")                
        



