import os
from random import randint
import time
import numpy as np

import pandas as pd
from code_pipeline.tests_generation import RoadTestFactory
from time import  sleep

import logging as log


class RandomTestGenerator():
    """
        This simple (naive) test generator creates roads using 4 points randomly placed on the map.
        We expect that this generator quickly creates plenty of tests, but many of them will be invalid as roads
        will likely self-intersect.
    """

    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size


    def is_redundant(self, all_roads, road_points):
        all_road_points = all_roads
        is_redundant_flag = False
        if road_points in all_road_points:
            is_redundant_flag = True
            return is_redundant_flag
        
      
        for i in range(len(all_road_points)):
            road_point_one = road_points
            road_point_two = all_road_points[i]
            print(road_point_one)
            print(road_point_two)
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


        all_roads = {
            # 0.1245: [(50, 50), (75, 75), (150, 150)], #should be removed
            # 0.3857: [(52, 52), (76, 73), (154, 153)], #should stay
            # 0.2857: [(50, 50), (76, 73), (150, 150)], #should be removed
            # 0.3002: [(50, 50), (72, 75), (150, 150)], #should be removed 
            # 0.3111: [(50, 50), (73, 76), (150, 150)], # should be removed
            # 0.9573: [(50, 50), (90, 95), (150, 150)], # should stay
            # 0.2111: [(50, 50), (73, 76), (150, 150)], # should be removed
        }

        candidate_solution = []

        testing_count = 0

        test_count = 0

        invalid_tests = 0
        iteration = 0

        time_elapsed_list = [
            # 1,2,3,4,5,6,7
            ]
        
        start_time = time.time()


        redundant_count = 0
        every_road = []
        every_test_outcome = []
        every_description = []
        every_fitness = []
        every_candidate = []
        every_time = []

        while test_count < 60:
            try :

                iteration += 1
                log.info("-------------------------------------------------")
                log.info(f"TEST NUMBER: {iteration}")

                # Simulate the time to generate a new test
                sleep(0.5)
                # Pick up random points from the map. They will be interpolated anyway to generate the road
                road_points = []
                # for i in range(0, 3):
                #     road_points.append((randint(10, self.map_size - 10), randint(10, self.map_size - 10)))

            

                if self.is_redundant(every_road, road_points):
                    test_count += 1
                    current_time = time.time() - start_time
                    redundant_count += 1
                    every_road.append(road_points)
                    every_test_outcome.append("Redundant test")
                    every_description.append("Redundant test")
                    every_fitness.append("Redundant test")
                    every_time.append(current_time)
                    if candidate_solution != []:
                        previous_candidate = every_candidate[-1]
                        every_candidate.append(previous_candidate)
                    else:
                        every_candidate.append(("None", "None"))
                    continue


                the_test = RoadTestFactory.create_road_test(road_points)

                # Try to execute the test
                test_outcome, description, execution_data = self.executor.execute_test(the_test)
                print(test_outcome)
            
                oob_percentages = [state.oob_percentage for state in execution_data]
                if len(oob_percentages) == 0:
                    fitness = 0.0
                else:
                    fitness = max(oob_percentages)  

                if fitness > 0:
                    test_outcome = "PASS"
                
                current_time = time.time() - start_time

                if test_outcome == "PASS":

                    all_roads[fitness] = list(road_points)
                    time_elapsed = time.time() - start_time
                    time_elapsed_list.append(time_elapsed)

                    if candidate_solution == []:
                        candidate_solution.append((fitness, road_points))
                    else:
                        if fitness > candidate_solution[0][0]:
                            candidate_solution[0] = (fitness,road_points)

                else:
                    invalid_tests += 1


                test_count += 1

                if candidate_solution != []:
                    candidate_fitness = round(candidate_solution[0][0], 5)
                    candidate_road = candidate_solution[0][1]

                # Print the result from the test and continue
                log.info(f"Road_points: {road_points}")
                every_road.append(road_points)
                log.info(f"Test_outcome: {test_outcome}")
                every_test_outcome.append(test_outcome)
                log.info(f"Description: {description}")
                every_description.append(description)
                log.info(f"Fitness: {fitness:.5f}")  
                every_fitness.append(fitness)
                every_time.append(current_time)
                if candidate_solution != []:
                    log.info(f"Current candidate solution: {candidate_road}, fitness: {candidate_fitness}") 
                    every_candidate.append((candidate_fitness,candidate_road))
                else:
                    log.info(f"Current candidate solution: {candidate_solution}") 
                    every_candidate.append(("None", "None"))
                log.info("-------------------------------------------------")   

            except KeyboardInterrupt:
                break




        log.info("TEST HAS BEEN COMPLETED")

        experiment = {
            'fitness': every_fitness,
            'road points' : every_road,
            'test outcome' : every_test_outcome,
            'description' : every_description,
            'current candidate' : every_candidate,
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



        base_filename = 'results/random_{}.csv'
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
        
        if self.candidate_solution != []:
            candidate_fitness = round(candidate_solution[0][0], 5)
            candidate_road = candidate_solution[0][1]

        with open(filename, 'a') as f:
            f.write('\n')
            f.write(f'Total number of roads generated: {test_count}')
            f.write('\n')
            f.write(f'Number of invalid roads: {invalid_tests}')
            f.write('\n')
            f.write(f'Number of redundant roads: {redundant_count}')
            f.write('\n')
            if self.candidate_solution != []:
                f.write(f"Candidate solution: {candidate_road}, fitness: {candidate_fitness}")