from random import randint
import random

import numpy as np
from code_pipeline.tests_generation import RoadTestFactory
from time import  sleep

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

        test_count = 0

        invalid_tests = 0

        time_elapsed_list = [
            # 1,2,3,4,5,6,7
            ]

        start_time = time.time()

        iteration = 0
        
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
                
        
                

                if candidate_solution == []:
                    for i in range(0, 3):
                        road_points.append((randint(0, self.map_size), randint(0, self.map_size)))

                else:
                    candidate_road_points = candidate_solution[0][1] #gives [(x,y),(x,y),(x,y)]
                    for i in range(0, 3):
                        random_number = randint(5, 30)
                        x, y = candidate_road_points[i] # unpack the tuple into separate x and y variables
                        is_addition = random.choice([False, True])
                        if is_addition:
                            if random_number + x <= 200:
                                x += random_number
                            else:
                                x -= random_number

                            if random_number + y <= 200:
                                y += random_number
                            else:
                                y -= random_number

                        else:
                            if x - random_number >= 0:
                                x -= random_number
                            else :
                                x += random_number

                            if y - random_number >= 0:
                                y -= random_number
                            else:
                                y += random_number

                        candidate_road_points[i] = (x, y) # create a new tuple with the modified values
                    road_points = candidate_road_points
                    

                

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


                candidate_fitness = round(candidate_solution[0][0], 5)
                candidate_road = candidate_solution[0][1]

                # Print the result from the test and continue
                log.info(f"Road_points: {road_points}")
                log.info(f"Test_outcome: {test_outcome}")
                log.info(f"Description: {description}")
                log.info(f"Fitness: {fitness:.5f}")  
                log.info(f"Current candidate solution: {candidate_road}, fitness: {candidate_fitness}") 
                log.info("-------------------------------------------------")   
                
            except KeyboardInterrupt:
                break
            
           
        
        removed_redundant, removed_redundant_time  = self.check_redundancy(all_roads, time_elapsed_list)
        log.info("TEST HAS BEEN COMPLETED")


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

        candidate_fitness = round(candidate_solution[0][0], 5)
        candidate_road = candidate_solution[0][1]
            
        with open('results/SHC.csv', 'a') as f:
            f.write('\n')
            f.write(f'Total number of roads generated: {test_count}')
            f.write('\n')
            f.write(f'Number of valid roads: {len(all_roads.keys())}')
            f.write('\n')
            f.write(f'Number of invalid roads: {invalid_tests}')
            f.write('\n')
            f.write(f'Number of redundant roads: {len(all_roads.keys()) - len(removed_redundant.keys())}')
            f.write('\n')
            f.write(f"Candidate solution: {candidate_road}, fitness: {candidate_fitness}")
                
        



