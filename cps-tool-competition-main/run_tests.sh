#!/bin/bash




for i in {1..20}
do
  python competition.py --time-budget 3600 --executor beamng --map-size 200 --module-name sample_test_generators.random_generator --class-name RandomTestGenerator
done
for i in {1..20}
do
  python competition.py --time-budget 3600 --executor beamng --map-size 200 --module-name sample_test_generators.HC_generator --class-name HillClimbingGenerator
done
for i in {1..20}
do
  python competition.py --time-budget 3600 --executor beamng --map-size 200 --module-name sample_test_generators.GA_test_generator --class-name GATestGenerator
done
for i in {1..20}
do
  python competition.py --time-budget 3600 --executor beamng --map-size 200 --module-name sample_test_generators.CMA-ES_generator --class-name CMATestGenerator
done
