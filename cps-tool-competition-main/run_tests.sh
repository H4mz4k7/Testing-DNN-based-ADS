#!/bin/bash

for i in {1..2}
do
  python competition.py --time-budget 3600 --executor beamng --map-size 200 --module-name sample_test_generators.CMA-ES_generator --class-name CMATestGenerator
done