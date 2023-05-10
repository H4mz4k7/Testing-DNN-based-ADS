# Testing-DNN-based-ADS

This program allows for the easy use of testing ADS with multiple algorithms including:
- Random Algorithm
- Hill-Climbing Algorithm
- Genetic Algorithm
- CMA-ES Algorithm

Users can use this codebase as a foundation to explore the effects of other algorithms when used to test DNN-Based ADS.

This program has been adapted from the code available at https://github.com/sbft-cps-tool-competition/cps-tool-competition, hence the setup is the same.

SETUP: https://github.com/sbft-cps-tool-competition/cps-tool-competition/blob/main/documentation/INSTALL.md

After installation, you can test the ADS using the following commands:

Navigate to the cps-tool-competition-main directory and run the following to run the algorithms:

-Random Algorithm
`python competition.py --time-budget 3600 --executor beamng --map-size 200 --module-name sample_test_generators.random_generator --class-name RandomTestGenerator`

-Hill-Climbing Algorithm
`python competition.py --time-budget 3600 --executor beamng --map-size 200 --module-name sample_test_generators.HC_generator --class-name HillClimbingGenerator
done`

-Genetic Algorithm
`python competition.py --time-budget 3600 --executor beamng --map-size 200 --module-name sample_test_generators.GA_test_generator --class-name GATestGenerator
done`

-CMA-ES Algorithm
`python competition.py --time-budget 3600 --executor beamng --map-size 200 --module-name sample_test_generators.CMA-ES_generator --class-name CMATestGenerator`


If you would like to run the algorithms one after the other 20 times each, you can run the following command in a linux terminal:
`bash run_tests.sh`
