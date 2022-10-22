# Team: DSD
Repository for autonomous driving in DuckieTown environment developed by the DSD team.

The goal of this project is to train and test a self driving AI vehicle in the Duckie Town World simulation. During our work, we are planning to use a TensorFlow based imitation learning algorithm. We are starting out from the base project [Behavior Cloning](https://github.com/duckietown/challenge-aido_LF-baseline-behavior-cloning) which contains utilities for data generation, data visualisation, model training and model testing. Apart from that we will use various open source Duckie Town repositories which will be mentioned as we advance. Certain components of these projects will be integrated into our own solution as we modified and expanded them with new ideas and solutions to reach better results.

## Milestone 1: Collecting data

To collect a proper amount of data for training our model we are using the Duckie Town Simulator available at [Gym Duckie Town](https://github.com/duckietown/gym-duckietown). We are generating data while driving the vehicle in the simulator, capturing images assigned with the appropriate action (e.g. right, left, forward, back).

Team members:
Ódor Dávid - IFZYRQ
Weyde Szabolcs - DC6KRO
Frey Dominik - AXHBUS

