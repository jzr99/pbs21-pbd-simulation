
# ETHz Physically-based Simulation Course Project
# Position Based Dynamics

## Introduction

- Our project is to simulate a cloth soft body to interact with it self and static object. Based on Taichi and Open3D tools, we implemented the simulation pipeline from scratch according to the paper by Muller. Our main reference code is from the course exercise of mass spring system. 

## Running Our Code

* Step 1: `git checkout <demo_branch>`
* Step 2: `python runner.py` 
* If you want to tune the parameters, you can change rendering option in `runner.py` and simulation parameters in `lib/simulation.py`. 
* If `saving` is true in render option, images will be exported to folder `Image/{timestamp}`. If you want to make video, please use `make_video.py` script to generate videos from images. 

## Demo Branch

* `demo-ball-cloth-colliding`: The cloth falls on the ball
* `demo-cloth-falling`: Cloth self collision.
* `demo-stanford-bunny`: The cloth falls on the stanford bunny.
* `demo-with-wind`: The cloth is blowed by wind
* `diffTaichi` Use DiffTaichi to make the cloth fall on the ball by controlling the wind speed.

## Project Presentation

* [Youtube Video](https://www.youtube.com/watch?v=h73SmnWRSQI&lc=UgynlYpae8q3t-EfEXB4AaABAg)

## Reference Paper

* [Position Based Dynamics](https://www.sciencedirect.com/science/article/pii/S1047320307000065)
