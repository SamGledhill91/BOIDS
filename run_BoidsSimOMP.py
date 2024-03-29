import sys
import BoidsSimOMP
import numpy as np

BoidsSimOMP.simulate_boids(num_boids=2500, num_frames=10, width=1000, height=1000, visual_range=50, min_distance=30,
                   centering_factor=0.05, matching_factor=0.1, avoid_factor=0.01, speed_limit=15, render=False)