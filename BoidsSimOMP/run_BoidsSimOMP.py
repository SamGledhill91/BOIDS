import sys
import BoidsSimOMP

if len(sys.argv) == 5:
    BoidsSimOMP.simulate_boids(
        num_boids=int(sys.argv[1]),
        num_frames=int(sys.argv[2]),
        threads=int(sys.argv[3]),
        width=1000,
        height=1000,
        visual_range=50,
        min_distance=30,
        centering_factor=0.05,
        matching_factor=0.1,
        avoid_factor=0.01,
        speed_limit=15,
        render=sys.argv[4].lower() == 'true'
    )
else:
    print("Usage: python {} <NUM BOIDS> <NUM FRAMES> <NUM THREADS> <RENDER (BOOL)>".format(sys.argv[0]))

