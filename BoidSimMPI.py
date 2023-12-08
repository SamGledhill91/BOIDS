from math import sqrt
import numpy as np
import time
from PIL import Image, ImageDraw
from mpi4py import MPI

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()

num_boids = 1000
num_frames = 100
width = height = 1000
visual_range = 50
min_distance = 30
centering_factor = 0.05
matching_factor = 0.1
avoid_factor = 0.01
speed_limit = 15
render = True

def GenerateNeighbours(x_values, y_values, num_boids):
    distances_squared = np.sum((np.stack([x_values, y_values], axis=1)[:, np.newaxis, :] -
                                np.stack([x_values, y_values], axis=1)[np.newaxis, :, :]) ** 2, axis=-1)

    is_neighbor = (distances_squared > 0) & (distances_squared < visual_range ** 2)

    neighbour_list = np.zeros((num_boids, num_boids + 1), dtype=int)

    for boid_outer in range(num_boids):
        neighbors = np.where(is_neighbor[boid_outer])[0]
        neighbour_list[boid_outer, 0] = len(neighbors)

        for i, neighbor in enumerate(neighbors, start=1):
            neighbour_list[boid_outer, i] = neighbor

    return neighbour_list

if my_rank == 0:
    print(f"Simulating {num_boids} Boids for {num_frames} frames")

start_time = time.time()

# Distribute the work among processes
boid_range = range(my_rank * num_boids // p, (my_rank + 1) * num_boids // p)
print(f"Rank {my_rank}: Boid range: {boid_range}")

# Broadcasting initial positions, velocities, and parameters
x_values = np.random.uniform(0, width, num_boids)
y_values = np.random.uniform(0, height, num_boids)
vx_values = np.random.uniform(-5, 5, num_boids)
vy_values = np.random.uniform(-5, 5, num_boids)

x_values = comm.bcast(x_values, root=0)
y_values = comm.bcast(y_values, root=0)
vx_values = comm.bcast(vx_values, root=0)
vy_values = comm.bcast(vy_values, root=0)

for frame_num in range(num_frames):
    if my_rank == 0:
        print(f"Simulating frame {frame_num}...")

    # Broadcasting parameters to all processes
    visual_range = comm.bcast(visual_range, root=0)
    min_distance = comm.bcast(min_distance, root=0)
    centering_factor = comm.bcast(centering_factor, root=0)
    matching_factor = comm.bcast(matching_factor, root=0)
    avoid_factor = comm.bcast(avoid_factor, root=0)
    speed_limit = comm.bcast(speed_limit, root=0)

    neighbour_list = GenerateNeighbours(x_values, y_values, num_boids)

    for boid_outer in boid_range:
        current_neighbours = neighbour_list[boid_outer, 1:1 + neighbour_list[boid_outer, 0]]
        n = neighbour_list[boid_outer, 0]
        if n == 0:
            avg_x = 0.0
            avg_y = 0.0
            avg_vx = 0.0
            avg_vy = 0.0
        else:
            sum_x = sum(x_values[neighbour] for neighbour in current_neighbours)
            sum_y = sum(y_values[neighbour] for neighbour in current_neighbours)
            sum_vx = sum(vx_values[neighbour] for neighbour in current_neighbours)
            sum_vy = sum(vy_values[neighbour] for neighbour in current_neighbours)

            inv_count = 1.0 / n
            avg_x = sum_x * inv_count
            avg_y = sum_y * inv_count
            avg_vx = sum_vx * inv_count
            avg_vy = sum_vy * inv_count

        avoid_dx, avoid_dy = 0, 0

        for other_boid in current_neighbours:
            if (x_values[other_boid] - x_values[boid_outer]) ** 2 + (
                    y_values[other_boid] - y_values[boid_outer]) ** 2 < min_distance ** 2:
                avoid_dx += x_values[boid_outer] - x_values[other_boid]
                avoid_dy += y_values[boid_outer] - y_values[other_boid]

        vx_values[boid_outer] += (avg_x - x_values[boid_outer]) * centering_factor \
            + (avg_vx - vx_values[boid_outer]) * matching_factor \
            + avoid_dx * avoid_factor

        vy_values[boid_outer] += (avg_y - y_values[boid_outer]) * centering_factor \
            + (avg_vy - vy_values[boid_outer]) * matching_factor \
            + avoid_dy * avoid_factor

        margin = 100
        turn_factor = 3

        if x_values[boid_outer] < margin:
            vx_values[boid_outer] += turn_factor

        if x_values[boid_outer] > width - margin:
            vx_values[boid_outer] -= turn_factor

        if y_values[boid_outer] < margin:
            vy_values[boid_outer] += turn_factor

        if y_values[boid_outer] > height - margin:
            vy_values[boid_outer] -= turn_factor

        speed = sqrt(vx_values[boid_outer] ** 2 + vy_values[boid_outer] ** 2)
        if speed > speed_limit:
            speed_factor = speed_limit / speed
            vx_values[boid_outer] *= speed_factor
            vy_values[boid_outer] *= speed_factor

        x_values[boid_outer] += vx_values[boid_outer]
        y_values[boid_outer] += vy_values[boid_outer]

    # Gather updated positions to process 0
    all_x_values = comm.gather(x_values, root=0)
    all_y_values = comm.gather(y_values, root=0)

    if my_rank == 0:
        # Combine positions from all processes
        x_values = np.concatenate(all_x_values)
        y_values = np.concatenate(all_y_values)

    # Broadcasting updated positions to all processes
    x_values = comm.bcast(x_values, root=0)
    y_values = comm.bcast(y_values, root=0)

# Scatter the boid states to all processes
scatter_boid_states = comm.scatterv(all_boid_states, root=0)

# Gather boid states to process 0
all_boid_states = comm.gather(scatter_boid_states, root=0)

elapsed_time = time.time() - start_time

if my_rank == 0:
    print("SIMULATION COMPLETE")
    print(f"Time taken: {elapsed_time}")
