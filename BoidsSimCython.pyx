from libc.math cimport sqrt
import numpy as np
cimport numpy as np
import cython
import time
from PIL import Image, ImageDraw

cdef int num_boids, width, height, visual_range,min_distance, speed_limit
cdef double[:] x_values, y_values, vx_values, vy_values
cdef double centering_factor, matching_factor, avoid_factor
cdef list neighbours, boid_states
cdef Py_ssize_t i, boid_index, neighbour_index, other_boid, other_index
cdef bint render, output

def create_boids(num_boids, width, height):
    x_values = np.random.uniform(0, width, num_boids)
    y_values = np.random.uniform(0, height, num_boids)
    vx_values = np.random.uniform(-5, 5, num_boids)
    vy_values = np.random.uniform(-5, 5, num_boids)
    neighbours = [[] for _ in range(num_boids)]

    return x_values, y_values, vx_values, vy_values, neighbours

def distance(boid1, boid2, x_values, y_values):
    return sqrt((x_values[boid1] - x_values[boid2])**2 + (y_values[boid1] - y_values[boid2])**2)

def neighbour_list(boid_index, x_values, y_values, visual_range):
    neighbours = []
    for other_boid in range(len(x_values)):
        if boid_index != other_boid and distance(boid_index, other_boid, x_values, y_values) < visual_range:
            neighbours.append(other_boid)
    return neighbours

def calculate_averages(boid_index, neighbours, x_values, y_values, vx_values, vy_values):
    if not neighbours:
        return 0, 0, 0, 0

    sum_x = sum(x_values[neighbor] for neighbor in neighbours)
    sum_y = sum(y_values[neighbor] for neighbor in neighbours)
    sum_vx = sum(vx_values[neighbor] for neighbor in neighbours)
    sum_vy = sum(vy_values[neighbor] for neighbor in neighbours)

    count = len(neighbours)
    return sum_x / count, sum_y / count, sum_vx / count, sum_vy / count

def keep_within_bounds(boid_index, width, height, x_values, y_values, vx_values, vy_values):
    margin = 100
    turn_factor = 3

    if x_values[boid_index] < margin:
        vx_values[boid_index] += turn_factor
    if x_values[boid_index] > width - margin:
        vx_values[boid_index] -= turn_factor
    if y_values[boid_index] < margin:
        vy_values[boid_index] += turn_factor
    if y_values[boid_index] > height - margin:
        vy_values[boid_index] -= turn_factor

def update_velocity_and_position(boid_index, x_values, y_values, vx_values, vy_values, visual_range, min_distance, centering_factor, matching_factor, avoid_factor, speed_limit, width, height):
    neighbours = neighbour_list(boid_index, x_values, y_values, visual_range)
    avg_x, avg_y, avg_vx, avg_vy = calculate_averages(boid_index, neighbours, x_values, y_values, vx_values, vy_values)
    avoid_dx, avoid_dy = 0, 0

    for other_index in neighbours:
        if distance(boid_index, other_index, x_values, y_values) < min_distance:
            avoid_dx += x_values[boid_index] - x_values[other_index]
            avoid_dy += y_values[boid_index] - y_values[other_index]

    # Update velocities
    vx_values[boid_index] += (avg_x - x_values[boid_index]) * centering_factor \
        + (avg_vx - vx_values[boid_index]) * matching_factor \
        + avoid_dx * avoid_factor

    vy_values[boid_index] += (avg_y - y_values[boid_index]) * centering_factor \
        + (avg_vy - vy_values[boid_index]) * matching_factor \
        + avoid_dy * avoid_factor

    # Keep within bounds
    keep_within_bounds(boid_index, width, height, x_values, y_values, vx_values, vy_values)

    # Limit speed
    speed = sqrt(vx_values[boid_index]**2 + vy_values[boid_index]**2)
    if speed > speed_limit:
        speed_factor = speed_limit / speed
        vx_values[boid_index] *= speed_factor
        vy_values[boid_index] *= speed_factor

    # Update position based on velocity
    x_values[boid_index] += vx_values[boid_index]
    y_values[boid_index] += vy_values[boid_index]


def simulate_boids(num_boids, num_frames, width=1000, height=1000, visual_range=50, min_distance=30,
                   centering_factor=0.05, matching_factor=0.1, avoid_factor=0.01, speed_limit=15, render=False, output=False):
    print(f"Simulating {num_boids} Boids for {num_frames} frames")

    start_time = time.time()

    x_values, y_values, vx_values, vy_values, _ = create_boids(num_boids, width, height)
    boid_states = []

    for frame_num in range(num_frames):
        for i in range(num_boids):
            update_velocity_and_position(i, x_values, y_values, vx_values, vy_values, visual_range, min_distance,
                                         centering_factor, matching_factor, avoid_factor, speed_limit, width, height)
        if output:
            return boid_states
        
        frame_boids = [(x_values[i], y_values[i], vx_values[i], vy_values[i]) for i in range(num_boids)]
        boid_states.append(frame_boids)

    elapsed_time = time.time() - start_time
    print("SIMULATION COMPLETE")
    print(f"Time taken: {elapsed_time:.3f} seconds")

    if render:
        for frame_num, frame_boids in enumerate(boid_states):
            image = Image.new("RGB", (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(image)

            for boid in frame_boids:
                draw.ellipse(
                    (boid[0] - 2, boid[1] - 2, boid[0] + 2, boid[1] + 2),
                    fill=(0, 0, 0)
                )

            image.save(f"frame_{frame_num:03d}.png")

        print(f"{num_frames} frames have been saved.")
