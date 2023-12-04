from libc.math cimport sqrt
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
import cython
from cython.parallel cimport prange
import time
from PIL import Image, ImageDraw

# cython: boundscheck=False

cdef int num_boids, width, height, visual_range,min_distance, speed_limit
cdef double[:] x_values, y_values, vx_values, vy_values
cdef double centering_factor, matching_factor, avoid_factor
cdef list neighbours, neighbour_list, boid_states
cdef Py_ssize_t i, boid_index, neighbour_index, other_boid, other_index
cdef bint render, output



num_boids = 1000
num_frames = 100
width = height = 1000

def simulate_boids(int num_boids=1000, int num_frames=100, int width=1000, int height=1000,
                    
                    int visual_range=50,int min_distance=30, double centering_factor=0.05, 
                    
                    double matching_factor=0.1, double avoid_factor=0.01, int speed_limit=15, 
                    
                    bint render=False):
    
    cdef int margin, turn_factor, counter, max_neighbours, local_counter
    cdef double[:] x_values, y_values, vx_values, vy_values, result
    cdef double  distance, sum_x, sum_y
    cdef double sum_vx, sum_vy, sum_x_final, sum_y_final, sum_vx_final, sum_vy_final, inv_count
    cdef double avoid_dx, avoid_dy
    cdef int[:, :] neighbour_list
    cdef list boid_states
    cdef Py_ssize_t i, n, boid, other_boid, boid1, boid2, neighbour, max_neighburs

    print(f"Simulating {num_boids} Boids for {num_frames} frames")
    
    start_time = time.time()

    #Init boids here
    

    x_values = np.random.uniform(0, width, num_boids)
    y_values = np.random.uniform(0, height, num_boids)
    vx_values = np.random.uniform(-5, 5, num_boids)
    vy_values = np.random.uniform(-5, 5, num_boids)
    
    boid_states = []
    
    max_neighbours = num_boids // 2
    
    for frame_num in range(num_frames):
        print(f"{frame_num} has finally been simulated!")
        neighbour_list = np.full((num_boids, 500), -1, dtype=int)  
        counter = 0
        for boid in prange(num_boids, nogil=True):
            local_counter = 0
            for other_boid in range(num_boids):
                distance = sqrt((x_values[other_boid] - x_values[boid])**2 + (y_values[other_boid] - y_values[boid])**2)
                if boid != other_boid and distance < visual_range:
                    neighbour_list[boid, counter] = other_boid
                    local_counter += 1
        with nogil:
            counter += local_counter
            
        for boid in range(num_boids):                
            n = len(neighbour_list[boid])
    
            if n == 0:
                avg_x = 0.0
                avg_y = 0.0
                avg_vx = 0.0
                avg_vy = 0.0
                
            else:
                sum_x = 0.0
                sum_y = 0.0
                sum_vx = 0.0
                sum_vy = 0.0
                
                for neighbour in neighbour_list[boid]:
                    sum_x += x_values[neighbour]
                    sum_y += y_values[neighbour]
                    sum_vx += vx_values[neighbour]
                    sum_vy += vy_values[neighbour]
    
                inv_count = 1.0 / n
                avg_x = sum_x * inv_count
                avg_y = sum_y * inv_count
                avg_vx = sum_vx * inv_count
                avg_vy = sum_vy * inv_count
            
            # update position and velocity here
            avoid_dx, avoid_dy = 0, 0
            
            neighbours = neighbour_list[boid]  # Fetch the correct neighbour list for each boid
            
            for other_boid in neighbours:
                if sqrt((x_values[other_boid] - x_values[boid])**2 + (y_values[other_boid] - y_values[boid])**2) < min_distance:  # Use the distance calculated in the inner loop
                    avoid_dx += x_values[boid] - x_values[other_boid]
                    avoid_dy += y_values[boid] - y_values[other_boid]
    
            # Update velocities
            vx_values[boid] += (avg_x - x_values[boid]) * centering_factor \
                            + (avg_vx - vx_values[boid]) * matching_factor \
                            + avoid_dx * avoid_factor

            vy_values[boid] += (avg_y - y_values[boid]) * centering_factor \
                            + (avg_vy - vy_values[boid]) * matching_factor \
                            + avoid_dy * avoid_factor
    
            # Keep within bounds
            margin = 100
            turn_factor = 3
    
            if x_values[boid] < margin:
                vx_values[boid] += turn_factor
                
            if x_values[boid] > width - margin:
                vx_values[boid] -= turn_factor
                
            if y_values[boid] < margin:
                vy_values[boid] += turn_factor
                
            if y_values[boid] > height - margin:
                vy_values[boid] -= turn_factor
    
            # Limit speed
            speed = sqrt(vx_values[boid]**2 + vy_values[boid]**2)
            if speed > speed_limit:
                speed_factor = speed_limit / speed
                vx_values[boid] *= speed_factor
                vy_values[boid] *= speed_factor
    
            # Update position based on velocity
            x_values[boid] += vx_values[boid]
            y_values[boid] += vy_values[boid]
        
        frame_boids = [(x_values[i], y_values[i], vx_values[i], vy_values[i]) for i in range(num_boids)]
        boid_states.append(frame_boids)
    
    elapsed_time = time.time() - start_time
    print("SIMULATION COMPLETE")
    print(f"Time taken: {elapsed_time:.3f} seconds")
    
    if render:

        for frame_num, boid_state in enumerate(boid_states):  # Use enumerate to get both frame_num and boid_state
            image = Image.new("RGB", (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(image)

            for x, y, _, _ in boid_state:  # Unpack the boid_state tuple
                draw.ellipse(
                    (x - 2, y - 2, x + 2, y + 2),  # Use the correct variables for coordinates
                    fill=(0, 0, 0)
                    )

            image.save(f"frame_{frame_num:03d}.png")

        print(f"{num_frames} frames have been saved.")
