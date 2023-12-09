# cython: boundscheck=False
# cython: wraparound=False
# cython: overflowcheck=False
# cython: nonecheck=False
# cython: language_level=3

import numpy as np
cimport numpy as np
from cython.parallel cimport prange
cimport openmp
import os
from PIL import Image, ImageDraw

cdef int num_boids, width, height, speed_limit, min_distance
cdef double[:] x_values, y_values, vx_values, vy_values
cdef double centering_factor, matching_factor, avoid_factor
cdef list boid_states
cdef Py_ssize_t i, boid_index, neighbour_index, other_boid, other_index
cdef bint render, output


#os.chdir("C:/Users/samgl/Desktop/MSci_AdvComp_Boids/BoidsSimOMP")



# num_boids = 500
# num_frames = 100
# width = height = 1000

# cdef long[:, :] GenerateNeighbours(double[:] x_values, double[:] y_values, int num_boids):
#     cdef int neighbour
#     cdef int visual_range = 50
#     cdef long[:, :] neighbour_list
#     cdef Py_ssize_t boid_outer, boid_inner, i
#     cdef double x_vals, y_vals
#     cdef long[:] neighbours


#     # Cast x_values and y_values to NumPy arrays
#     cdef np.ndarray[np.float64_t, ndim=1] x_values_np = np.asarray(x_values)
#     cdef np.ndarray[np.float64_t, ndim=1] y_values_np = np.asarray(y_values)

#     # Calculate the squared distances between all pairs of boids using np methods
#     distances_squared = np.sum((np.stack([x_values_np, y_values_np], axis=1)[:, np.newaxis, :] - 
#                                   np.stack([x_values_np, y_values_np], axis=1)[np.newaxis, :, :]) ** 2, axis=-1)

#     # Identify neighbors based on the distance criterion
#     is_neighbour = (distances_squared > 0) & (distances_squared < visual_range**2)

#     # Initialize the neighbor list
#     neighbour_list = np.zeros((num_boids, num_boids + 1), dtype=int)

#     for boid_outer in range(num_boids):
#         # Extract indices of neighbors
#         neighbours = np.asarray(np.where(is_neighbour[boid_outer])[0])
        
#         # Set the count of nearest neighbors as the first element
#         neighbour_list[boid_outer, 0] = len(neighbours)

#         # Set the neighbor indices
#         for i, neighbour in enumerate(neighbours, start=1):
#             neighbour_list[boid_outer, i] = neighbour

#     return neighbour_list

cdef long[:,:] GenerateNeighbours(x_values, y_values, num_boids):
    cdef int visual_range = 50
    cdef int counter
    cdef long[:,:] neighbour_list
    cdef double x_vals, y_vals, distance
    cdef Py_ssize_t boid_outer, boid_inner
    cdef double[:] x_values_inner = x_values
    cdef double[:] y_values_inner = y_values
    
    neighbour_list = np.full((num_boids, num_boids + 1), 0, dtype=int)
    
    for boid_outer in range(num_boids):  # Use a different variable name for the outer loop
        x_vals = x_values[boid_outer]
        y_vals = y_values[boid_outer]
        counter = 1
        for boid_inner in range(num_boids):  # Use a different variable name for the inner loop
            distance = (x_values_inner[boid_inner] - x_vals)**2 + (y_values_inner[boid_inner] - y_vals)**2
            if boid_outer != boid_inner and distance < visual_range**2:
                neighbour_list[boid_outer, counter] = boid_inner
                counter += 1
        # Update the count of nearest neighbors
        neighbour_list[boid_outer, 0] = counter - 1
    return neighbour_list 

def simulate_boids(int num_boids, int num_frames, int threads, int width, int height,
                    
                    int visual_range,int min_distance, double centering_factor, 
                    
                    double matching_factor, double avoid_factor, int speed_limit, 
                    
                    bint render):
    
    cdef int margin, turn_factor, counter, n
    cdef double[:] x_values, y_values, vx_values, vy_values, result
    cdef double  distance, sum_x, sum_y, avg_x, avg_y, avg_vx, avg_vy
    cdef double sum_vx, sum_vy, inv_count, speed, speed_factor
    cdef double avoid_dx, avoid_dy
    cdef int[:] neighbours, current_neighbours
    cdef long[:, :] neighbour_list
    cdef list boid_states
    cdef Py_ssize_t i, boid, other_boid, boid1, boid2, neighbour, boid_outer, boid_inner

    print(f"Simulating {num_boids} Boids for {num_frames} frames")

    #Init boids here
    
    x_values = np.random.uniform(0, width, num_boids)
    y_values = np.random.uniform(0, height, num_boids)
    vx_values = np.random.uniform(-5, 5, num_boids)
    vy_values = np.random.uniform(-5, 5, num_boids)
    
    boid_states = []
    
    start_time = openmp.omp_get_wtime()
        
    for frame_num in range(num_frames):
        # Construct neighbour list here
        # Initialize outside the loop
        # ================================= make neighbours list =======================================
        
        neighbour_list = GenerateNeighbours(x_values, y_values, num_boids)
            # ================================== update boids =========================================
            
        for boid_outer in prange(num_boids, nogil=True, num_threads=threads):
            
            n = neighbour_list[boid_outer, 0]
            #current_neighbours = neighbour_list[boid_outer, 1 : 1 + n]
            
            sum_x = 0.0
            sum_y = 0.0
            sum_vx = 0.0
            sum_vy = 0.0
            
            if n == 0:
                avg_x = 0.0
                avg_y = 0.0
                avg_vx = 0.0
                avg_vy = 0.0
                
            else:
                for neighbour in neighbour_list[boid_outer, 1 : 1 + n]:
                    
                    sum_x = sum_x + x_values[neighbour]
                    sum_y = sum_y +  y_values[neighbour]
                    sum_vx = sum_vx + vx_values[neighbour]
                    sum_vy = sum_vy + vy_values[neighbour]
                    
        
                inv_count = 1.0 / n
                avg_x = sum_x * inv_count
                avg_y = sum_y * inv_count
                avg_vx = sum_vx * inv_count
                avg_vy = sum_vy * inv_count

            avoid_dx, avoid_dy = 0, 0

        
            #neighbours = neighbour_list[boid]  # Fetch the correct neighbour list for each boid

            for other_boid in neighbour_list[boid_outer, 1 : 1 + n]:
                if (x_values[other_boid] - x_values[boid_outer])**2 + (y_values[other_boid] - y_values[boid_outer])**2 < min_distance**2:  # Use the distance calculated in the inner loop
                    avoid_dx += x_values[boid_outer] - x_values[other_boid]
                    avoid_dy += y_values[boid_outer] - y_values[other_boid]

            # Update velocities
            vx_values[boid_outer] += (avg_x - x_values[boid_outer]) * centering_factor \
                + (avg_vx - vx_values[boid_outer]) * matching_factor \
                + avoid_dx * avoid_factor

            vy_values[boid_outer] += (avg_y - y_values[boid_outer]) * centering_factor \
                + (avg_vy - vy_values[boid_outer]) * matching_factor \
                + avoid_dy * avoid_factor

            # Keep within bounds
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

            # Limit speed
            speed = vx_values[boid_outer]**2 + vy_values[boid_outer]**2
            if speed > speed_limit**2:
                speed_factor = speed_limit / speed
                vx_values[boid_outer] *= speed_factor
                vy_values[boid_outer] *= speed_factor
            # Update position based on velocity
            x_values[boid_outer] += vx_values[boid_outer]
            y_values[boid_outer] += vy_values[boid_outer]

        frame_boids = [(x_values[i], y_values[i], vx_values[i], vy_values[i]) for i in range(num_boids)]
        boid_states.append(frame_boids)

    elapsed_time = openmp.omp_get_wtime() - start_time
    print("SIMULATION COMPLETE")
    print(f"Time taken: {elapsed_time:.3f} seconds")

    if render:
        for frame_num, boid_state in enumerate(boid_states):  # Use enumerate to get both frame_num and boid_state
            image = Image.new("RGB", (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(image)

            for x, y, _, _ in boid_state:  # Unpack the boid_state tuple
                draw.ellipse(
                    (x - 2, y - 2, x + 2, y + 2),  # Use the correct variables for coordinates
                    fill=(0, 0, 0))

            image.save(f"BigCanvas5kboids_{frame_num:03d}.png")

        print(f"{num_frames} frames have been saved.")
