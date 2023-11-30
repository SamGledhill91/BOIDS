# boids_simulation.pyx
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw
import time

# Define the Boid class
cdef class Boid:
    cdef public float x, y, vx, vy
    cdef public list neighbours

    def __cinit__(self, float x, float y, float vx, float vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.neighbours = []

    cdef void update_neighbours(self, list neighbours):
        self.neighbours = neighbours


# Define the BoidSimulation class
cdef class BoidSimulation:
    cdef public int num_boids, num_frames, width, height, visual_range, min_distance, speed_limit
    cdef public float centering_factor, matching_factor, avoid_factor
    cdef public list boids
    cdef object boid_kdtree  # Use 'object' type for the kdtree

    def __cinit__(self, int num_boids, int num_frames, int width=1000, int height=1000):
        self.num_boids = num_boids
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.boids = []
        self.visual_range = 50
        self.min_distance = 30
        self.centering_factor = 0.05
        self.matching_factor = 0.1
        self.avoid_factor = 0.01
        self.speed_limit = 15

    cdef void init_boids(self):
        self.boids = [Boid(
            x=np.random.uniform(0, self.width),
            y=np.random.uniform(0, self.height),
            vx=np.random.uniform(-5, 5),
            vy=np.random.uniform(-5, 5)
        ) for _ in range(self.num_boids)]

        self.update_kdtree()

    cdef void update_kdtree(self):
        self.boid_kdtree = cKDTree([(boid.x, boid.y) for boid in self.boids])

    cdef void find_neighbors(self, Boid boid):
        neighbors_indices = self.boid_kdtree.query_ball_point((boid.x, boid.y), r=self.visual_range)
        boid.update_neighbours([self.boids[i] for i in neighbors_indices])

    cdef float distance(self, Boid boid1, Boid boid2):
        return np.sqrt((boid1.x - boid2.x)**2 + (boid1.y - boid2.y)**2)

    cdef void keep_within_bounds(self, Boid boid):
        cdef float margin = 100
        cdef float turn_factor = 3

        if boid.x < margin:
            boid.vx += turn_factor
        if boid.x > self.width - margin:
            boid.vx -= turn_factor
        if boid.y < margin:
            boid.vy += turn_factor
        if boid.y > self.height - margin:
            boid.vy -= turn_factor

    cdef void calculate_averages(self, list neighbours, float[4] result):
        if not neighbours:
            result[0] = result[1] = result[2] = result[3] = 0
            return

        cdef float sum_x, sum_y, sum_vx, sum_vy
        sum_x = sum_y = sum_vx = sum_vy = 0

        for neighbor in neighbours:
            sum_x += neighbor.x
            sum_y += neighbor.y
            sum_vx += neighbor.vx
            sum_vy += neighbor.vy

        cdef int count = len(neighbours)
        result[0] = sum_x / count
        result[1] = sum_y / count
        result[2] = sum_vx / count
        result[3] = sum_vy / count

    cdef void update_velocity_and_position(self, Boid boid):
        self.find_neighbors(boid)

        cdef float[4] avg_result
        self.calculate_averages(boid.neighbours, avg_result)

        cdef float avoid_dx, avoid_dy
        avoid_dx = avoid_dy = 0

        for other_boid in boid.neighbours:
            if self.distance(boid, other_boid) < self.min_distance:
                avoid_dx += boid.x - other_boid.x
                avoid_dy += boid.y - other_boid.y

        # Update velocities
        boid.vx += (avg_result[0] - boid.x) * self.centering_factor \
            + (avg_result[2] - boid.vx) * self.matching_factor \
            + avoid_dx * self.avoid_factor

        boid.vy += (avg_result[1] - boid.y) * self.centering_factor \
            + (avg_result[3] - boid.vy) * self.matching_factor \
            + avoid_dy * self.avoid_factor

        # Keep within bounds
        self.keep_within_bounds(boid)

        # Limit speed
        cdef float speed = np.sqrt(boid.vx**2 + boid.vy**2)
        if speed > self.speed_limit:
            boid.vx = (boid.vx / speed) * self.speed_limit
            boid.vy = (boid.vy / speed) * self.speed_limit

        # Update position based on velocity
        boid.x += boid.vx
        boid.y += boid.vy

    cpdef simulate_boids(self, bint render):
        print(f"Simulating {self.num_boids} Boids for {self.num_frames} frames")
    
        # Use a list to store arrays for each frame
        cdef list boid_states_list = []
        
        cdef float[:, ::1] frame_boids
    
        cdef int frame_num
    
        start_time = time.time()
    
        self.init_boids()
    
        for frame_num in range(self.num_frames):
            self.update_kdtree()  # Update KDTree at each frame
    
            # Create an array for the current frame
            frame_boids = np.zeros((self.num_boids, 4), dtype=np.float32)
    
            for i, boid in enumerate(self.boids):
                self.update_velocity_and_position(boid)
    
                # Update the array for the current boid
                frame_boids[i, 0] = boid.x
                frame_boids[i, 1] = boid.y
                frame_boids[i, 2] = boid.vx
                frame_boids[i, 3] = boid.vy
    
            # Append the array for the current frame to the list
            boid_states_list.append(frame_boids)
    
        # Convert the list to a NumPy array
        cdef float[:, :, ::1] boid_states = np.array(boid_states_list)
    
        elapsed_time = time.time() - start_time
        print("SIMULATION COMPLETE")
        print(f"Time taken: {elapsed_time:.3f} seconds")
    
        if render:
            for frame_num, frame_boids in enumerate(boid_states):
                image = Image.new("RGB", (self.width, self.height), (255, 255, 255))
                draw = ImageDraw.Draw(image)
    
                for boid in frame_boids:
                    draw.ellipse(
                        (boid[0] - 2, boid[1] - 2, boid[0] + 2, boid[1] + 2),
                        fill=(0, 0, 0)
                    )
    
                image.save(f"frame_{frame_num:03d}.png")
    
            print(f"{self.num_frames} frames have been saved.")
