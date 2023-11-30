import math
import random
import time
from scipy.spatial import KDTree
from PIL import Image, ImageDraw

class Boid:
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.neighbours = []

    def update_neighbours(self, neighbours):
        self.neighbours = neighbours

class BoidSimulation:
    def __init__(self, num_boids, num_frames, width=1000, height=1000):
        self.num_boids = num_boids
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.boids = []
        self.boid_kdtree = None
        self.visual_range = 50
        self.min_distance = 30
        self.centering_factor = 0.05
        self.matching_factor = 0.1
        self.avoid_factor = 0.01
        self.speed_limit = 15

    def init_boids(self):
        self.boids = [Boid(
            x=random.uniform(0, self.width),
            y=random.uniform(0, self.height),
            vx=random.uniform(-5, 5),
            vy=random.uniform(-5, 5)
        ) for _ in range(self.num_boids)]

        self.update_kdtree()

    def update_kdtree(self):
        self.boid_kdtree = KDTree([(boid.x, boid.y) for boid in self.boids])

    def find_neighbors(self, boid):
        neighbors_indices = self.boid_kdtree.query_ball_point((boid.x, boid.y), r=self.visual_range)
        boid.update_neighbours([self.boids[i] for i in neighbors_indices])

    def distance(self, boid1, boid2):
        return math.sqrt((boid1.x - boid2.x)**2 + (boid1.y - boid2.y)**2)

    def keep_within_bounds(self, boid):
        margin = 100
        turn_factor = 3

        if boid.x < margin:
            boid.vx += turn_factor
        if boid.x > self.width - margin:
            boid.vx -= turn_factor
        if boid.y < margin:
            boid.vy += turn_factor
        if boid.y > self.height - margin:
            boid.vy -= turn_factor

    def calculate_averages(self, neighbours):
        if not neighbours:
            return 0, 0, 0, 0

        sum_x, sum_y, sum_vx, sum_vy = 0, 0, 0, 0

        for neighbor in neighbours:
            sum_x += neighbor.x
            sum_y += neighbor.y
            sum_vx += neighbor.vx
            sum_vy += neighbor.vy

        count = len(neighbours)
        return sum_x / count, sum_y / count, sum_vx / count, sum_vy / count

    def update_velocity_and_position(self, boid):
        self.find_neighbors(boid)

        avg_x, avg_y, avg_vx, avg_vy = self.calculate_averages(boid.neighbours)

        avoid_dx, avoid_dy = 0, 0

        for other_boid in boid.neighbours:
            if self.distance(boid, other_boid) < self.min_distance:
                avoid_dx += boid.x - other_boid.x
                avoid_dy += boid.y - other_boid.y

        # Update velocities
        boid.vx += (avg_x - boid.x) * self.centering_factor \
            + (avg_vx - boid.vx) * self.matching_factor \
            + avoid_dx * self.avoid_factor

        boid.vy += (avg_y - boid.y) * self.centering_factor \
            + (avg_vy - boid.vy) * self.matching_factor \
            + avoid_dy * self.avoid_factor

        # Keep within bounds
        self.keep_within_bounds(boid)

        # Limit speed
        speed = math.sqrt(boid.vx**2 + boid.vy**2)
        if speed > self.speed_limit:
            speed_factor = self.speed_limit / speed
            boid.vx *= speed_factor
            boid.vy *= speed_factor

        # Update position based on velocity
        boid.x += boid.vx
        boid.y += boid.vy

    def simulate_boids(self, render=False):
        print(f"Simulating {self.num_boids} Boids for {self.num_frames} frames")

        start_time = time.time()

        self.init_boids()

        boid_states = []

        for frame_num in range(self.num_frames):
            self.update_kdtree()  # Update KDTree at each frame
            for boid in self.boids:
                self.update_velocity_and_position(boid)

            frame_boids = [(boid.x, boid.y, boid.vx, boid.vy) for boid in self.boids]
            boid_states.append(frame_boids)

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

# Example usage:
boid_simulation = BoidSimulation(num_boids=1000, num_frames=100)
boid_simulation.simulate_boids(render=True)
