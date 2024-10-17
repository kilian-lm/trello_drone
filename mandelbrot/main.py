from djitellopy import Tello
import time
import numpy as np

class MandelbrotPattern:
    def __init__(self, scale=0.5):
        self.tello = Tello()
        self.scale = scale  # Scale factor for the pattern size

    def generate_pattern(self):
        # Simplified representation of the Mandelbrot set
        # This will be a precomputed path suitable for the drone
        path = [
            (0, 0),
            (50, 0),
            (50, 50),
            (0, 50),
            (-50, 50),
            (-50, 0),
            (-50, -50),
            (0, -50),
            (50, -50),
            (50, 0),
        ]
        # Scale the path
        scaled_path = [(x * self.scale, y * self.scale) for x, y in path]
        return scaled_path

    def start(self):
        # Connect to the drone
        self.tello.connect()
        print(f"Battery Life Percentage: {self.tello.get_battery()}%")

        # Takeoff
        self.tello.takeoff()
        time.sleep(2)

        # Ascend to drawing height (e.g., pencil touching the ground)
        self.tello.move_down(20)  # Adjust based on the pencil length
        time.sleep(2)

        # Get the flight path
        path = self.generate_pattern()

        # Fly the pattern
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]

            # Move in X direction
            if dx > 0:
                self.tello.move_right(abs(int(dx)))
            else:
                self.tello.move_left(abs(int(dx)))
            time.sleep(2)

            # Move in Y direction
            if dy > 0:
                self.tello.move_forward(abs(int(dy)))
            else:
                self.tello.move_back(abs(int(dy)))
            time.sleep(2)

        # Ascend and land
        self.tello.move_up(20)  # Move back to safe height
        time.sleep(2)
        self.tello.land()

        # Disconnect
        self.tello.end()

if __name__ == "__main__":
    mandelbrot_pattern = MandelbrotPattern(scale=1)
    mandelbrot_pattern.start()