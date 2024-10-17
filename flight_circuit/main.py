from djitellopy import Tello
import time

class BasicFlightCircuit:
    def __init__(self):
        self.tello = Tello()

    def start(self):
        # Connect to the drone
        self.tello.connect()
        print(f"Battery Life Percentage: {self.tello.get_battery()}%")

        # Takeoff
        self.tello.takeoff()
        time.sleep(2)

        # Ascend to a safe height (e.g., 1 meter)
        self.tello.move_up(100)
        time.sleep(2)

        # Perform a square circuit
        for _ in range(4):
            self.tello.move_forward(100)  # Move forward 1 meter
            time.sleep(2)
            self.tello.rotate_clockwise(90)  # Rotate 90 degrees
            time.sleep(2)

        # Descend and land
        self.tello.move_down(100)
        time.sleep(2)
        self.tello.land()

        # Disconnect
        self.tello.end()

if __name__ == "__main__":
    flight_circuit = BasicFlightCircuit()
    flight_circuit.start()