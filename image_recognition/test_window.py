from djitellopy import Tello
import time

tello = Tello()
tello.connect()  # Attempt the 'command' handshake
print(f"Battery: {tello.get_battery()}%")

# Optional: turn on video, see if it times out
tello.streamon()
time.sleep(2)
frame = tello.get_frame_read().frame
if frame is not None:
    print("Got a frame from Tello camera!")
tello.streamoff()
