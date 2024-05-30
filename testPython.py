import pydirectinput
import pygetwindow as gw
import time

# Define the movement keys for each direction
MOVEMENT_KEYS = {
    'forward': 'w',
    'backward': 's',
    'left': 'a',
    'right': 'd'
}

# Function to check if Unreal Editor window is active
def is_ue4_editor_active():
    active_window = gw.getActiveWindow()
    if active_window is not None and "DungeonProcedural (64-bit DebugGame PCD3D_SM6)" in active_window.title:
        return True
    return False

# Function to navigate the camera slowly in a specific direction
def navigate_camera(direction, duration):
    key = MOVEMENT_KEYS.get(direction)
    if key:
        print(f"Moving camera {direction}")
        pydirectinput.keyDown(key)  # Press the movement key
        time.sleep(duration)  # Wait for the specified duration
        pydirectinput.keyUp(key)  # Release the movement key

# Function to turn the camera left or right using mouse movement
def turn_camera(direction, duration):
    sensitivity = 200  # Adjust the sensitivity of the mouse movement
    if direction == 'left':
        pydirectinput.moveRel(-duration * sensitivity, 0, duration=0.1)
    elif direction == 'right':
        pydirectinput.moveRel(duration * sensitivity, 0, duration=0.1)

# Main function
def main():
    while True:
        if is_ue4_editor_active():
            # Slowly move the camera forward for 1 second
            navigate_camera('forward', 1)
            # Turn the camera left for 1 second
            turn_camera('left', 1)
        else:
            print("Unreal Editor window is not active.")
        time.sleep(0.1)  # Adjust the delay as needed

if __name__ == "__main__":
    main()
