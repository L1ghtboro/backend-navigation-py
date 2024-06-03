import time
import pygetwindow as gw
import pyautogui
import pydirectinput
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans

pyautogui.FAILSAFE = False

windowToNavigate = "DungeonProcedural (64-bit DebugGame PCD3D_SM6)"

def is_ue4_editor_active():
    active_window = gw.getActiveWindow()
    if active_window is not None and windowToNavigate in active_window.title:
        return True
    return False

def capture_ue4_editor_screenshot(compressed_resolution):
    ue4_window = gw.getWindowsWithTitle(windowToNavigate)[0]
    left, top, right, bottom = ue4_window.left, ue4_window.top, ue4_window.right, ue4_window.bottom
    screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    screenshot = cv2.resize(screenshot, (compressed_resolution[0], compressed_resolution[1]))  # Reduce size to speed up processing
    return screenshot

def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def extract_features(model, image):
    features = model.predict(image)
    return features

def dominant_color(image, k=3):
    pixels = image.reshape((image.shape[0] * image.shape[1], 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    counts = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
    dominant_color_rgb = dominant_color[::-1]  # Reverse the order of components (BGR to RGB)
    return dominant_color_rgb.astype(int)

def load_model():
    model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-1].output)
    return model

def find_navigation_cube_position(screenshot):
    hsv_image = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([130, 0, 220])
    upper_bound = np.array([170, 255, 255])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
    return None

def get_desktop_resolution():
    return pyautogui.size()

def move_to_find_cube(ue4_window, model, desktop_resolution, compressed_resolution):
    for _ in range(4):  # Try turning the camera 4 times in each direction
        screenshot = capture_ue4_editor_screenshot(compressed_resolution)
        position = find_navigation_cube_position(screenshot)
        if position:
            scaled_position = (position[0] * (desktop_resolution[0] / compressed_resolution[0]),
                               position[1] * (desktop_resolution[1] / compressed_resolution[1]))
            print(f"Pink Cube Position: {scaled_position}")
            navigate_to_position(scaled_position, ue4_window)
            return True  # Stop searching once the cube is found
        else:
            # Turn the camera in the opposite direction if the cube is not found
            turn_camera('right', duration=0.1)  # Adjust the direction and duration as needed
    return False  # Return False if the cube is not found

def navigate_to_position(position, ue4_window):
    ue4_window_left, ue4_window_top, ue4_window_width, ue4_window_height = ue4_window.box
    center_x = ue4_window_left + ue4_window_width // 2
    center_y = ue4_window_top + ue4_window_height // 2
    offset_x = position[0] - center_x
    offset_y = position[1] - center_y
    sensitivity = 200  # Adjust the sensitivity of the camera movement
    duration = 0.25  # Adjust the duration of the camera movement
    offset_x = int(offset_x / sensitivity)
    offset_y = int(offset_y / sensitivity)
    pydirectinput.moveRel(offset_x, offset_y, duration=duration)  # Move camera relative to the center

def turn_camera(direction, duration):
    sensitivity = 200  # Adjust the sensitivity of the camera movement
    offset = int(duration * sensitivity)
    if direction == 'left':
        pydirectinput.moveRel(-offset, 0, duration=0.1)
    elif direction == 'right':
        pydirectinput.moveRel(offset, 0, duration=0.1)

MOVEMENT_KEYS = {
    'forward': 'w',
    'backward': 's',
    'left': 'a',
    'right': 'd'
}

def navigate_camera(direction, duration):
    key = MOVEMENT_KEYS.get(direction)
    if key:
        print(f"Moving camera {direction}")
        pydirectinput.keyDown(key)  # Press the movement key
        time.sleep(duration)  # Wait for the specified duration
        pydirectinput.keyUp(key)  # Release the movement key

def main():
    compressed_resolution = [640, 360]
    desktop_resolution = get_desktop_resolution()
    model = load_model()
    search_directions = ['forward', 'left', 'backward', 'right']
    search_index = 0

    while True:
        if is_ue4_editor_active():
            start_time = time.time()
            ue4_windows = gw.getWindowsWithTitle(windowToNavigate)
            if ue4_windows:
                ue4_window = ue4_windows[0]
                screenshot = capture_ue4_editor_screenshot(compressed_resolution)
                position = find_navigation_cube_position(screenshot)
                if position:
                    scaled_position = (position[0] * (desktop_resolution[0] / compressed_resolution[0]),
                                       position[1] * (desktop_resolution[1] / compressed_resolution[1]))
                    print(f"Pink Cube Position: {scaled_position}")
                    navigate_to_position(scaled_position, ue4_window)
                    end_time = time.time()
                    time_taken = end_time - start_time
                    print(f"Time taken to find Pink Cube: {time_taken:.2f} seconds")
                else:
                    # If the pink cube is not in the frame, turn the camera and move to find it
                    if not move_to_find_cube(ue4_window, model, desktop_resolution, compressed_resolution):
                        # Move the player if the cube is still not found
                        navigate_camera(search_directions[search_index], duration=0.5)
                        search_index = (search_index + 1) % len(search_directions)
        time.sleep(0.1)

if __name__ == "__main__":
    main()
