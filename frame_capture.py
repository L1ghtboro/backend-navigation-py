import cv2
import os
import time
import pygetwindow as gw
import pyautogui

# Directory to save captured frames
capture_dir = 'captured_frames'
actions = ['forward', 'left', 'backward', 'right']

# Ensure directories exist
for action in actions:
    os.makedirs(os.path.join(capture_dir, action), exist_ok=True)

windowToCapture = "DungeonProcedural (64-bit DebugGame PCD3D_SM6)"

def capture_frame(action):
    ue4_window = gw.getWindowsWithTitle(windowToCapture)[0]
    left, top, right, bottom = ue4_window.left, ue4_window.top, ue4_window.right, ue4_window.bottom
    screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    timestamp = int(time.time() * 1000)
    filepath = os.path.join(capture_dir, action, f"{timestamp}.png")
    cv2.imwrite(filepath, screenshot)
    print(f"Captured frame for action '{action}' at {filepath}")

# Example capture loop (Capture frames for each action)
for action in actions:
    for _ in range(100):  # Capture 100 frames for each action
        capture_frame(action)
        time.sleep(0.1)  # Adjust the delay as needed
