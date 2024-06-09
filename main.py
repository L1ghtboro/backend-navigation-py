import time
import pygetwindow as gw
import pydirectinput
from algorithm import (
    is_ue4_editor_active,
    capture_ue4_editor_screenshot,
    find_navigation_cube_position,
    get_desktop_resolution,
    move_to_find_cube,
    navigate_to_position,
    navigate_camera,
    load_model_d,
    load_model_from_file
)

windowToNavigate = "DungeonProcedural (64-bit DebugGame PCD3D_SM6)"

def main():
    compressed_resolution = [640, 360]
    desktop_resolution = get_desktop_resolution()
    model = load_model_d()
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
                    if not move_to_find_cube(ue4_window, model, desktop_resolution, compressed_resolution):
                        navigate_camera(search_directions[search_index], duration=0.5)
                        search_index = (search_index + 1) % len(search_directions)
        time.sleep(0.01)

if __name__ == "__main__":
    main()
