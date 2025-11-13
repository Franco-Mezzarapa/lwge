import os
import time
import glm
import engine
import pygame
import math
import threading
from ui import ControlPanel


def handle_camera_events(engine_instance, event):
    """Handle camera-related events (MMB drag, scroll wheel)"""
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
        engine_instance.camera_orbit_active = True
        engine_instance.last_mouse_pos = pygame.mouse.get_pos()
        mods = pygame.key.get_mods()
        if mods & pygame.KMOD_CTRL:
            engine_instance.camera_drag_mode = 'zoom'
        elif mods & pygame.KMOD_SHIFT:
            engine_instance.camera_drag_mode = 'pan'
        else:
            engine_instance.camera_drag_mode = 'orbit'
        return True
    
    elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
        engine_instance.camera_orbit_active = False
        engine_instance.last_mouse_pos = None
        engine_instance.camera_drag_mode = None
        return True
    
    elif event.type == pygame.MOUSEMOTION:
        if engine_instance.camera_orbit_active and engine_instance.last_mouse_pos:
            current_pos = pygame.mouse.get_pos()
            dx = current_pos[0] - engine_instance.last_mouse_pos[0]
            dy = current_pos[1] - engine_instance.last_mouse_pos[1]

            if engine_instance.camera_drag_mode == 'orbit':
                engine_instance.camera_azimuth += dx * engine_instance.orbit_sensitivity
                engine_instance.camera_elevation -= dy * engine_instance.orbit_sensitivity
                engine_instance.camera_elevation = max(-89.0, min(89.0, engine_instance.camera_elevation))
                engine_instance.update_camera_view()
            elif engine_instance.camera_drag_mode == 'pan':
                fov = engine_instance.camera.get('fov', 45.0)
                d = max(engine_instance.min_camera_distance, engine_instance.camera_distance)
                world_per_pixel = (2.0 * d * math.tan(math.radians(fov) * 0.5)) / max(1, engine_instance.height)
                scale = engine_instance.pan_sensitivity * world_per_pixel
                forward = glm.normalize(engine_instance.camera['center'] - engine_instance.camera['pos'])
                right = glm.normalize(glm.cross(forward, engine_instance.camera['up']))
                up = engine_instance.camera['up']
                engine_instance.camera['center'] -= right * (dx * scale)
                engine_instance.camera['center'] += up * (dy * scale)
                engine_instance.update_camera_view()
            elif engine_instance.camera_drag_mode == 'zoom':
                factor = math.exp(engine_instance.zoom_sensitivity * (dy / 100.0))
                engine_instance.camera_distance = max(
                    engine_instance.min_camera_distance,
                    min(engine_instance.max_camera_distance, engine_instance.camera_distance * factor)
                )
                engine_instance.update_camera_view()
            engine_instance.last_mouse_pos = current_pos
            return True
    
    elif event.type == pygame.MOUSEWHEEL:
        if engine_instance.camera is not None:
            factor = (1.0 - 0.15) if event.y > 0 else (1.0 + 0.15)
            engine_instance.camera_distance = max(
                engine_instance.min_camera_distance,
                min(engine_instance.max_camera_distance, engine_instance.camera_distance * factor)
            )
            engine_instance.update_camera_view()
        return True
    
    return False


def main():
    # Initialize UI
    control_panel = ControlPanel()
    control_panel.start()
    time.sleep(0.1)

    # Create engine
    engine_instance = engine.Engine(width=800, height=600, title="LWGE Engine", control_panel=control_panel)
    control_panel.engine = engine_instance

    # Setup camera
    engine_instance.setup_camera(
        eye=glm.vec3(0.0, 4.0, 10.0),
        center=glm.vec3(0.0, 0.0, 0.0),
        fov=45.0
    )

    # Add light
    light = glm.vec4(glm.normalize(glm.vec3(-0.6, -1.0, -0.3)), 0.0)
    engine_instance.add_light(light)

    # Main loop
    is_running = True
    while is_running:
        # Process UI commands
        try:
            while not control_panel.command_queue.empty():
                command, data = control_panel.command_queue.get_nowait()
                if command == 'delete_selected':
                    engine_instance.delete_selected_model()
                elif command == 'duplicate_selected':
                    if engine_instance.selected_model:
                        duplicated = engine_instance.duplicate_model(engine_instance.selected_model)
                        if duplicated:
                            engine_instance.selected_model = duplicated
                            control_panel.update_scene_list()
                elif command == 'rename_model':
                    model, new_name = data
                    model.name = new_name
                    control_panel.update_scene_list()
                    print(f"Renamed model to: {new_name}")
                elif command == 'enter_play_mode':
                    # Launch play mode in separate thread
                    def start_play_mode():
                        try:
                            from play_mode import PlayModeWindow
                            play_window = PlayModeWindow(engine_instance, width=1280, height=720)
                            play_window.start()
                        except Exception as e:
                            print(f"Error starting play mode: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    play_thread = threading.Thread(target=start_play_mode, daemon=False)
                    play_thread.start()
                    print("Starting play mode...")
                elif command == 'load_skybox_cubemap':
                    engine_instance.load_skybox_cubemap(data)
                elif command == 'load_skybox_equirect':
                    engine_instance.load_skybox_equirectangular(data)
                elif command == 'clear_skybox':
                    engine_instance.clear_skybox()
        except Exception as e:
            print(f"Error processing command: {e}")
        
        # Check if UI wants to load a new model
        try:
            while not control_panel.model_load_queue.empty():
                model_path = control_panel.model_load_queue.get_nowait()
                try:
                    print(f"Loading model: {model_path}")
                    engine_instance.add_model(model_path)
                    print(f"Model loaded successfully!")
                except Exception as e:
                    print(f"Error loading model: {e}")
        except Exception as e:
            print(f"Error checking model queue: {e}")
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
                continue
            
            # Let engine handle camera events
            if handle_camera_events(engine_instance, event):
                continue
            
            # Handle mouse clicks (selection, transform confirm)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                engine_instance.handle_mouse_click(mouse_pos[0], mouse_pos[1], event.button)
            
            # Handle mouse motion for active transforms
            elif event.type == pygame.MOUSEMOTION:
                if engine_instance.transform_active:
                    mouse_pos = pygame.mouse.get_pos()
                    engine_instance.update_transform(mouse_pos[0], mouse_pos[1])
            
            # Handle keyboard (G/R/S/X/Y/Z/Enter/ESC)
            elif event.type == pygame.KEYDOWN:
                # Let engine handle transform keys
                if not engine_instance.handle_key_press(event.key):
                    # If engine didn't handle it, try camera arrow keys
                    engine_instance.handle_keydown(event.key)
            
            # Handle window resize
            elif event.type == pygame.VIDEORESIZE:
                engine_instance.resize(event.w, event.h)
        
        # Render
        engine_instance.clear((0.15, 0.2, 0.25, 1.0))
        engine_instance.render()
        engine_instance.swap_buffers()
        engine_instance.clock.tick(60)
        
        # Check if UI was closed
        if control_panel.ui_started and not control_panel.running:
            is_running = False
    
    # Clean up
    pygame.quit()
    print("Program exited cleanly")


if __name__ == "__main__":
    main()
