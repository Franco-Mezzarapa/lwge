import os
import glm
import engine
import imgui
import pygame

def main():

    # Create engine with window and initialize default shader
    engine_instance = engine.Engine(width=800, height=600, title="LWGE Engine")
    
    # Setup camera
    engine_instance.setup_camera(
        eye=glm.vec3(0.0, 4.0, 10.0),
        center=glm.vec3(0.0, 0.0, 0.0),
        fov=45.0
    )

    # Add light
    light = glm.vec4(glm.normalize(glm.vec3(-0.6, -1.0, -0.3)), 0.0)
    engine_instance.add_light(light)

    # Load Mario model
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'mario_obj', 'scene.gltf')
    mario = engine_instance.add_model(model_path)

    imgui.create_context()
    imgui_renderer = pygame

    # Main loop
    is_running = True
    while is_running:
        engine_instance.dispatch_events()

        # 3D render
        engine_instance.clear((0.15, 0.2, 0.25, 1.0))
        engine_instance.render()


        # Swap
        engine_instance.swap_buffers()
        engine_instance.clock.tick(60)

if __name__ == "__main__":
    main()
