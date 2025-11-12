import math
import os
import numpy
import pygame
import moderngl
import glm
import model

#   Orbit:       MMB drag
#   Pan:         Shift + MMB drag
#   Dolly Zoom:  Ctrl + MMB drag (vertical motion) or Mouse Wheel scroll
#   Arrow Keys:  Pan forward/back/left/right (no modifier)
#                Move vertically up/down (Shift + Up/Down)
#                Zoom in/out (Ctrl + Up/Down)
# Elevation clamped to avoid flipping; zoom distance clamped.
class Engine:
    def __init__(self, width=800, height=600, title="LWGE Engine"):
        pygame.init()
        self.width = width
        self.height = height
        self.window_size = (width, height)
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(self.window_size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        pygame.display.set_caption(title)

        self.gl = moderngl.create_context()
        self.gl.viewport = (0, 0, width, height)
        self.gl.enable(moderngl.DEPTH_TEST)

        # scene models
        self.models = []
        self.default_program = None
        
        # camera and lighting
        self.camera = None
        self.lights = []
        
        # Camera interaction state
        self.camera_orbit_active = False
        self.last_mouse_pos = None
        # Blender-style drag mode: 'orbit' (MMB), 'pan' (Shift+MMB), 'zoom' (Ctrl+MMB)
        self.camera_drag_mode = None
        self.camera_azimuth = 0.0  # horizontal rotation angle
        self.camera_elevation = 20.0  # vertical rotation angle (degrees)
        self.camera_distance = 10.0  # distance from center
        self.camera_pan_speed = 0.5  # units per arrow key press
        # Sensitivities (tuned for a 800x600 window; scale independent math used for pan)
        self.orbit_sensitivity = 0.3   # deg per pixel
        self.pan_sensitivity = 1.0     # scalar applied to world-per-pixel pan
        self.zoom_sensitivity = 0.08   # dolly factor per pixel
        self.min_camera_distance = 0.2
        self.max_camera_distance = 1000.0
        self.orbit_key_step = 3.0  # degrees per arrow key press when orbiting
        
        # Initialize default shader program
        self._initialize_default_shader()

    def clear(self, color=(0.1, 0.1, 0.1, 1.0)):
        self.gl.clear(*color)

    def swap_buffers(self):
        pygame.display.flip()
    
    def _initialize_default_shader(self):
        """Compile and configure the default shader program for 3D models.
        
        This method creates a basic Blinn-Phong shader that supports:
        - Per-vertex positions, normals, and texture coordinates
        - Model/view/perspective transforms
        - Point and directional lighting
        - Diffuse texture sampling
        - Ambient, diffuse, and specular shading
        """
        # Create a temporary model instance to get shader source
        temp_model = model.Model(meshes=[], materials=[], bounds=None)
        vertex_src, fragment_src = temp_model.compile_3D_shaders()
        
        # Compile the shader program
        self.default_program = self.gl.program(vertex_shader=vertex_src, fragment_shader=fragment_src)
        
        # Set common uniform defaults
        try:
            self.default_program["map"].value = 0
            self.default_program["diff_reflectance"].value = (1.0, 1.0, 1.0)
            self.default_program["specular_color"].value = (0.04, 0.04, 0.04)
            self.default_program["shininess"].value = 32.0
        except KeyError:
            # Shader may not have all these uniforms
            pass
    
    def setup_camera(self, eye=None, center=None, up=None, fov=45.0, near=0.1, far=100.0):
        """Configure the camera for the scene.
        
        Parameters:
        - eye: camera position (glm.vec3). Defaults to (0, 4, 10)
        - center: look-at target (glm.vec3). Defaults to origin
        - up: up vector (glm.vec3). Defaults to (0, 1, 0)
        - fov: field of view in degrees
        - near: near clipping plane distance
        - far: far clipping plane distance
        
        Returns the camera dict with view, perspective, and pos.
        """
        if eye is None:
            eye = glm.vec3(0.0, 4.0, 10.0)
        if center is None:
            center = glm.vec3(0.0, 0.0, 0.0)
        if up is None:
            up = glm.vec3(0.0, 1.0, 0.0)
        
        # Store camera parameters for interactive updates
        self.camera_distance = glm.length(eye - center)
        direction = glm.normalize(eye - center)
        self.camera_azimuth = math.degrees(math.atan2(direction.z, direction.x))
        self.camera_elevation = math.degrees(math.asin(direction.y))
        
        view = glm.lookAt(eye, center, up)
        aspect = self.width / self.height
        perspective = glm.perspective(glm.radians(fov), aspect, near, far)
        
        self.camera = {
            'view': view,
            'perspective': perspective,
            'pos': eye,
            'center': center,
            'up': up,
            'fov': fov,
            'near': near,
            'far': far
        }
        return self.camera
    
    def update_camera_view(self):
        """Recompute camera view matrix from current orbit parameters."""
        if self.camera is None:
            return
        
        # Convert spherical coordinates to Cartesian
        azimuth_rad = math.radians(self.camera_azimuth)
        elevation_rad = math.radians(self.camera_elevation)
        
        eye_x = self.camera_distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        eye_y = self.camera_distance * math.sin(elevation_rad)
        eye_z = self.camera_distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        
        eye = self.camera['center'] + glm.vec3(eye_x, eye_y, eye_z)
        view = glm.lookAt(eye, self.camera['center'], self.camera['up'])
        
        self.camera['view'] = view
        self.camera['pos'] = eye
    
    def add_light(self, light):
        """Add a light to the scene.
        
        Parameters:
        - light: glm.vec4 where xyz is direction/position and w indicates type:
                 w=0 for directional light (xyz is direction)
                 w=1 for point light (xyz is position)
        """
        self.lights.append(light)
    
    def clear_lights(self):
        """Remove all lights from the scene."""
        self.lights = []

    # General pygame resize handler.
    def resize(self, width, height):
        self.width = width
        self.height = height
        self.gl.viewport = (0, 0, width, height)
        self.window_size = (width, height)
        self.aspect_ratio = width / height
        # keep camera perspective aspect in sync
        if self.camera is not None:
            fov = self.camera.get('fov', 45.0)
            near = self.camera.get('near', 0.1)
            far = self.camera.get('far', 100.0)
            self.camera['perspective'] = glm.perspective(glm.radians(fov), self.aspect_ratio, near, far)

    # Handles engine dispatching.
    def dispatch_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            elif event.type == pygame.VIDEORESIZE:
                self.resize(event.w, event.h)

            elif event.type == pygame.KEYDOWN:
                self.handle_keydown(event.key)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 2:  # Middle mouse button
                    self.camera_orbit_active = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_CTRL:
                        self.camera_drag_mode = 'zoom'
                    elif mods & pygame.KMOD_SHIFT:
                        self.camera_drag_mode = 'pan'
                    else:
                        self.camera_drag_mode = 'orbit'
            
            elif event.type == pygame.MOUSEWHEEL:
                # Scroll wheel to zoom in/out (dolly)
                if self.camera is not None:
                    # exponential zoom for smoothness
                    factor = (1.0 - 0.15) if event.y > 0 else (1.0 + 0.15)
                    self.camera_distance = max(self.min_camera_distance, min(self.max_camera_distance, self.camera_distance * factor))
                    self.update_camera_view()
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    self.camera_orbit_active = False
                    self.last_mouse_pos = None
                    self.camera_drag_mode = None
            
            elif event.type == pygame.MOUSEMOTION:
                if self.camera_orbit_active and self.last_mouse_pos is not None and self.camera is not None:
                    current_pos = pygame.mouse.get_pos()
                    dx = current_pos[0] - self.last_mouse_pos[0]
                    dy = current_pos[1] - self.last_mouse_pos[1]

                    if self.camera_drag_mode == 'orbit':
                        # Update azimuth (horizontal rotation) and elevation (vertical rotation)
                        self.camera_azimuth += dx * self.orbit_sensitivity
                        self.camera_elevation -= dy * self.orbit_sensitivity
                        # Clamp elevation to avoid flipping
                        self.camera_elevation = max(-89.0, min(89.0, self.camera_elevation))
                        self.update_camera_view()

                    elif self.camera_drag_mode == 'pan':
                        # Pan amount in world units per pixel at the current distance
                        # world per pixel at target plane ~ 2 * d * tan(fov/2) / height
                        fov = self.camera.get('fov', 45.0)
                        d = max(self.min_camera_distance, self.camera_distance)
                        world_per_pixel = (2.0 * d * math.tan(math.radians(fov) * 0.5)) / max(1, self.height)
                        scale = self.pan_sensitivity * world_per_pixel
                        forward = glm.normalize(self.camera['center'] - self.camera['pos'])
                        right = glm.normalize(glm.cross(forward, self.camera['up']))
                        up = self.camera['up']
                        # Drag to the right should move the scene with the mouse (camera left), hence center -= right*dx
                        self.camera['center'] -= right * (dx * scale)
                        # Dragging down should move the scene down (camera up), hence center += up*dy
                        self.camera['center'] += up * (dy * scale)
                        self.update_camera_view()

                    elif self.camera_drag_mode == 'zoom':
                        # Dolly based on vertical mouse motion (dy). Positive dy (drag down) zooms out.
                        factor = math.exp(self.zoom_sensitivity * (dy / 100.0))
                        self.camera_distance = max(self.min_camera_distance, min(self.max_camera_distance, self.camera_distance * factor))
                        self.update_camera_view()

                    self.last_mouse_pos = current_pos

            

    # Handles Keydown events.
    def handle_keydown(self, key):
        if key == pygame.K_ESCAPE:
            pygame.quit()
            exit()
        
        # Arrow keys: pan camera (no modifier), vertical move (Shift), zoom (Ctrl)
        if self.camera is not None:
            # Check if Shift is held down
            mods = pygame.key.get_mods()
            shift_held = mods & pygame.KMOD_SHIFT
            ctrl_held = mods & pygame.KMOD_CTRL
            
            if shift_held:
                # Shift + Up/Down: move camera center vertically along world Y-axis
                pan_amount = self.camera_pan_speed
                world_up = glm.vec3(0.0, 1.0, 0.0)  # Always use absolute world up
                if key == pygame.K_UP:
                    self.camera['center'] += world_up * pan_amount
                    self.update_camera_view()
                    return
                elif key == pygame.K_DOWN:
                    self.camera['center'] -= world_up * pan_amount
                    self.update_camera_view()
                    return
            elif ctrl_held:
                # Ctrl + Up/Down: zoom in/out
                zoom_speed = 0.5
                if key == pygame.K_UP:
                    self.camera_distance = max(self.min_camera_distance, self.camera_distance - zoom_speed)
                    self.update_camera_view()
                    return
                elif key == pygame.K_DOWN:
                    self.camera_distance = min(self.max_camera_distance, self.camera_distance + zoom_speed)
                    self.update_camera_view()
                    return
            else:
                # No modifiers: Pan camera center (forward/back/left/right relative to view)
                forward = glm.normalize(self.camera['center'] - self.camera['pos'])
                right = glm.normalize(glm.cross(forward, self.camera['up']))
                pan_amount = self.camera_pan_speed
                
                if key == pygame.K_LEFT:
                    self.camera['center'] -= right * pan_amount
                    self.update_camera_view()
                elif key == pygame.K_RIGHT:
                    self.camera['center'] += right * pan_amount
                    self.update_camera_view()
                elif key == pygame.K_UP:
                    # Move forward along the camera's view direction (projected to horizontal plane)
                    forward_horizontal = glm.normalize(glm.vec3(forward.x, 0.0, forward.z))
                    self.camera['center'] += forward_horizontal * pan_amount
                    self.update_camera_view()
                elif key == pygame.K_DOWN:
                    # Move backward along the camera's view direction (projected to horizontal plane)
                    forward_horizontal = glm.normalize(glm.vec3(forward.x, 0.0, forward.z))
                    self.camera['center'] -= forward_horizontal * pan_amount
                    self.update_camera_view()


    def add_model(self, model_input, program=None):
        """Add a Model instance to the engine. If a filepath is given it is loaded.

        Parameters
        - model_input: either a `Model` instance or a string path to a model file
        - program: optional compiled `moderngl.Program` to use for this model. If not provided
                   the model's built-in `compile_3D_shaders()` is used to create a program.

        Returns the Model instance that was added to the scene.
        """
        # Resolve input: accept either a Model instance or a filesystem path
        if isinstance(model_input, str):
            loaded_model = model.Model.load_from_file(model_input)
        else:
            loaded_model = model_input

        # If user supplied a program, use it. Otherwise prefer the engine default program
        # if present, else compile the model's built-in shaders.
        if program is not None:
            program_to_use = program
        elif getattr(self, 'default_program', None) is not None:
            program_to_use = self.default_program
        else:
            vertex_src, fragment_src = loaded_model.compile_3D_shaders()
            program_to_use = self.gl.program(vertex_shader=vertex_src, fragment_shader=fragment_src)

        # Try to set common uniforms with safe fallbacks
        try:
            program_to_use["map"].value = 0
        except Exception:
            pass
        try:
            program_to_use["diff_reflectance"].value = (1.0, 1.0, 1.0)
            program_to_use["specular_color"].value = (0.04, 0.04, 0.04)
            program_to_use["shininess"].value = 32.0
        except Exception:
            pass

        # Create GPU resources with the selected program (must run on main GL thread)
        loaded_model.create_gpu_resources(self.gl, program_to_use)

        # Remember which program the model uses so render() can prefer it
        loaded_model.program = program_to_use

        # Track the model in the scene and return the Model instance
        self.models.append(loaded_model)
        return loaded_model


    def remove_model(self, model):
        """Remove a Model instance from the engine."""
        try:
            self.models.remove(model)
        except ValueError:
            pass
    
    def render(self, camera=None, lights=None, shadow_map=None):
        """
        Render the scene using the provided camera and lights.

        Parameters
        - camera: optional camera dict. If None, uses engine's camera (set via setup_camera)
        - lights: optional list of light descriptors. If None, uses engine's lights
        - shadow_map: optional moderngl texture (depth) bound by caller if using shadows.

        This method updates uniforms for each model's shader program (or the engine default program)
        so the main loop can remain uncluttered. It also binds per-mesh samplers and issues draw calls.
        """
        # Use engine camera/lights if not provided
        if camera is None:
            camera = self.camera
        if lights is None:
            lights = self.lights
        
        # Early exit if no camera configured
        if camera is None:
            return
        # Normalize camera input
        cam_view = camera.get('view') if isinstance(camera, dict) else getattr(camera, 'view', None)
        cam_persp = camera.get('perspective') if isinstance(camera, dict) else getattr(camera, 'perspective', None)
        cam_pos = camera.get('pos') if isinstance(camera, dict) else getattr(camera, 'pos', None)

        # Precompute bytes for view/perspective to avoid realloc each draw
        view_bytes = numpy.array(cam_view.to_list(), dtype='f4').tobytes() if cam_view is not None else None
        persp_bytes = numpy.array(cam_persp.to_list(), dtype='f4').tobytes() if cam_persp is not None else None

        # Choose a single light to set as uniform (caller can extend to multi-light later)
        active_light = None
        if isinstance(lights, list) and len(lights) > 0:
            active_light = lights[0]
        elif lights is not None:
            active_light = lights

        # Iterate models and render
        for m in self.models:
            # Determine program to use: model may have been created with a program stored, otherwise use engine default
            prog = getattr(m, 'program', None) or self.default_program
            if prog is None:
                # nothing to draw without a shader program
                continue

            # Update common camera uniforms if they exist in the program
            try:
                if view_bytes is not None:
                    prog['view'].write(view_bytes)
                if persp_bytes is not None:
                    prog['perspective'].write(persp_bytes)
            except Exception:
                # program may not have those uniforms; ignore
                pass

            # view position uniform (for specular) if present
            if cam_pos is not None:
                try:
                    prog['view_pos'].value = tuple(cam_pos)
                except Exception:
                    pass

            # light uniform if provided
            if active_light is not None:
                try:
                    # support either dict or glm/tuple as light
                    if isinstance(active_light, dict) and 'vec' in active_light:
                        prog['light'].value = tuple(active_light['vec'])
                    else:
                        prog['light'].value = tuple(active_light)
                except Exception:
                    pass

            # Optional shadow map binding
            if shadow_map is not None:
                try:
                    prog['shadow_map'].value = 1
                    shadow_map.use(location=1)
                except Exception:
                    pass

            # Per-model transform: use model.model_matrix if present else identity
            model_mat = getattr(m, 'model_matrix', glm.mat4(1.0))
            try:
                prog['model'].write(numpy.array(model_mat.to_list(), dtype='f4').tobytes())
            except Exception:
                pass

            # Now draw each mesh in the model
            # Each model.meshes corresponds to a VAO/sampler/material set created in create_gpu_resources
            for vao, sampler, material in zip(m.vaos, m.samplers, getattr(m, 'materials', [])):
                # bind texture sampler to unit 0 (shader should expect map at unit 0)
                try:
                    if sampler is not None:
                        sampler.use(location=0)
                    else:
                        # if no sampler, shader should still sample a bound texture unit; skip if none
                        pass
                except Exception:
                    pass

                # set material uniforms if present (diffuse/specular/shininess)
                if material is not None:
                    try:
                        if 'diffuse' in material:
                            prog['diff_reflectance'].value = material['diffuse']
                        if 'shininess' in material:
                            prog['shininess'].value = material['shininess']
                        if 'specular' in material:
                            prog['specular_color'].value = material['specular']
                    except Exception:
                        pass

                # Finally render; support instanced draws if model exposes instance_count
                instances = getattr(m, 'instance_count', None)
                try:
                    if instances is not None and instances > 1:
                        vao.render(instances=instances)
                    else:
                        vao.render()
                except Exception:
                    # drawing failed for this vao; continue with next
                    continue
