import math
import os
import numpy
import pygame
import moderngl
import glm
import model
from skybox import Skybox

#   Orbit:       MMB drag
#   Pan:         Shift + MMB drag
#   Dolly Zoom:  Ctrl + MMB drag (vertical motion) or Mouse Wheel scroll
#   Arrow Keys:  Pan forward/back/left/right (no modifier)
#                Move vertically up/down (Shift + Up/Down)
#                Zoom in/out (Ctrl + Up/Down)
# Elevation clamped to avoid flipping; zoom distance clamped.
class Engine:
    def __init__(self, width=800, height=600, title="LWGE Engine", control_panel=None):
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
        
        # Skybox
        self.skybox = Skybox(self.gl)
        
        # Model selection and manipulation
        self.selected_model = None
        self.transform_mode = None  # None, 'move', 'rotate', 'scale'
        self.transform_active = False  # True when in modal transform (after pressing G/R/S)
        self.transform_axis_constraint = None  # None, 'X', 'Y', 'Z'
        self.transform_start_mouse = None
        self.transform_start_value = None  # Store original transform
        self.control_panel = control_panel
        
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
        self.camera_pan_speed = 1.0  # units per arrow key press (increased from 0.5)
        # Sensitivities (tuned for a 800x600 window; scale independent math used for pan)
        self.orbit_sensitivity = 0.3   # deg per pixel
        self.pan_sensitivity = 1.5     # scalar applied to world-per-pixel pan (increased from 1.0)
        self.zoom_sensitivity = 0.08   # dolly factor per pixel
        self.min_camera_distance = 0.1  # decreased from 0.2 for closer views
        self.max_camera_distance = 5000.0  # increased from 1000 for larger scenes
        self.orbit_key_step = 3.0  # degrees per arrow key press when orbiting
        
        # Custom cursors for different modes
        self.cursors = {
            'normal': pygame.SYSTEM_CURSOR_ARROW,
            'move': pygame.SYSTEM_CURSOR_SIZEALL,
            'rotate': pygame.SYSTEM_CURSOR_CROSSHAIR,
            'scale': pygame.SYSTEM_CURSOR_SIZENWSE,
            'delete': pygame.SYSTEM_CURSOR_NO
        }
        self.current_cursor = 'normal'
        pygame.mouse.set_cursor(self.cursors['normal'])
        self.control_panel = control_panel
        
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
    
    def load_skybox_cubemap(self, image_paths):
        """Load a cubemap skybox from 6 images.
        
        Parameters:
        - image_paths: dict with keys 'right', 'left', 'top', 'bottom', 'front', 'back'
                      OR list of 6 paths in that order
        """
        try:
            self.skybox.load_cubemap(image_paths)
        except Exception as e:
            print(f"Error loading cubemap skybox: {e}")
    
    def load_skybox_equirectangular(self, image_path):
        """Load an equirectangular skybox from a single image.
        
        Parameters:
        - image_path: path to equirectangular image (.hdr, .jpg, .png, etc.)
        """
        try:
            self.skybox.load_equirectangular(image_path)
        except Exception as e:
            print(f"Error loading equirectangular skybox: {e}")
    
    def clear_skybox(self):
        """Remove current skybox"""
        self.skybox.clear()

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
                # Use camera position to center vector for accurate direction
                forward = glm.normalize(self.camera['center'] - self.camera['pos'])
                # Calculate right vector perpendicular to forward and up
                right = glm.normalize(glm.cross(forward, self.camera['up']))
                pan_amount = self.camera_pan_speed
                
                if key == pygame.K_LEFT:
                    # Pan left: move camera center to the left relative to view
                    self.camera['center'] -= right * pan_amount
                    self.update_camera_view()
                elif key == pygame.K_RIGHT:
                    # Pan right: move camera center to the right relative to view
                    self.camera['center'] += right * pan_amount
                    self.update_camera_view()
                elif key == pygame.K_UP:
                    # Move forward along the camera's view direction (projected to horizontal plane)
                    forward_horizontal = glm.vec3(forward.x, 0.0, forward.z)
                    if glm.length(forward_horizontal) > 0.001:  # Avoid division by zero
                        forward_horizontal = glm.normalize(forward_horizontal)
                        self.camera['center'] += forward_horizontal * pan_amount
                    self.update_camera_view()
                elif key == pygame.K_DOWN:
                    # Move backward along the camera's view direction (projected to horizontal plane)
                    forward_horizontal = glm.vec3(forward.x, 0.0, forward.z)
                    if glm.length(forward_horizontal) > 0.001:  # Avoid division by zero
                        forward_horizontal = glm.normalize(forward_horizontal)
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
    
    def duplicate_model(self, model_to_duplicate):
        """Duplicate a model instance with a slight offset.
        
        Creates a new Model instance with the same mesh data and textures,
        but with a new transform. Places it slightly offset from the original.
        """
        if model_to_duplicate is None:
            return None
        
        # Create new model from the same file path if available
        if hasattr(model_to_duplicate, 'file_path') and model_to_duplicate.file_path:
            new_model = model.Model.load_from_file(model_to_duplicate.file_path)
        else:
            # If no file path, we need to manually copy the data (more complex)
            # For now, return None - we'll need the file path
            print("Cannot duplicate model without file_path attribute")
            return None
        
        # Copy the transform from the original
        if hasattr(model_to_duplicate, 'model_matrix'):
            new_model.model_matrix = glm.mat4(model_to_duplicate.model_matrix)
        else:
            new_model.model_matrix = glm.mat4(1.0)
        
        # Offset the duplicate slightly (2 units to the right)
        offset = glm.vec3(2.0, 0.0, 0.0)
        new_model.model_matrix = glm.translate(new_model.model_matrix, offset)
        
        # Copy position, rotation, scale if they exist
        if hasattr(model_to_duplicate, 'position'):
            new_model.position = glm.vec3(model_to_duplicate.position) + offset
        if hasattr(model_to_duplicate, 'rotation'):
            new_model.rotation = glm.vec3(model_to_duplicate.rotation)
        if hasattr(model_to_duplicate, 'scale'):
            new_model.scale = glm.vec3(model_to_duplicate.scale)
        
        # Generate a unique name
        base_name = getattr(model_to_duplicate, 'name', 'Model')
        # Find existing duplicates to number properly
        counter = 1
        new_name = f"{base_name}.{counter:03d}"
        while any(getattr(m, 'name', None) == new_name for m in self.models):
            counter += 1
            new_name = f"{base_name}.{counter:03d}"
        new_model.name = new_name
        
        # Add to scene with same program
        program_to_use = getattr(model_to_duplicate, 'program', None)
        if program_to_use:
            new_model.create_gpu_resources(self.gl, program_to_use)
            new_model.program = program_to_use
        else:
            self.add_model(new_model)
            return new_model
        
        self.models.append(new_model)
        return new_model
    
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

        # Render skybox first (if present)
        if self.skybox and self.skybox.texture is not None:
            self.skybox.render(cam_persp, cam_view)

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
                # Bind texture samplers - now sampler can be a dict of multiple samplers
                try:
                    if sampler is not None:
                        # Check if sampler is a dict (new multi-texture format)
                        if isinstance(sampler, dict):
                            # Bind base color texture to unit 0
                            if 'baseColor' in sampler and sampler['baseColor'] is not None:
                                sampler['baseColor'].use(location=0)
                                prog['map'].value = 0
                            
                            # Bind normal map to unit 1
                            if 'normal' in sampler and sampler['normal'] is not None:
                                sampler['normal'].use(location=1)
                                prog['normalMap'].value = 1
                                try:
                                    prog['useNormalMap'].value = True
                                except Exception:
                                    pass
                            else:
                                try:
                                    prog['useNormalMap'].value = False
                                except Exception:
                                    pass
                            
                            # Bind roughness/metallic map to unit 2
                            if 'roughness' in sampler and sampler['roughness'] is not None:
                                sampler['roughness'].use(location=2)
                                prog['roughnessMap'].value = 2
                                try:
                                    prog['useRoughnessMap'].value = True
                                except Exception:
                                    pass
                            else:
                                try:
                                    prog['useRoughnessMap'].value = False
                                except Exception:
                                    pass
                        else:
                            # Legacy single sampler format (backwards compatibility)
                            sampler.use(location=0)
                            prog['map'].value = 0
                            try:
                                prog['useNormalMap'].value = False
                                prog['useRoughnessMap'].value = False
                            except Exception:
                                pass
                except Exception as e:
                    # Fallback: disable advanced features
                    try:
                        prog['useNormalMap'].value = False
                        prog['useRoughnessMap'].value = False
                    except Exception:
                        pass

                # set material uniforms if present (diffuse/specular/shininess)
                if material is not None:
                    try:
                        # Highlight selected model with a color tint
                        if m == self.selected_model:
                            # Add orange/yellow highlight tint to selected model
                            base_diffuse = material.get('diffuse', (1.0, 1.0, 1.0))
                            highlighted = (
                                min(1.0, base_diffuse[0] * 1.3 + 0.3),
                                min(1.0, base_diffuse[1] * 1.2 + 0.2),
                                min(1.0, base_diffuse[2] * 0.8)
                            )
                            prog['diff_reflectance'].value = highlighted
                        elif 'diffuse' in material:
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

    def ray_from_mouse(self, mouse_x, mouse_y):
        """Convert mouse coordinates to a ray in world space for picking"""
        if self.camera is None:
            return None, None
        
        # Convert screen coordinates to NDC (Normalized Device Coordinates)
        x = (2.0 * mouse_x) / self.width - 1.0
        y = 1.0 - (2.0 * mouse_y) / self.height
        
        # Ray in clip space
        ray_clip = glm.vec4(x, y, -1.0, 1.0)
        
        # Convert to eye space
        ray_eye = glm.inverse(self.camera['perspective']) * ray_clip
        ray_eye = glm.vec4(ray_eye.x, ray_eye.y, -1.0, 0.0)
        
        # Convert to world space
        ray_world = glm.vec3(glm.inverse(self.camera['view']) * ray_eye)
        ray_world = glm.normalize(ray_world)
        
        return self.camera['pos'], ray_world
    
    def ray_intersects_sphere(self, ray_origin, ray_dir, center, radius):
        """Check if ray intersects a bounding sphere"""
        oc = ray_origin - center
        a = glm.dot(ray_dir, ray_dir)
        b = 2.0 * glm.dot(oc, ray_dir)
        c = glm.dot(oc, oc) - radius * radius
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return False, float('inf')
        else:
            t = (-b - math.sqrt(discriminant)) / (2.0 * a)
            return True, t
    
    def pick_model(self, mouse_x, mouse_y):
        """Pick a model at the given mouse coordinates"""
        ray_origin, ray_dir = self.ray_from_mouse(mouse_x, mouse_y)
        
        if ray_origin is None or ray_dir is None:
            return None
        
        closest_model = None
        closest_distance = float('inf')
        
        for m in self.models:
            # Get model position (center of bounds or model matrix translation)
            model_mat = getattr(m, 'model_matrix', glm.mat4(1.0))
            model_pos = glm.vec3(model_mat[3])
            
            # Get bounding sphere radius
            bounds = getattr(m, 'bounds', None)
            if bounds:
                # bounds is a SceneBound object with boundingBox, center, and radius
                if hasattr(bounds, 'radius'):
                    radius = bounds.radius / 2.0
                elif hasattr(bounds, 'boundingBox'):
                    # Calculate from boundingBox
                    bbox = bounds.boundingBox
                    radius = glm.length(bbox[1] - bbox[0]) / 2.0
                else:
                    radius = 1.0
            else:
                radius = 1.0  # default radius
            
            # Check ray-sphere intersection
            hit, distance = self.ray_intersects_sphere(ray_origin, ray_dir, model_pos, radius)
            
            if hit and distance < closest_distance:
                closest_distance = distance
                closest_model = m
        
        return closest_model
    
    def delete_selected_model(self):
        """Delete the currently selected model"""
        if self.selected_model and self.selected_model in self.models:
            self.models.remove(self.selected_model)
            self.selected_model = None
            print("Model deleted")
            return True
        return False
    
    def start_transform(self, mode):
        """Start a modal transform operation (G/R/S keys)"""
        if self.selected_model is None:
            print("No model selected")
            return False
        
        self.transform_mode = mode
        self.transform_active = True
        self.transform_axis_constraint = None
        self.transform_start_mouse = pygame.mouse.get_pos()
        
        # Store the starting state
        if not hasattr(self.selected_model, 'model_matrix'):
            self.selected_model.model_matrix = glm.mat4(1.0)
        
        # Store original matrix
        self.transform_start_value = glm.mat4(self.selected_model.model_matrix)
        
        # Update cursor
        cursor = self.cursors.get(mode, self.cursors['normal'])
        pygame.mouse.set_cursor(cursor)
        
        print(f"Transform {mode} started - Move mouse, press X/Y/Z for axis, Enter to confirm, ESC to cancel")
        return True
    
    def set_axis_constraint(self, axis):
        """Constrain transform to specific axis (X/Y/Z)"""
        if self.transform_active:
            self.transform_axis_constraint = axis
            print(f"Transform constrained to {axis} axis")
    
    def cancel_transform(self):
        """Cancel the current transform and restore original state"""
        if self.transform_active and self.selected_model and self.transform_start_value:
            self.selected_model.model_matrix = glm.mat4(self.transform_start_value)
            print("Transform cancelled")
        
        self.transform_active = False
        self.transform_mode = None
        self.transform_axis_constraint = None
        self.transform_start_mouse = None
        self.transform_start_value = None
        pygame.mouse.set_cursor(self.cursors['normal'])
    
    def confirm_transform(self):
        """Confirm the current transform"""
        if self.transform_active:
            print(f"Transform {self.transform_mode} confirmed")
        
        self.transform_active = False
        self.transform_mode = None
        self.transform_axis_constraint = None
        self.transform_start_mouse = None
        self.transform_start_value = None
        pygame.mouse.set_cursor(self.cursors['normal'])
    
    def update_transform(self, mouse_x, mouse_y):
        """Update the transform based on mouse movement"""
        if not self.transform_active or not self.selected_model or not self.transform_start_mouse:
            return
        
        # Calculate mouse delta
        dx = (mouse_x - self.transform_start_mouse[0]) * 0.01
        dy = (mouse_y - self.transform_start_mouse[1]) * 0.01
        
        # Reset to original state
        self.selected_model.model_matrix = glm.mat4(self.transform_start_value)
        
        # Get camera vectors
        forward = glm.normalize(self.camera['center'] - self.camera['pos'])
        right = glm.normalize(glm.cross(forward, self.camera['up']))
        up = glm.vec3(0, 1, 0)  # World up
        
        if self.transform_mode == 'move':
            # Translation
            if self.transform_axis_constraint == 'X':
                # Move only along world X axis
                offset = glm.vec3(dx * 10, 0, 0)
            elif self.transform_axis_constraint == 'Y':
                # Move only along world Y axis
                offset = glm.vec3(0, -dy * 10, 0)
            elif self.transform_axis_constraint == 'Z':
                # Move only along world Z axis
                offset = glm.vec3(0, 0, dx * 10)
            else:
                # Free movement in camera plane
                offset = right * dx * 10 - up * dy * 10
            
            self.selected_model.model_matrix = glm.translate(self.selected_model.model_matrix, offset)
        
        elif self.transform_mode == 'rotate':
            # Rotation
            angle = dx * 100.0  # degrees
            
            if self.transform_axis_constraint == 'X':
                axis = glm.vec3(1, 0, 0)
            elif self.transform_axis_constraint == 'Y':
                axis = glm.vec3(0, 1, 0)
            elif self.transform_axis_constraint == 'Z':
                axis = glm.vec3(0, 0, 1)
            else:
                # Default to Y axis (like Blender)
                axis = glm.vec3(0, 1, 0)
            
            # Get model center
            model_pos = glm.vec3(self.selected_model.model_matrix[3])
            
            # Rotate around model's center
            self.selected_model.model_matrix = glm.translate(glm.mat4(1.0), model_pos)
            self.selected_model.model_matrix = glm.rotate(self.selected_model.model_matrix, glm.radians(angle), axis)
            self.selected_model.model_matrix = glm.translate(self.selected_model.model_matrix, -model_pos)
            self.selected_model.model_matrix = self.selected_model.model_matrix * self.transform_start_value
        
        elif self.transform_mode == 'scale':
            # Scaling
            scale_delta = 1.0 + dx * 5.0
            
            if scale_delta <= 0.01:  # Prevent negative/zero scale
                scale_delta = 0.01
            
            if self.transform_axis_constraint == 'X':
                scale_vec = glm.vec3(scale_delta, 1.0, 1.0)
            elif self.transform_axis_constraint == 'Y':
                scale_vec = glm.vec3(1.0, scale_delta, 1.0)
            elif self.transform_axis_constraint == 'Z':
                scale_vec = glm.vec3(1.0, 1.0, scale_delta)
            else:
                # Uniform scaling
                scale_vec = glm.vec3(scale_delta)
            
            self.selected_model.model_matrix = glm.scale(self.selected_model.model_matrix, scale_vec)
    
    def handle_mouse_click(self, mouse_x, mouse_y, button):
        """Handle mouse click for selection"""
        if button == 1:  # Left mouse button
            if self.transform_active:
                # If transform is active, left click confirms it
                self.confirm_transform()
            else:
                # Otherwise, try to select a model
                picked = self.pick_model(mouse_x, mouse_y)
                if picked:
                    self.selected_model = picked
                    print(f"Model selected")
                else:
                    self.selected_model = None
                    print("Deselected")
            return True
        return False
    
    def handle_key_press(self, key):
        """Handle keyboard input for transform operations"""
        # Check for Ctrl modifier
        mods = pygame.key.get_mods()
        ctrl_pressed = mods & pygame.KMOD_CTRL
        
        # Ctrl+D: Duplicate selected model
        if key == pygame.K_d and ctrl_pressed:
            if self.selected_model and not self.transform_active:
                duplicated = self.duplicate_model(self.selected_model)
                if duplicated:
                    self.selected_model = duplicated
                    print(f"Duplicated model: {duplicated.name}")
                    # Update UI if available
                    if self.control_panel:
                        self.control_panel.update_scene_list([
                            getattr(m, 'name', f'Model {i}') 
                            for i, m in enumerate(self.models)
                        ])
            return True
        
        # F2: Rename selected model
        elif key == pygame.K_F2:
            if self.selected_model and not self.transform_active:
                # Trigger rename in UI
                if self.control_panel:
                    self.control_panel.show_rename_dialog(self.selected_model)
            return True
        
        elif key == pygame.K_g:
            if not self.transform_active:
                self.start_transform('move')
            return True
        
        elif key == pygame.K_r:
            if not self.transform_active:
                self.start_transform('rotate')
            return True
        
        elif key == pygame.K_s:
            if not self.transform_active:
                self.start_transform('scale')
            return True
        
        elif key == pygame.K_x:
            if self.transform_active:
                if self.transform_axis_constraint == 'X':
                    # Delete if X pressed twice
                    self.cancel_transform()
                    self.delete_selected_model()
                else:
                    self.set_axis_constraint('X')
            else:
                # Delete model
                self.delete_selected_model()
            return True
        
        elif key == pygame.K_y:
            if self.transform_active:
                self.set_axis_constraint('Y')
            return True
        
        elif key == pygame.K_z:
            if self.transform_active:
                self.set_axis_constraint('Z')
            return True
        
        elif key == pygame.K_RETURN or key == pygame.K_KP_ENTER:
            if self.transform_active:
                self.confirm_transform()
            return True
        
        elif key == pygame.K_ESCAPE:
            if self.transform_active:
                self.cancel_transform()
            elif self.selected_model:
                self.selected_model = None
                print("Deselected")
            return True
        
        return False
