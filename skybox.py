"""
Skybox rendering for LWGE engine.
Supports both cubemap (6 images) and equirectangular (single HDR/image) formats.
"""
import moderngl
import numpy as np
from PIL import Image
import os


class Skybox:
    """Skybox renderer with support for cubemap and equirectangular textures"""
    
    def __init__(self, gl: moderngl.Context):
        self.gl = gl
        self.program = None
        self.vao = None
        self.texture = None
        self.is_cubemap = False
        
        self._create_cube_geometry()
        self._compile_shaders()
    
    def _create_cube_geometry(self):
        """Create a cube VAO for skybox rendering"""
        # Cube vertices (just positions, no normals/UVs needed)
        # Use very large coordinates so it encompasses the scene
        vertices = np.array([
            # positions          
            -1.0,  1.0, -1.0,
            -1.0, -1.0, -1.0,
             1.0, -1.0, -1.0,
             1.0, -1.0, -1.0,
             1.0,  1.0, -1.0,
            -1.0,  1.0, -1.0,

            -1.0, -1.0,  1.0,
            -1.0, -1.0, -1.0,
            -1.0,  1.0, -1.0,
            -1.0,  1.0, -1.0,
            -1.0,  1.0,  1.0,
            -1.0, -1.0,  1.0,

             1.0, -1.0, -1.0,
             1.0, -1.0,  1.0,
             1.0,  1.0,  1.0,
             1.0,  1.0,  1.0,
             1.0,  1.0, -1.0,
             1.0, -1.0, -1.0,

            -1.0, -1.0,  1.0,
            -1.0,  1.0,  1.0,
             1.0,  1.0,  1.0,
             1.0,  1.0,  1.0,
             1.0, -1.0,  1.0,
            -1.0, -1.0,  1.0,

            -1.0,  1.0, -1.0,
             1.0,  1.0, -1.0,
             1.0,  1.0,  1.0,
             1.0,  1.0,  1.0,
            -1.0,  1.0,  1.0,
            -1.0,  1.0, -1.0,

            -1.0, -1.0, -1.0,
            -1.0, -1.0,  1.0,
             1.0, -1.0, -1.0,
             1.0, -1.0, -1.0,
            -1.0, -1.0,  1.0,
             1.0, -1.0,  1.0
        ], dtype='f4')
        
        self.vbo = self.gl.buffer(vertices.tobytes())
    
    def _compile_shaders(self):
        """Compile skybox shaders"""
        vertex_shader = """
        #version 330 core
        in vec3 position;
        
        out vec3 v_texcoord;
        
        uniform mat4 projection;
        uniform mat4 view;
        
        void main() {
            v_texcoord = position;
            // Remove translation from view matrix
            mat4 view_no_translation = mat4(mat3(view));
            vec4 pos = projection * view_no_translation * vec4(position, 1.0);
            // Trick: set z = w so depth is always 1.0 (far plane)
            gl_Position = pos.xyww;
        }
        """
        
        fragment_shader = """
        #version 330 core
        in vec3 v_texcoord;
        out vec4 fragColor;
        
        uniform samplerCube skybox;
        uniform bool useCubemap;
        uniform sampler2D equirectangular;
        
        // Convert direction vector to equirectangular UV coordinates
        vec2 directionToEquirectangular(vec3 dir) {
            vec2 uv;
            uv.x = atan(dir.z, dir.x) / (2.0 * 3.14159265) + 0.5;
            uv.y = asin(dir.y) / 3.14159265 + 0.5;
            return uv;
        }
        
        void main() {
            if (useCubemap) {
                fragColor = texture(skybox, v_texcoord);
            } else {
                vec2 uv = directionToEquirectangular(normalize(v_texcoord));
                fragColor = texture(equirectangular, uv);
            }
        }
        """
        
        self.program = self.gl.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        
        # Create VAO
        self.vao = self.gl.vertex_array(
            self.program,
            [(self.vbo, '3f', 'position')]
        )
    
    def load_cubemap(self, image_paths):
        """
        Load a cubemap from 6 images.
        
        Parameters:
        - image_paths: dict with keys 'right', 'left', 'top', 'bottom', 'front', 'back'
                      OR list of 6 paths in that order
        """
        # Convert list to dict if needed
        if isinstance(image_paths, list):
            if len(image_paths) != 6:
                raise ValueError("Cubemap requires exactly 6 images")
            keys = ['right', 'left', 'top', 'bottom', 'front', 'back']
            image_paths = dict(zip(keys, image_paths))
        
        # Expected order for OpenGL cubemap
        face_order = ['right', 'left', 'top', 'bottom', 'front', 'back']
        
        # Load all images
        face_images = []
        for face in face_order:
            if face not in image_paths:
                raise ValueError(f"Missing cubemap face: {face}")
            
            img_path = image_paths[face]
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            img = Image.open(img_path)
            img = img.convert('RGB')  # Ensure RGB format
            face_images.append(img)
        
        # All faces must be same size
        width, height = face_images[0].size
        if not all(img.size == (width, height) for img in face_images):
            raise ValueError("All cubemap faces must have the same dimensions")
        
        # Create cubemap texture
        self.texture = self.gl.texture_cube(
            (width, height),
            3,  # RGB
            None  # We'll write data per face
        )
        
        # Upload each face
        for i, img in enumerate(face_images):
            data = np.array(img, dtype='uint8').tobytes()
            self.texture.write(face=i, data=data)
        
        self.texture.build_mipmaps()
        self.is_cubemap = True
        print(f"Loaded cubemap skybox: {width}x{height}")
    
    def load_equirectangular(self, image_path):
        """
        Load an equirectangular HDR or image file.
        
        Parameters:
        - image_path: path to equirectangular image (.hdr, .jpg, .png, etc.)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        img = Image.open(image_path)
        img = img.convert('RGB')
        width, height = img.size
        
        # Create 2D texture
        data = np.array(img, dtype='uint8')
        self.texture = self.gl.texture(
            (width, height),
            3,  # RGB
            data.tobytes()
        )
        self.texture.build_mipmaps()
        self.is_cubemap = False
        print(f"Loaded equirectangular skybox: {width}x{height}")
    
    def render(self, projection_matrix, view_matrix):
        """
        Render the skybox.
        Call this AFTER clearing but BEFORE rendering scene objects.
        """
        if self.texture is None:
            return
        
        # Disable depth writing but keep depth test
        self.gl.depth_func = moderngl.LEQUAL
        
        # Set uniforms
        self.program['projection'].write(np.array(projection_matrix.to_list(), dtype='f4').tobytes())
        self.program['view'].write(np.array(view_matrix.to_list(), dtype='f4').tobytes())
        self.program['useCubemap'].value = self.is_cubemap
        
        # Bind texture
        if self.is_cubemap:
            self.texture.use(location=0)
            self.program['skybox'].value = 0
        else:
            self.texture.use(location=0)
            self.program['equirectangular'].value = 0
        
        # Render cube
        self.vao.render()
        
        # Restore depth function
        self.gl.depth_func = moderngl.LESS
    
    def clear(self):
        """Remove current skybox"""
        if self.texture:
            self.texture.release()
            self.texture = None
        self.is_cubemap = False
        print("Skybox cleared")
