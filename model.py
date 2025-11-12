import numpy as np
import glm
import random
import moderngl
import assimp_loader as assimp

class Model:
    def __init__(self, meshes, materials, bounds):
        self.meshes = meshes
        self.materials = materials
        self.bounds = bounds

        # GPU resources
        self.vbos = []
        self.ibos = []
        self.vaos = []
        self.samplers = []

        # kept for texture loading / metadata
        self._loader = None
        self.geom_list = []
        self.index_list = []
        self.tex_refs = []

    def create_gpu_resources(self, gl: moderngl.Context, program):
        """
        Create VBO/IBO/VAO and upload textures. Assumes vertex layout: position(3), normal(3), uv(2).
        If a mesh has no texture reference, a 1x1 flat gray texture is created and used.
        """
        self.vbos = []
        self.ibos = []
        self.vaos = []
        self.samplers = []

        for i, (geom, idx) in enumerate(zip(self.geom_list, self.index_list)):
            # Ensure numpy arrays of correct dtype
            verts = np.array(geom, dtype='f4')
            indices = np.array(idx, dtype='i4')

            vbo = gl.buffer(verts.tobytes())
            ibo = gl.buffer(indices.tobytes())
            vao = gl.vertex_array(program, [(vbo, '3f 3f 2f', 'position', 'normal', 'uv')], ibo)

            # Texture / sampler
            tex_ref = self.tex_refs[i] if i < len(self.tex_refs) else None
            sampler = None
            if self._loader is not None:
                sampler = self._loader.load_texture(tex_ref, gl)

            # create flat gray texture if loader did not provide one
            if sampler is None:
                gray = bytes([128, 128, 128, 255])  # 1x1 RGBA
                tex = gl.texture((1, 1), 4, gray)
                tex.build_mipmaps()
                sampler = gl.sampler(texture=tex, filter=(moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR),
                                     repeat_x=True, repeat_y=True)

            self.vbos.append(vbo)
            self.ibos.append(ibo)
            self.vaos.append(vao)
            self.samplers.append(sampler)

    @staticmethod
    def load_from_file(filepath):
        """
        Load model using the Assimp loader. Ensures normals are generated,
        collects meshes, indices, texture references and bounding box information.
        Returns a Model instance with CPU-side data populated. Call create_gpu_resources()
        on the main thread with a moderngl.Context and shader program to create GPU resources.
        """
        loader = assimp.AssimpLoader(filepath, gen_normals=True, gen_uvs=False, calc_tangents=False, verbose=False)

        model = Model(meshes=[], materials=[], bounds=loader.bounds)
        model._loader = loader
        model.geom_list = loader.geom_list          # flattened float32 arrays (pos3, norm3, uv2)
        model.index_list = loader.index_list        # int32 index arrays
        model.tex_refs = loader.tex_names           # per-mesh texture reference (path, ("embedded", data), or None)

        # attempt to expose material list if available
        try:
            model.materials = getattr(loader.scene, "materials", []) or []
        except Exception:
            model.materials = []

        return model
    
    def compile_3D_shaders(self):

        vertex_shader = """
        #version 330 core
        in vec3 position;
        in vec3 normal;
        in vec2 uv;

        out vec2 v_texcoord;
        out vec3 v_normal;
        out vec3 v_fragpos;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 perspective;

        void main() {
            vec4 world_pos = model * vec4(position, 1.0);
            v_fragpos = world_pos.xyz;
            gl_Position = perspective * view * world_pos;
            v_texcoord = uv;
            v_normal = mat3(transpose(inverse(model))) * normal;
        }
        """

        fragment_shader = """
        #version 330 core

        in vec2 v_texcoord;
        in vec3 v_normal;
        in vec3 v_fragpos;

        out vec4 fragColor;

        uniform sampler2D map;
        uniform vec4 light;
        uniform vec3 view_pos;
        uniform vec3 diff_reflectance;
        uniform vec3 specular_color;
        uniform float shininess;

        void main() {
            vec3 normal = normalize(v_normal);

            // compute light direction (point vs directional)
            vec3 light_dir;
            if (light.w == 1.0) {
                light_dir = normalize(light.xyz - v_fragpos);
            } else {
                light_dir = normalize(light.xyz);
            }

            // albedo
            vec4 albedo = texture(map, v_texcoord);
            vec3 mat_color = albedo.rgb;

            // diffuse
            float diff = max(dot(normal, light_dir), 0.0);
            vec3 diffuse = diff * diff_reflectance * mat_color;

            // specular (Blinn-Phong)
            vec3 view_dir = normalize(view_pos - v_fragpos);
            vec3 half_vector = normalize(light_dir + view_dir);
            float spec = pow(max(dot(normal, half_vector), 0.0), shininess);
            vec3 specular = spec * specular_color;

            // ambient
            vec3 ambient = vec3(0.1) * mat_color;

            vec3 color = ambient + diffuse + specular;
            fragColor = vec4(color, albedo.a);
        }
        """
        return vertex_shader, fragment_shader