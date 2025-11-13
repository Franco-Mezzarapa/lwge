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
        Now loads multiple texture types: base color, normal map, roughness/metallic.
        """
        self.vbos = []
        self.ibos = []
        self.vaos = []
        self.samplers = []  # Will store dict of samplers: {'baseColor': sampler, 'normal': sampler, ...}

        for i, (geom, idx) in enumerate(zip(self.geom_list, self.index_list)):
            # Ensure numpy arrays of correct dtype
            verts = np.array(geom, dtype='f4')
            indices = np.array(idx, dtype='i4')

            vbo = gl.buffer(verts.tobytes())
            ibo = gl.buffer(indices.tobytes())
            vao = gl.vertex_array(program, [(vbo, '3f 3f 2f', 'position', 'normal', 'uv')], ibo)

            # Load all texture types
            tex_ref = self.tex_refs[i] if i < len(self.tex_refs) else None
            samplers_dict = {}
            
            if self._loader is not None and tex_ref:
                # tex_ref is now a dict: {'baseColor': path, 'normal': path, etc.}
                if isinstance(tex_ref, dict):
                    for tex_type, tex_path in tex_ref.items():
                        try:
                            sampler = self._loader.load_texture(tex_path, gl)
                            if sampler:
                                samplers_dict[tex_type] = sampler
                        except Exception as e:
                            print(f"Warning: Failed to load {tex_type} texture: {e}")
                else:
                    # Fallback: old format (single texture)
                    sampler = self._loader.load_texture(tex_ref, gl)
                    if sampler:
                        samplers_dict['baseColor'] = sampler

            # Create default textures for missing types
            if 'baseColor' not in samplers_dict:
                gray = bytes([128, 128, 128, 255])  # 1x1 RGBA
                tex = gl.texture((1, 1), 4, gray)
                tex.build_mipmaps()
                samplers_dict['baseColor'] = gl.sampler(texture=tex, filter=(moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR),
                                     repeat_x=True, repeat_y=True)
            
            if 'normal' not in samplers_dict:
                # Default normal map (flat: 0.5, 0.5, 1.0 in RGB = up in tangent space)
                normal_default = bytes([128, 128, 255, 255])  # 1x1 RGBA
                tex = gl.texture((1, 1), 4, normal_default)
                tex.build_mipmaps()
                samplers_dict['normal'] = gl.sampler(texture=tex, filter=(moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR),
                                     repeat_x=True, repeat_y=True)
            
            if 'roughness' not in samplers_dict and 'metallic' not in samplers_dict:
                # Default roughness (0.5) and metallic (0.0) - combined
                rough_metal = bytes([128, 0, 0, 255])  # R=roughness, G=metallic
                tex = gl.texture((1, 1), 4, rough_metal)
                tex.build_mipmaps()
                samplers_dict['roughness'] = gl.sampler(texture=tex, filter=(moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR),
                                     repeat_x=True, repeat_y=True)

            self.vbos.append(vbo)
            self.ibos.append(ibo)
            self.vaos.append(vao)
            self.samplers.append(samplers_dict)  # Now a dict instead of single sampler

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
        model.file_path = filepath                  # Store for duplication
        
        # Extract filename as default name
        import os
        model.name = os.path.splitext(os.path.basename(filepath))[0]

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

        uniform sampler2D map;              // Base color / albedo
        uniform sampler2D normalMap;        // Normal map
        uniform sampler2D roughnessMap;     // Roughness (R channel) / Metallic (G channel)
        uniform vec4 light;
        uniform vec3 view_pos;
        uniform vec3 diff_reflectance;
        uniform vec3 specular_color;
        uniform float shininess;
        uniform bool useNormalMap;          // Toggle normal mapping
        uniform bool useRoughnessMap;       // Toggle roughness mapping

        // Simple normal mapping without tangent space (approximation)
        vec3 perturbNormal(vec3 N, vec3 V, vec2 texcoord) {
            // Sample normal map
            vec3 normalTex = texture(normalMap, texcoord).rgb * 2.0 - 1.0;
            
            // Derive tangent space from position derivatives
            vec3 dp1 = dFdx(v_fragpos);
            vec3 dp2 = dFdy(v_fragpos);
            vec2 duv1 = dFdx(texcoord);
            vec2 duv2 = dFdy(texcoord);
            
            vec3 dp2perp = cross(dp2, N);
            vec3 dp1perp = cross(N, dp1);
            vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
            vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
            
            float invmax = inversesqrt(max(dot(T,T), dot(B,B)));
            mat3 TBN = mat3(T * invmax, B * invmax, N);
            
            return normalize(TBN * normalTex);
        }

        void main() {
            vec3 normal = normalize(v_normal);
            vec3 view_dir = normalize(view_pos - v_fragpos);
            
            // Apply normal mapping if enabled
            if (useNormalMap) {
                normal = perturbNormal(normal, view_dir, v_texcoord);
            }

            // Compute light direction (point vs directional)
            vec3 light_dir;
            if (light.w == 1.0) {
                light_dir = normalize(light.xyz - v_fragpos);
            } else {
                light_dir = normalize(light.xyz);
            }

            // Albedo from base color texture
            vec4 albedo = texture(map, v_texcoord);
            vec3 mat_color = albedo.rgb;

            // Sample roughness/metallic if enabled
            float roughness = 0.5;
            float metallic = 0.0;
            if (useRoughnessMap) {
                vec4 roughMetalSample = texture(roughnessMap, v_texcoord);
                roughness = roughMetalSample.r;  // R channel = roughness
                metallic = roughMetalSample.g;   // G channel = metallic (if present)
            }
            
            // Convert roughness to shininess (inverse relationship)
            float adjustedShininess = shininess * (1.0 - roughness * 0.9);

            // Diffuse
            float diff = max(dot(normal, light_dir), 0.0);
            vec3 diffuse = diff * diff_reflectance * mat_color;

            // Specular (Blinn-Phong with roughness adjustment)
            vec3 half_vector = normalize(light_dir + view_dir);
            float spec = pow(max(dot(normal, half_vector), 0.0), adjustedShininess);
            vec3 specular = spec * specular_color * (1.0 - roughness * 0.7);

            // Ambient
            vec3 ambient = vec3(0.1) * mat_color;

            vec3 color = ambient + diffuse + specular;
            fragColor = vec4(color, albedo.a);
        }
        """
        return vertex_shader, fragment_shader