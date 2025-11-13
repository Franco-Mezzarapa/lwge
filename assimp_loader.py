import assimp_py
import numpy as np
import glm
from pathlib import Path
from PIL import Image
from io import BytesIO
import pygame

class SceneBound:
    def __init__(self, minmax):
        self.boundingBox = minmax
        self.center = (minmax[0] + minmax[1]) / 2
        self.radius = glm.length(minmax[1] - minmax[0])

class AssimpLoader:
    def __init__(self, filename, gen_normals=True, gen_uvs=False, calc_tangents=False, verbose=False):
        filepath = Path(filename)
        folder = filepath.parent
        flags = assimp_py.Process_Triangulate
        if gen_normals:
            flags |= assimp_py.Process_GenNormals
        if gen_uvs:
            flags |= assimp_py.Process_GenUVCoords
        if calc_tangents:
            flags |= assimp_py.Process_CalcTangentSpace

        self.scene = assimp_py.import_file(str(filepath), flags)
        self.folder = folder
        self._collect_mesh_data()

    def _collect_mesh_data(self):
        self.geom_list = []
        self.index_list = []
        self.tex_names = []  # Changed to store dict of texture types
        self._vertices_list = []
        
        for mesh in self.scene.meshes:
            verts = np.asarray(mesh.vertices, dtype='f4').reshape(-1,3)
            if mesh.normals is not None and len(mesh.normals) > 0:
                verts = np.concatenate((verts, np.asarray(mesh.normals, dtype='f4').reshape(-1,3)), axis=1)
            else:
                verts = np.concatenate((verts, np.zeros((len(verts),3), dtype='f4')), axis=1)
            if mesh.texcoords and len(mesh.texcoords) > 0:
                verts = np.concatenate((verts, np.asarray(mesh.texcoords[0], dtype='f4').reshape(-1,2)), axis=1)
            else:
                verts = np.concatenate((verts, np.zeros((len(verts),2), dtype='f4')), axis=1)
            
            self.geom_list.append(verts.flatten())
            self.index_list.append(np.array(mesh.indices).astype('i4'))
            
            # Get material and extract ALL texture types
            mat = self.scene.materials[mesh.material_index]
            textures_dict = {}
            
            # Try to get textures from different sources
            # 1. Direct properties (OBJ style)
            tex_base = mat.get("TEXTURE_BASE")
            if tex_base:
                textures_dict['baseColor'] = (self.folder / tex_base).as_posix()
            
            # 2. TEXTURES dictionary (GLTF/FBX style)
            textures = mat.get("TEXTURES")
            if textures:
                # Diffuse/Base Color
                if assimp_py.TextureType_DIFFUSE in textures:
                    tex_entry = textures[assimp_py.TextureType_DIFFUSE]
                    # Handle list (GLTF often returns lists)
                    if isinstance(tex_entry, list) and len(tex_entry) > 0:
                        tex_entry = tex_entry[0]
                    if isinstance(tex_entry, str):
                        textures_dict['baseColor'] = (self.folder / tex_entry).as_posix()
                    else:
                        textures_dict['baseColor'] = ("embedded", tex_entry)
                
                # Normal map
                if assimp_py.TextureType_NORMALS in textures:
                    tex_entry = textures[assimp_py.TextureType_NORMALS]
                    if isinstance(tex_entry, list) and len(tex_entry) > 0:
                        tex_entry = tex_entry[0]
                    if isinstance(tex_entry, str):
                        textures_dict['normal'] = (self.folder / tex_entry).as_posix()
                    else:
                        textures_dict['normal'] = ("embedded", tex_entry)
                
                # Height/Bump map (alternative to normal)
                if assimp_py.TextureType_HEIGHT in textures and 'normal' not in textures_dict:
                    tex_entry = textures[assimp_py.TextureType_HEIGHT]
                    if isinstance(tex_entry, list) and len(tex_entry) > 0:
                        tex_entry = tex_entry[0]
                    if isinstance(tex_entry, str):
                        textures_dict['normal'] = (self.folder / tex_entry).as_posix()
                    else:
                        textures_dict['normal'] = ("embedded", tex_entry)
                
                # Specular (can be used for metallic/roughness in some workflows)
                if assimp_py.TextureType_SPECULAR in textures:
                    tex_entry = textures[assimp_py.TextureType_SPECULAR]
                    if isinstance(tex_entry, list) and len(tex_entry) > 0:
                        tex_entry = tex_entry[0]
                    if isinstance(tex_entry, str):
                        textures_dict['specular'] = (self.folder / tex_entry).as_posix()
                    else:
                        textures_dict['specular'] = ("embedded", tex_entry)
                
                # Shininess (can contain roughness)
                if assimp_py.TextureType_SHININESS in textures:
                    tex_entry = textures[assimp_py.TextureType_SHININESS]
                    if isinstance(tex_entry, list) and len(tex_entry) > 0:
                        tex_entry = tex_entry[0]
                    if isinstance(tex_entry, str):
                        textures_dict['roughness'] = (self.folder / tex_entry).as_posix()
                    else:
                        textures_dict['roughness'] = ("embedded", tex_entry)
            
            # 3. Try to find textures by common naming patterns in the folder
            if not textures_dict.get('baseColor'):
                # Look for common base color names
                for pattern in ['baseColor', 'Base_Color', 'diffuse', 'Diffuse', 'albedo', 'Albedo', 'color', 'Color']:
                    possible_files = list(self.folder.glob(f'**/*{pattern}*.png'))
                    if not possible_files:
                        possible_files = list(self.folder.glob(f'**/*{pattern}*.jpg'))
                    if possible_files:
                        textures_dict['baseColor'] = str(possible_files[0])
                        print(f"Found base color texture: {possible_files[0].name}")
                        break
            
            if not textures_dict.get('normal'):
                # Look for common normal map names
                for pattern in ['normal', 'Normal', 'norm', 'Norm', 'nrm', 'NRM']:
                    possible_files = list(self.folder.glob(f'**/*{pattern}*.png'))
                    if not possible_files:
                        possible_files = list(self.folder.glob(f'**/*{pattern}*.jpg'))
                    if possible_files:
                        textures_dict['normal'] = str(possible_files[0])
                        print(f"Found normal map: {possible_files[0].name}")
                        break
            
            if not textures_dict.get('specular') and not textures_dict.get('roughness'):
                # Look for metallic/roughness (combined or separate)
                for pattern in ['metallicRoughness', 'MetallicRoughness', 'metallic', 'Metallic', 'roughness', 'Roughness']:
                    possible_files = list(self.folder.glob(f'**/*{pattern}*.png'))
                    if not possible_files:
                        possible_files = list(self.folder.glob(f'**/*{pattern}*.jpg'))
                    if possible_files:
                        if 'metallic' in pattern.lower():
                            textures_dict['metallic'] = str(possible_files[0])
                            print(f"Found metallic map: {possible_files[0].name}")
                        else:
                            textures_dict['roughness'] = str(possible_files[0])
                            print(f"Found roughness map: {possible_files[0].name}")
                        break
            
            self.tex_names.append(textures_dict if textures_dict else None)
            self._vertices_list.append(np.asarray(mesh.vertices,dtype='f4').reshape(-1,3))

        # bounding box
        self._compute_bounds()

    def _compute_bounds(self):
        minc = glm.vec3(np.inf)
        maxc = glm.vec3(-np.inf)
        for verts in self._vertices_list:
            for v in verts:
                vv = glm.vec3(float(v[0]), float(v[1]), float(v[2]))
                minc = glm.min(minc, vv)
                maxc = glm.max(maxc, vv)
        self.bounds = SceneBound([minc, maxc])

    def load_texture(self, tex_ref, gl):
        """Load a texture. tex_ref can be:
        - None: no texture
        - dict: multiple texture types {'baseColor': path, 'normal': path, etc.}
        - string: single texture path
        - tuple: ('embedded', data)
        
        Returns the base color/diffuse texture for backward compatibility.
        """
        if tex_ref is None:
            return None
        
        # If it's a dictionary, extract the base color texture
        if isinstance(tex_ref, dict):
            tex_ref = tex_ref.get('baseColor', None)
            if tex_ref is None:
                return None
        
        # Handle embedded texture
        if isinstance(tex_ref, tuple) and tex_ref[0] == "embedded":
            data = tex_ref[1]
            # data may be raw bytes or an object depending on assimp_py; try BytesIO -> PIL
            if isinstance(data, (bytes, bytearray)):
                img = Image.open(BytesIO(data)).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
            else:
                # fall back to treating it as a path-like or string content
                img = Image.open(BytesIO(bytes(data))).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
            tex = gl.texture(img.size, 4, img.tobytes())
            tex.build_mipmaps()
            return gl.sampler(texture=tex, filter=(gl.LINEAR_MIPMAP_LINEAR, gl.LINEAR), repeat_x=True, repeat_y=True)
        else:
            # External file path
            try:
                img_surf = pygame.image.load(tex_ref)
                img_data = pygame.image.tobytes(img_surf, "RGBA", True)
                tex = gl.texture(img_surf.get_size(), data=img_data, components=4)
                tex.build_mipmaps()
                return gl.sampler(texture=tex, filter=(gl.LINEAR_MIPMAP_LINEAR, gl.LINEAR), repeat_x=True, repeat_y=True)
            except Exception as e:
                print(f"Warning: Failed to load texture {tex_ref}: {e}")
                return None

    # minimal API to create GL renderables (similar to your existing code)
    def createRenderables(self, program, gl):
        self.renderables = []
        self.samplers = []
        for i, geom in enumerate(self.geom_list):
            vbo = gl.buffer(geom)
            ibo = gl.buffer(self.index_list[i])
            vao = gl.vertex_array(program, [(vbo, "3f 3f 2f", "position", "normal", "uv")], index_buffer=ibo, index_element_size=4)
            self.renderables.append(vao)
            sampler = self.load_texture(self.tex_names[i], gl)
            self.samplers.append(sampler)
        return self.renderables, self.samplers