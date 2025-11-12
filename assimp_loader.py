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
        self.tex_names = []
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
            # texture lookup: try TEXTURE_BASE then material TEXTURES (embedded)
            mat = self.scene.materials[mesh.material_index]
            tex_base = mat.get("TEXTURE_BASE")
            if tex_base:
                tex_path = (self.folder / tex_base).as_posix()
                self.tex_names.append(tex_path)
            else:
                # try embedded textures in material or scene
                textures = mat.get("TEXTURES")
                if textures and assimp_py.TextureType_DIFFUSE in textures:
                    tex_entry = textures[assimp_py.TextureType_DIFFUSE]
                    self.tex_names.append(("embedded", tex_entry))
                else:
                    self.tex_names.append(None)
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
        # tex_ref can be None, a filepath string, or ("embedded", data)
        if tex_ref is None:
            return None
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
            # external file path
            img_surf = pygame.image.load(tex_ref)
            img_data = pygame.image.tobytes(img_surf, "RGBA", True)
            tex = gl.texture(img_surf.get_size(), data=img_data, components=4)
            tex.build_mipmaps()
            return gl.sampler(texture=tex, filter=(gl.LINEAR_MIPMAP_LINEAR, gl.LINEAR), repeat_x=True, repeat_y=True)

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