from Primitives.Triangle import Triangle
import numpy as np
from collections import defaultdict
import torch
from Primitives.base import *

class Triangle_Soup():

    def __init__(self, triangles,device):
    
        num_tris = len(triangles)

        
        # Convertemos os vértices de cada triângulo (que estão em NumPy) 
        # para Tensores do PyTorch apenas aqui no agrupamento.
        self.v0 = torch.stack([torch.as_tensor(t.list_vertex[0], device=device, dtype=torch.float32) for t in triangles])
        self.v1 = torch.stack([torch.as_tensor(t.list_vertex[1], device=device, dtype=torch.float32) for t in triangles])
        self.v2 = torch.stack([torch.as_tensor(t.list_vertex[2], device=device, dtype=torch.float32) for t in triangles])
        
        # Pré-calculamos as arestas (já em Tensores na GPU)
        self.e1 = self.v1 - self.v0
        self.e2 = self.v2 - self.v0
        self.device = device
    def print_triangles(self):
        for tri in self.triangles:
            print(tri)

    def hit(self, ray):
        # ray.ori e ray.dir são [N_raios, 3]
        # self.v0, e1, e2 são [M_tris, 3]
        
        N = ray.ori.shape[0]
        M = self.v0.shape[0]
        device = self.device

        # Expandimos para que cada raio encontre cada triângulo [N, M, 3]
        # CUIDADO: Se N*M for gigante, o chunking no Raster.py é obrigatório!
        ori = ray.ori.unsqueeze(1) # [N, 1, 3]
        direction = ray.dir.unsqueeze(1) # [N, 1, 3]
        
        # Möller-Trumbore Vetorizado
        pvec = torch.cross(direction, self.e2.unsqueeze(0), dim=-1) # [N, M, 3]
        det = torch.sum(self.e1.unsqueeze(0) * pvec, dim=-1) # [N, M]
        
        inv_det = 1.0 / (det + 1e-8)
        tvec = ori - self.v0.unsqueeze(0)
        u = torch.sum(tvec * pvec, dim=-1) * inv_det
        
        qvec = torch.cross(tvec, self.e1.unsqueeze(0), dim=-1)
        v = torch.sum(direction * qvec, dim=-1) * inv_det
        
        t = torch.sum(self.e2.unsqueeze(0) * qvec, dim=-1) * inv_det

        # Máscara de intersecção válida
        # t > 0, u entre 0 e 1, v entre 0 e 1, u+v < 1
        hit_mask = (det.abs() > 1e-8) & (u >= 0) & (v >= 0) & (u + v <= 1.0) & (t > 1e-4)

        # Para cada raio, pegamos o triângulo que teve o menor 't' positivo
        t[~hit_mask] = float('inf')
        best_t, best_tri_indices = torch.min(t, dim=1)
        
        final_mask = (best_t != float('inf')).long()
        
        # Calculamos os pontos e normais apenas dos triângulos vencedores
        # Normal = cross(e1, e2)
        normals_all = torch.cross(self.e1, self.e2, dim=-1)
        normals_all = normals_all / torch.linalg.norm(normals_all, dim=-1, keepdim=True)
        
        # Indexamos para pegar a normal do triângulo certo para cada raio
        # best_tri_indices tem tamanho (N,)
        best_normals = normals_all[best_tri_indices]
        best_points = ray.ori + ray.dir * best_t.unsqueeze(1)

        return HitRecord(final_mask, best_t, best_points, best_normals, None)

class Vertex():
    
    def __init__(self, idx = None, coord = None, normal = None):
        self.idx = idx
        self.normal = normal
        self.coord = coord
    
    def __str__(self):

        return f"Vertex {self.coord[0], self.coord[1], self.coord[2]}"

    

def ObjToTri(path, device):

    vertices = []
    faces = []

    # -------------------------
    # 1. LER OBJ
    # -------------------------
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "v":
                vertices.append(np.array(list(map(float, parts[1:4]))))

            elif parts[0] == "f":
                indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(indices)

    # -------------------------
    # 2. ACUMULAR NORMAIS
    # -------------------------
    vertex_normals = defaultdict(lambda: np.zeros(3))

    for f in faces:
        i1, i2, i3 = f

        v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]

        # normal NÃO normalizada (peso por área)
        n = np.cross(v2 - v1, v3 - v1)

        vertex_normals[i1] += n
        vertex_normals[i2] += n
        vertex_normals[i3] += n

    # -------------------------
    # 3. NORMALIZAR
    # -------------------------
    for k in vertex_normals:
        norm = np.linalg.norm(vertex_normals[k])
        if norm > 0:
            vertex_normals[k] /= norm

    # -------------------------
    # 4. CRIAR TRIÂNGULOS
    # -------------------------
    triangles = []

    for f in faces:
        i1, i2, i3 = f

        # Se o seu Triangle espera uma lista de tensores em .list_vertex:
        tri = Triangle(
            [v1, v2, v3], 
            vertex_normals=[
                vertex_normals[i1],
                vertex_normals[i2],
                vertex_normals[i3]
            ],
            device=device
        )

        triangles.append(tri)

    return triangles

