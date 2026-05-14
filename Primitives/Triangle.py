import torch
import numpy as np
from Primitives.base import *


class Triangle(Obj):

    def __init__(
        self, list_vertex, vertex_normals=None, material_id=None, device="cpu"
    ):
        super().__init__(name="Triangle")

        if len(list_vertex) != 3:
            raise ValueError("Triangle precisa de 3 vértices")

        self.list_vertex = list_vertex
        self.vertex_normals = vertex_normals
        self.device = device

        # normal da face (fallback)
        cross = np.cross(
            list_vertex[0] - list_vertex[2], list_vertex[1] - list_vertex[2]
        )
        self.N = cross / np.linalg.norm(cross)

        self.material_id = material_id

    def hit(self, ray):
        O, D = ray.ori, ray.dir
        device = self.device

        v1 = torch.tensor(self.list_vertex[0], device=device).float()
        v2 = torch.tensor(self.list_vertex[1], device=device).float()
        v3 = torch.tensor(self.list_vertex[2], device=device).float()
        N_tensor = torch.tensor(self.N, device=device).float()

        # 1. Interseção Raio-Plano
        denominator = D @ N_tensor

        near_zero = torch.abs(denominator) < 1e-6

        numerator = (v1 - O) @ N_tensor
        T = numerator / (denominator + 1e-10)  # Estabilidade numérica

        # 2. Coordenadas Baricêntricas
        Points = O + T.unsqueeze(1) * D

        edge1 = v2 - v1
        edge2 = v3 - v1
        w = Points - v1

        # Usando torch.sum(... dim=1) para garantir compatibilidade com o batch de raios
        d00 = torch.dot(edge1, edge1)
        d01 = torch.dot(edge1, edge2)
        d11 = torch.dot(edge2, edge2)
        d20 = torch.sum(w * edge1, dim=1)
        d21 = torch.sum(w * edge2, dim=1)

        denom = d00 * d11 - d01 * d01
        u = (d11 * d20 - d01 * d21) / denom
        v = (d00 * d21 - d01 * d20) / denom
        w_bary = 1 - u - v

        hit_mask = (T > 0.001) & (u >= 0) & (v >= 0) & (u + v <= 1.0) & (~near_zero)

        if self.vertex_normals is not None:
            n1 = torch.tensor(self.vertex_normals[0], device=device).float()
            n2 = torch.tensor(self.vertex_normals[1], device=device).float()
            n3 = torch.tensor(self.vertex_normals[2], device=device).float()

            normals = (
                w_bary.unsqueeze(1) * n1 + u.unsqueeze(1) * n2 + v.unsqueeze(1) * n3
            )
            normals = normals / (
                torch.linalg.norm(normals, dim=1, keepdim=True) + 1e-10
            )
        else:
            normals = N_tensor.unsqueeze(0).expand(Points.shape[0], 3)
            print(hit_mask)
        return HitRecord(hit_mask.float(), T, Points, normals, None)
