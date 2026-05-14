import torch
from Primitives.base import *


class Plane:

    def __init__(self, normal, point, material_id=None, device="cpu"):

        self.normal = torch.tensor(normal, dtype=torch.float32, device=device)
        self.normal = self.normal / torch.linalg.norm(self.normal)

        self.point = torch.tensor(point, dtype=torch.float32, device=device)
        self.device = device
        self.material_id = material_id

    def hit(self, ray):

        O, D = ray.ori, ray.dir  # (N,3)

        N = self.normal  # (3,)
        P0 = self.point  # (3,)

        # denominador: D · N
        denominator = D @ N  # (N,)

        # numerador: (P0 - O) · N
        numerator = (P0 - O) @ N  # (N,)

        # evitar divisão por zero
        eps = 1e-8
        valid = torch.abs(denominator) > eps

        T = torch.where(
            valid,
            numerator / denominator,
            torch.tensor(float("nan"), device=self.device),
        )

        # pontos de interseção
        Points = O + T.unsqueeze(1) * D  # (N,3)

        # normal expandida
        batched_normals = N.unsqueeze(0).expand(O.shape[0], 3)

        # frente do plano (opcional, igual seu triangle)
        is_front_facing = denominator < 0

        # máscara final
        Hit_mask = (T > 0) & valid & is_front_facing

        return HitRecord(Hit_mask.float(), T, Points, batched_normals, None)
