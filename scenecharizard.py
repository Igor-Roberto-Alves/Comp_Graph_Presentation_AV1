import torch
import numpy as np
from Base.Cam import Camera
from Base.Material import SimpleMaterial
from Primitives.base import HitRecord
from Primitives.Triangle import Triangle
from Base.Lights import PointLight
from ObjTotri import ObjToTri
from ObjTotri import Triangle_Soup
from Primitives.Plane import Plane

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Triangles_bunny = Triangle_Soup(ObjToTri("Charizard.obj", device=DEVICE), device=DEVICE)


class PointLight:
    def __init__(self, pos, color, intensity):
        self.pos = pos
        self.color = color
        self.intensity = intensity


class Scene:
    def __init__(
        self, device=DEVICE, material_name=None, object_size=1.0, rotate_cam=0.0
    ):

        self.depth_max = 3
        self.background = [0, 0.714, 1]
        self.ambient_light = [0.2, 0.2, 0.2]

        self.camera = Camera(
            eye=torch.tensor([0, 10, 40], device=DEVICE),
            look_at=torch.tensor([0.0, 10.0, 0.0], device=DEVICE),
            up=torch.tensor([0.0, -1.0, 0.0], device=DEVICE),
            fov=45,
            img_width=100,
            img_height=75,
            device=DEVICE,
        )

        self.lights = [
            PointLight(pos=[15.0, 25.0, 15.0], color=[1.0, 0.9, 0.8], intensity=0.8),
            PointLight(pos=[-15.0, 10.0, 10.0], color=[0.5, 0.6, 0.8], intensity=0.8),
            PointLight(pos=[0.0, 20.0, -20.0], color=[1.0, 1.0, 1.0], intensity=0.5),
        ]

        material_triangulo = SimpleMaterial(
            ambient_coefficient=1.0,
            diffuse_coefficient=0.8,
            diffuse_color=[1.0, 0.2, 0.2],
            specular_coefficient=0.8,
            specular_color=[1.0, 1.0, 1.0],
            specular_shininess=64,
            reflectivity=0.0,
        )

        material_plane = SimpleMaterial(
            ambient_coefficient=0.2,
            diffuse_coefficient=0.9,
            diffuse_color=[0.8, 0.8, 0.8],
            specular_coefficient=0.2,
            specular_color=[1.0, 1.0, 1.0],
            specular_shininess=16,
            reflectivity=0.0,
        )

        plane = Plane(normal=[0.0, 1.0, 0.0], point=[0.0, 0.0, 0.0], device=DEVICE)

        self.objects = []
        for tri in Triangles_bunny.triangles:
            tri.material_id = 1
            self.objects.append(tri)
        self.objects.append(plane)
        plane.material_id = 2
        self.materials = [material_triangulo, material_plane]

    def hit(self, ray):
        N = ray.ori.shape[0]
        device = ray.ori.device

        # 0 será o fundo (vazio)
        best_t = torch.full((N,), float("inf"), device=device)
        best_mat_id = torch.zeros((N,), dtype=torch.long, device=device)
        best_points = torch.zeros((N, 3), device=device)
        best_normals = torch.zeros((N, 3), device=device)

        # Itera sobre os objetos
        for shape in self.objects:
            current_hit = shape.hit(ray)
            hit_mask_bool = current_hit.hit_mask.bool().to(device)
            current_t = current_hit.t.to(device)

            is_closer = hit_mask_bool & (current_t < best_t)

            best_t = torch.where(is_closer, current_t, best_t)

            # ATENÇÃO: Pegamos o material_id definido no objeto
            # Usamos +1 para que o ID 0 fique reservado para "nada"
            mat_id_tensor = torch.tensor(
                shape.material_id, device=device, dtype=torch.long
            )
            best_mat_id = torch.where(is_closer, mat_id_tensor, best_mat_id)

            is_closer_xyz = is_closer.unsqueeze(1)
            best_points = torch.where(
                is_closer_xyz, current_hit.point.to(device), best_points
            )
            best_normals = torch.where(
                is_closer_xyz, current_hit.normal.to(device), best_normals
            )

        return HitRecord(best_mat_id, best_t, best_points, best_normals, None)
