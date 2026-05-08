import torch
import numpy as np
from Base.Cam import Camera
from Base.Material import SimpleMaterial
from Primitives.base import HitRecord
from Primitives.Triangle import Triangle 
from Base.Lights import PointLight
from ObjTotri import ObjToTri
from ObjTotri import Triangle_Soup


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Triangles_bunny = Triangle_Soup(ObjToTri("bunny.obj", device = DEVICE))

class PointLight:
    def __init__(self, pos, color, intensity):
        self.pos = pos
        self.color = color
        self.intensity = intensity


class Scene:
    def __init__(self, device=DEVICE, material_name=None, object_size=1.0, rotate_cam=0.0):
        
        self.depth_max = 3
        self.background = [0.1, 0.1, 0.15] 
        self.ambient_light = [0.2, 0.2, 0.2]


        self.camera = Camera(
            eye=torch.tensor([0, 0.5, 0.2], device=DEVICE),
            look_at=torch.tensor([0.0, 0.0, 0.0], device=DEVICE),
            up=torch.tensor([0.0, -1.0, 0.0], device=DEVICE),
            fov=45,
            img_width=800,
            img_height=600,
            device=DEVICE
        )


        self.lights = [
            PointLight(pos=[2.0, 3.0, 2.0], color=[1.0, 1.0, 1.0], intensity=1.0)
        ]


        material_triangulo = SimpleMaterial(
            ambient_coefficient=1.0,
            diffuse_coefficient=0.8,
            diffuse_color=[1.0, 0.2, 0.2], 
            specular_coefficient=0.8,
            specular_color=[1.0, 1.0, 1.0],
            specular_shininess=64,
            reflectivity=0.0
        )
        self.materials = [material_triangulo]*len(Triangles_bunny.triangles)

        self.objects = [Triangles_bunny.triangles[i] for i in range(len(Triangles_bunny.triangles))]



    def hit(self, ray):
        N = ray.ori.shape[0]
        device = ray.ori.device

        best_t = torch.full((N,), float('inf'), device=device)
        best_mask = torch.zeros((N,), dtype=torch.long, device=device) 
        best_points = torch.zeros((N, 3), device=device)
        best_normals = torch.zeros((N, 3), device=device)

        count = 0
        for shape, material in zip(self.objects, self.materials):
            count += 1 
            current_hit = shape.hit(ray)
            
            # Garante que os tensores estão no mesmo device que os raios
            hit_mask_bool = current_hit.hit_mask.bool().to(device)
            current_t = current_hit.t.to(device)
            
            is_closer = hit_mask_bool & (current_t < best_t)
            
            best_t = torch.where(is_closer, current_t, best_t)
            best_mask = torch.where(is_closer, torch.tensor(count, device=device), best_mask)
            
            is_closer_xyz = is_closer.unsqueeze(1)
            best_points = torch.where(is_closer_xyz, current_hit.point.to(device), best_points)
            best_normals = torch.where(is_closer_xyz, current_hit.normal.to(device), best_normals)

        # Importante: Como não passamos materials direto pro hit_rec (não é suportado em tensores),
        # deixamos None e a classe raster acessa `scene.materials` via o ID da máscara.
        return HitRecord(best_mask, best_t, best_points, best_normals, None)