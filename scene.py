from Base.Cam import *
from Primitives.base import *

eps = 10**(-4)

class Scene:

    def __init__(self, objects, materials, depth_max):

        self.objects = objects
        self.materials = materials
        self.depth_max = depth_max

        self.background = [0, 0, 0]
        self.ambient_light = [0.1, 0.1, 0.1]

        self.camera = Camera(
            eye= [0, 0, 5],
            look_at= [0, 0, 0],
            up= [0, 1, 0],
            fov=45,
            img_width=800,
            img_height=600
        )

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
            
 
            valid_hit = current_hit.hit_mask.bool() 
            is_closer = valid_hit & (current_hit.t < best_t)
            
            best_t = torch.where(is_closer, current_hit.t, best_t)
            
        
            best_mask = torch.where(is_closer, torch.tensor(count, device=device), best_mask)

            is_closer_xyz = is_closer.unsqueeze(1)
            best_points = torch.where(is_closer_xyz, current_hit.point, best_points)
            best_normals = torch.where(is_closer_xyz, current_hit.normal, best_normals)


        hit_rec = HitRecord(best_mask, best_t, best_points, best_normals, self.materials)
        
        return hit_rec

