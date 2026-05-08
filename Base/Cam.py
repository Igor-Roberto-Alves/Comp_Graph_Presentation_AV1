import torch
import math
from Primitives.base import *


class Camera:
    def __init__(self, eye, look_at, up, fov, img_width, img_height, device='cpu'):

        self.device = device
        self.img_width = img_width
        self.img_height = img_height

        # Convert all for torch tensors
        self.eye = torch.as_tensor(eye, device=device, dtype=torch.float32)
        self.look_at = torch.as_tensor(look_at, device=device, dtype=torch.float32)
        self.up = torch.as_tensor(up, device=device, dtype=torch.float32)
        print(f"Tipo do eye: {type(self.eye)}")
        print(f"Tipo do look_at: {type(self.look_at)}")
        
        aspect_ratio = img_height / img_width

        self.su = 2 * math.tan(math.radians(fov) / 2)
        self.sv = self.su * aspect_ratio
        self.fov = fov


        self.w = (self.eye - look_at)
        self.w = self.w / torch.linalg.norm(self.w)

        u_raw = torch.linalg.cross(up, self.w)
        self.u = u_raw / torch.linalg.norm(u_raw)

        v_raw = torch.linalg.cross(self.w, self.u)
        self.v = v_raw / torch.linalg.norm(v_raw)

    def generate_all_rays(self, randomize=False):

        """
        This function is the responsible for generate the Tensor off all rays that are casting throught the image
        The arg randomize can be pass for do a anti-aliasing effect
        """

        # Creating a Image
        j = torch.linspace(0, self.img_width - 1, self.img_width, device=self.device)
        i = torch.linspace(0, self.img_height - 1, self.img_height, device=self.device)
        

        grid_j, grid_i = torch.meshgrid(j, i, indexing='xy')

        # Determine the off-set
        if randomize:
            off_j = torch.rand((self.img_height, self.img_width), device=self.device)
            off_i = torch.rand((self.img_height, self.img_width), device=self.device)
        else:
            off_j = 0.5
            off_i = 0.5

        # Transforms
        x_ndc = self.su * (grid_j + off_j) / self.img_width - self.su / 2
        y_ndc = self.sv * (grid_i + off_i) / self.img_height - self.sv / 2


        x_flat = x_ndc.reshape(-1, 1)
        y_flat = y_ndc.reshape(-1, 1)

        direction_unorm = (self.u * x_flat) + (self.v * y_flat) - self.w
        
        # Normalazing direction vector
        direction = direction_unorm / torch.linalg.norm(direction_unorm, dim=1, keepdim=True)

        # Pinhole Cam
        origin = self.eye.expand(direction.shape[0], 3)

        # Return a Tensor of rays
        return Ray(origin, direction)

    def __str__(self):
        return f"Camera(eye={self.eye}, look_at={self.look_at}, up={self.up}, fov={self.fov}, img_width={self.img_width}, img_height={self.img_height})"
