"""
In this file we are going to define the basic module for objects, scenes, and hit management
(IRA)
"""

from torch import tensor

class Ray:
    def __init__(self, origin:tensor, direction:tensor):
        
        """
        Origin is a tensor (N, 3) where N indicate how many rays are throw in the same time for the window
        Direction too
        """

        self.ori = origin
        self.dir = direction

    def point_at(self, k: tensor):
        
        # K is a tensor (N, 1)
        return self.ori + k * self.dir


        

class Obj:
    def __init__(self, name = None): 
        self.name = name # Name is a formality, but not necessarily
    
    def hit(self, ray):
        raise NotImplementedError("For this Object there isn't a hit logic implemented")
    

class HitRecord:
    def __init__(self, hit_mask = None, t = None, point = None, normal = None, materials = None):
        self.hit_mask = hit_mask # (N,)
        self.t = t               # (N,)
        self.point = point       # (N, 3)
        self.normal = normal     # (N, 3)
        self.materials = materials       # (N, 3)
