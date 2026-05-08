import torch

class Light:
    def __init__(self):
        pass

    def position(self):
        raise NotImplementedError("Subclasses should implement this method")
    

class PointLight:
    def __init__(self, position, color, intensity=1.0, device="cpu"):

        self.pos = torch.tensor(position, device=device).float()
        
     
        self.color = torch.tensor([color[0], color[1], color[2]], device=device).float()
            
        self.intensity = intensity

    def position(self):
        return self.pos