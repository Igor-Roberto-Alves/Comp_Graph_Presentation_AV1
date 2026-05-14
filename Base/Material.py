import torch
from Primitives.base import Ray


class SimpleMaterial:
    def __init__(
        self,
        ambient_coefficient: float,
        diffuse_coefficient: float,
        diffuse_color,
        specular_coefficient: float,
        specular_color,
        specular_shininess: float = 32,
        reflectivity: float = 0.0,
    ):
        self.amb_coeff = ambient_coefficient
        self.diff_coeff = diffuse_coefficient
        self.diff_color = diffuse_color
        self.spec_coeff = specular_coefficient
        self.spec_color = specular_color
        self.spec_shiny = specular_shininess
        self.reflectivity = reflectivity

    def shade(self, hit_record, scene):

        device = hit_record.point.device
        N = hit_record.point.shape[0]

        # FIX: Garantir que a luz ambiente e as cores sejam Tensores no mesmo device
        # Use as_tensor ou clone().detach() para não criar cópias inúteis
        amb_light_tensor = torch.as_tensor(
            scene.ambient_light, device=device, dtype=torch.float32
        )
        diff_color_tensor = torch.as_tensor(
            self.diff_color, device=device, dtype=torch.float32
        )
        spec_color_tensor = torch.as_tensor(
            self.spec_color, device=device, dtype=torch.float32
        )
        # Pegue a posição da câmera sem criar um novo tensor do zero
        cam_eye_tensor = scene.camera.eye.clone().detach().to(device)
        # Agora a multiplicação funciona corretamente (elemento a elemento)
        total_color = amb_light_tensor * self.amb_coeff

        # Garante que total_color tenha o formato (N, 3)
        if total_color.dim() == 1:
            total_color = total_color.unsqueeze(0).expand(N, 3).clone()
        else:
            total_color = total_color.clone()

        # Componente Especular (Phong)
        view_dir = cam_eye_tensor - hit_record.point
        view_dir = view_dir / torch.linalg.norm(view_dir, dim=1, keepdim=True)

        for light in scene.lights:

            light_pos_tensor = torch.tensor(
                light.pos, device=device, dtype=torch.float32
            )
            light_color_tensor = torch.tensor(
                light.color, device=device, dtype=torch.float32
            )

            # Vetores da Luz
            light_vec = light_pos_tensor - hit_record.point
            dist = torch.linalg.norm(light_vec, dim=1, keepdim=True)
            light_dir = light_vec / dist

            epsilon = 1e-4
            shadow_ray_origin = hit_record.point + hit_record.normal * epsilon

            shadow_rays = Ray(shadow_ray_origin, light_dir)

            shadow_record = scene.hit(shadow_rays)
            # Se ele for (N), o .view(-1, 1) o transforma em (N, 1)
            shadow_hit_mask = shadow_record.hit_mask.view(-1, 1)

            # 2. Garanta que o t (tempo de interseção) seja (N, 1)
            shadow_t = shadow_record.t.view(-1, 1)

            # 3. Agora a comparação é segura, pois todos são (N, 1)
            # Ponto visível se: Não bateu em nada OU o que bateu está atrás da luz
            visible_mask = (shadow_hit_mask == 0) | (shadow_t >= (dist - epsilon))

            # 4. Converte para float (N, 1) - o unsqueeze não é mais necessário se já for (N, 1)
            vis = visible_mask.float()

            # Componente Difusa
            dot_diff = torch.sum(hit_record.normal * light_dir, dim=1, keepdim=True)
            diff_intensity = torch.clamp(dot_diff, min=0)
            diff_contribution = (diff_color_tensor * light_color_tensor) * (
                self.diff_coeff * diff_intensity
            )

            # Componente Especular (Phong)
            dot_nl = torch.sum(hit_record.normal * light_dir, dim=1, keepdim=True)
            reflect_dir = (2 * dot_nl * hit_record.normal) - light_dir
            reflect_dir = reflect_dir / torch.linalg.norm(
                reflect_dir, dim=1, keepdim=True
            )

            dot_spec = torch.sum(view_dir * reflect_dir, dim=1, keepdim=True)
            spec_intensity = torch.clamp(dot_spec, min=0) ** self.spec_shiny
            spec_contribution = (spec_color_tensor * light_color_tensor) * (
                self.spec_coeff * spec_intensity
            )

            # Adiciona a contribuição desta luz ao total
            total_color += (
                vis * (diff_contribution + spec_contribution) * light.intensity
            )

        return total_color


"""
epsilon = 1e-4
shadow_ray_origin = hit_record.point + hit_record.normal * epsilon


shadow_rays = Ray(shadow_ray_origin, light_dir)


shadow_record = scene.hit(shadow_rays)


visible_mask = (shadow_record.hit_mask == 0) | (shadow_record.t >= dist - epsilon)

vis = visible_mask.float().unsqueeze(1)

"""
