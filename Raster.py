from Base.Cam import *
from Base.Lights import *
from Base.Material import *
from Primitives.base import *
from Primitives.Triangle import Triangle
from Base.Cam import *
from tqdm import tqdm
import scenecharizard as scenecharizard
import importlib
import argparse
import torch
import numpy as np
from PIL import Image

def render_colors(scene, ray, hit_record):
    N = ray.ori.shape[0]
    device = ray.ori.device
    final_colors = torch.zeros((N, 3), device=device)
    
    # Removido o tqdm daqui para não poluir o terminal, 
    # pois agora a barra de progresso principal será a dos chunks
    for i, material in enumerate(scene.materials):
        material_id = i + 1
        mat_mask = (hit_record.hit_mask == material_id)
        if mat_mask.any():
            color_calculated = material.shade(hit_record, scene)
            final_colors[mat_mask] = color_calculated[mat_mask]

    return final_colors 

def rotate_cam(eye, theta, phi):

    device = eye.device
    theta = torch.tensor(float(theta), device=device)
    phi = torch.tensor(float(phi), device=device)
    
    # Matrizes de Rotação
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    cos_p, sin_p = torch.cos(phi), torch.sin(phi)
    
    # R_y (Rotação no eixo Y)
    R_y = torch.stack([
        torch.stack([cos_t, torch.tensor(0.0, device=device), sin_t]),
        torch.tensor([0.0, 1.0, 0.0], device=device),
        torch.stack([-sin_t, torch.tensor(0.0, device=device), cos_t])
    ])

    # R_x (Rotação no eixo X)
    R_x = torch.stack([
        torch.tensor([1.0, 0.0, 0.0], device=device),
        torch.stack([torch.tensor(0.0, device=device), cos_p, -sin_p]),
        torch.stack([torch.tensor(0.0, device=device), sin_p, cos_p])
    ])

    # Aplicar a rotação na posição da câmera
    # Nota: A ordem R_y @ R_x orbita a câmera ao redor da origem (0,0,0)
    R = R_y @ R_x
    new_eye = R @ eye
    
    return new_eye

def main(args):

    try:
        scene_module = importlib.import_module(args.scene)
    except ImportError:
        print(f"Erro: Não foi possível encontrar o módulo da cena '{args.scene}'")
        return
    
    device = args.device
    print(f"Renderizando usando: {device}")

    scene = scene_module.Scene(
            device=device,
        )

    camera = scene.camera
    lk_at = camera.look_at
    print(f"Câmera original: {camera}")

    if args.theta != 0 or args.phi != 0:
        camera.eye = rotate_cam(camera.eye, args.theta, args.phi)
        
    new_camera = Camera(camera.eye, lk_at, camera.up, camera.fov, camera.img_width, camera.img_height, camera.device)
    
    total_pixels = camera.img_height * camera.img_width
    chunk_size = args.chunk_size

    with torch.no_grad():
        final_colors = torch.zeros((total_pixels, 3), device=device)
        bg_color = torch.tensor(scene.background, device=device)

        for sample in range(args.num_samples):

            print(f"\nGerando raios da amostra {sample + 1}/{args.num_samples}...")
            # Gera todos os raios (apenas as origens e direções ocupam pouca memória)
            rays = new_camera.generate_all_rays(randomize=(args.num_samples > 1))

            print("Calculando interseções e cores por chunks...")
            # O tqdm agora itera sobre os chunks da imagem
            for i in tqdm(range(0, total_pixels, chunk_size), desc=f"Amostra {sample+1}"):
                end = min(i + chunk_size, total_pixels)
                
                # Fatiar raios para o chunk atual
                chunk_ori = rays.ori[i:end]
                chunk_dir = rays.dir[i:end]
                ray_chunk = Ray(chunk_ori, chunk_dir)

                # Processar as interseções apenas para o lote
                hit_rec_chunk = scene.hit(ray_chunk)

                # Computar as cores do lote
                current_colors_chunk = render_colors(scene, ray_chunk, hit_rec_chunk)
                
                # Aplicar a cor de fundo nos raios que não atingiram nada
                miss_mask = (hit_rec_chunk.hit_mask == 0)
                current_colors_chunk[miss_mask] = bg_color
                
                # Acumular no tensor final
                final_colors[i:end] += current_colors_chunk

        # Média das amostras para Anti-Aliasing
        final_colors /= args.num_samples

        # 3. Limitar os valores das cores entre 0.0 e 1.0 (evita que luzes fortes estourem a imagem)
        final_colors = torch.clamp(final_colors, 0.0, 1.0)

        print("\nProcessando a imagem final...")
        # 4. Redimensionar o Tensor (N, 3) para o formato de imagem (Altura, Largura, 3)
        img_height = camera.img_height
        img_width = camera.img_width
        image_tensor = final_colors.view(img_height, img_width, 3)

        # 5. Converter de PyTorch (GPU/CPU) para NumPy e escalar para o padrão RGB (0-255)
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)

        # 6. Salvar a imagem no disco
        img = Image.fromarray(image_np)
        output_filename = f"{args.output}.png" # Ajustado para .png
        img.save(output_filename)
        
        print(f"Renderização concluída com sucesso! Imagem salva como '{output_filename}'")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Raster module main function")
    parser.add_argument('-s', '--scene', type=str, help='Scene name', default='ball_scene')
    parser.add_argument('-n', '--num_samples', type=int, help='Number of samples per pixel for anti-aliasing', default=1)
    parser.add_argument('-l', '--lens_samples', type=int, help='Number of samples on the lens for depth of field', default=1)
    parser.add_argument('-j', '--num_jobs', type=int, help='Number of parallel jobs for rendering', default=4)
    parser.add_argument('-o', '--output', type=str, help='Output image file name (without extension)', default='output')
    parser.add_argument('-t', '--theta', type=float, help='Camera rotation angle around the Y-axis', default=0.0)
    parser.add_argument('-p', '--phi', type=float, help='Camera rotation angle around the X-axis', default=0.0)
    parser.add_argument('-d', '--device', type=str, help='Device to use for rendering (cpu or cuda)', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-c', '--chunk_size', type=int, help='Número de raios computados por lote (evita Out of Memory)', default=32768)
    args = parser.parse_args()
    main(args)