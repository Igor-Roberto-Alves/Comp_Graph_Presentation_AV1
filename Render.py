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

    for i, material in enumerate(scene.materials):
        target_id = i + 1
        mat_mask = hit_record.hit_mask == target_id

        if mat_mask.any():

            color_calculated = material.shade(hit_record, scene)
            final_colors[mat_mask] = color_calculated[mat_mask]

    return final_colors


def process_chunk(scene, camera, chunk_start, chunk_end, num_samples, device):

    chunk_size = chunk_end - chunk_start
    chunk_colors = torch.zeros((chunk_size, 3), device=device)

    for sample in range(num_samples):

        rays = camera.generate_all_rays(randomize=(num_samples > 1))
        rays_chunk = Ray(
            origin=rays.ori[chunk_start:chunk_end],
            direction=rays.dir[chunk_start:chunk_end],
        )

        hit_rec = scene.hit(rays_chunk)

        current_colors = render_colors(scene, rays_chunk, hit_rec)

        bg_color = torch.tensor(scene.background, device=device)
        miss_mask = hit_rec.hit_mask == 0
        current_colors[miss_mask] = bg_color

        chunk_colors += current_colors

    chunk_colors /= num_samples

    return chunk_colors


def main(args):
    try:
        scene_module = importlib.import_module(args.scene)
    except ImportError:
        print(f"Erro: Não foi possível encontrar o módulo da cena '{args.scene}'")
        return

    device = args.device
    print(f"Renderizando usando: {device}")

    scene = scene_module.Scene(device=device)

    camera = scene.camera
    lk_at = camera.look_at

    new_camera = Camera(
        camera.eye,
        lk_at,
        camera.up,
        camera.fov,
        camera.img_width,
        camera.img_height,
        camera.device,
    )

    total_pixels = camera.img_height * camera.img_width
    chunk_size = args.chunk_size

    print(f"\n{'='*70}")
    print(f"📊 CONFIGURAÇÃO DE RENDERIZAÇÃO:")
    print(f"  Total de pixels: {total_pixels}")
    print(f"  Tamanho do chunk: {chunk_size}")
    print(f"  Número de chunks: {(total_pixels + chunk_size - 1) // chunk_size}")
    print(f"  Amostras por pixel: {args.num_samples}")
    print(f"  Materiais disponíveis: {len(scene.materials)}")
    print(f"  Objetos na cena: {len(scene.objects)}")
    print(f"{'='*70}\n")

    final_colors = torch.zeros((total_pixels, 3), device=device)
    num_chunks = (total_pixels + chunk_size - 1) // chunk_size

    with torch.no_grad():
        for chunk_idx in tqdm(range(num_chunks), desc="Processando chunks"):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, total_pixels)

            chunk_colors = process_chunk(
                scene, new_camera, chunk_start, chunk_end, args.num_samples, device
            )

            final_colors[chunk_start:chunk_end] = chunk_colors

    print("Finalizando renderização...")

    # Clamp cores
    final_colors = torch.clamp(final_colors, 0.0, 1.0)

    # Redimensionar para imagem
    img_height = camera.img_height
    img_width = camera.img_width
    image_tensor = final_colors.view(img_height, img_width, 3)

    # Converter para NumPy e escalar RGB
    image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)

    # Salvar
    img = Image.fromarray(image_np)
    output_filename = f"{args.output}.png"
    img.save(output_filename)

    print(f"\n✅ Renderização concluída!")
    print(f"   Imagem salva como: '{output_filename}'")
    print(f"   Resolução: {img_width}x{img_height}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ray tracing renderer com processamento por chunks"
    )

    parser.add_argument(
        "-s", "--scene", type=str, help="Scene name", default="scenecharizard"
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        help="Number of samples per pixel for anti-aliasing",
        default=1,
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output image file name", default="output"
    )
    parser.add_argument(
        "-t",
        "--theta",
        type=float,
        help="Camera rotation angle around the Y-axis",
        default=0.0,
    )
    parser.add_argument(
        "-p",
        "--phi",
        type=float,
        help="Camera rotation angle around the X-axis",
        default=0.0,
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        help="Device to use for rendering (cpu or cuda)",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "-c",
        "--chunk_size",
        type=int,
        help="Chunk size for memory efficiency",
        default=19200,
    )

    args = parser.parse_args()
    main(args)
