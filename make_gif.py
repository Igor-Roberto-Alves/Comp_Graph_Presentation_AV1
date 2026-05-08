import PIL.Image
import numpy as np
import os


def make_gif(images, output_path):
    if not images:
        print("Erro: A lista de imagens está vazia!")
        return
    
    # Salva o GIF
    images[0].save(output_path, save_all=True, append_images=images[1:], loop=0, duration=100)
    print(f"GIF salvo em: {output_path}")

# 1. Carregar imagens com ordenação
folder = "bunny"
image_files = [f for f in os.listdir(folder) if f.endswith(".png")]

if not image_files:
    print(f"Nenhuma imagem .png encontrada na pasta '{folder}'")
else:
    pil_images = []
    for filename in image_files:
        path = os.path.join(folder, filename)
        img = PIL.Image.open(path)
        # Opcional: converter para RGB se necessário
        pil_images.append(img.convert("RGB"))
    
    make_gif(pil_images, "output.gif")