from PIL import Image
from torchvision.utils import make_grid
import datetime
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import torch


def pil_to_np_array(pil_image):
    ar = np.array(pil_image)
    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]
    return ar.astype(np.float32) / 255.


def np_to_torch_array(np_array):
    return torch.from_numpy(np_array)[None, :]


def torch_to_np_array(torch_array):
    return torch_array.detach().cpu().numpy()[0]


def read_image(path, image_size=-1):
    pil_image = Image.open(path)
    return pil_image


def crop_image(image, crop_factor=64):
    shape = (image.size[0] - image.size[0] % crop_factor, image.size[1] - image.size[1] % crop_factor)
    bbox = [int((image.shape[0] - shape[0]) / 2), int((image.shape[1] - shape[1]) / 2),
            int((image.shape[0] + shape[0]) / 2), int((image.shape[1] + shape[1]) / 2)]
    return image.crop(bbox)


def get_image_grid(images, nrow=3):
    torch_images = [torch.from_numpy(x) for x in images]
    grid = make_grid(torch_images, nrow)
    return grid.numpy()


def visualize_sample(*images_np, nrow=3, size_factor=10, interactive=False):
    c = max(x.shape[0] for x in images_np)
    images_np = [x if (x.shape[0] == c) else np.concatenate([x, x, x], axis=0) for x in images_np]
    grid = get_image_grid(images_np, nrow)
    p = mp.Process(target=show_image_grid, args=(grid, len(images_np), size_factor, interactive, ))
    p.start()
    return p


def show_image_grid(grid, images_np_len, size_factor, interactive):
    if interactive:
      plt.ion()
    plt.figure(figsize=(images_np_len + size_factor, 12 + size_factor))
    plt.axis('off')
    plt.imshow(grid.transpose(1, 2, 0))
    if interactive:
      plt.draw()
      plt.pause(2)
    else:
      plt.show()


def max_dimension_resize(image_pil, mask_pil, max_dim):
    w, h = image_pil.size
    if w > max_dim:
        h = int((h / w) * max_dim)
        w = int(max_dim)
    if h > max_dim:
        w = int((w / h) * max_dim)
        h = int(max_dim)
    print(f'Output size: {w} x {h}')
    return image_pil.resize((w, h)), mask_pil.resize((w, h))


def preprocess_images(image_path, mask_path, max_dim, interactive=False):
    image_pil = read_image(image_path).convert('RGB')
    mask_pil = read_image(mask_path).convert('RGB')

    image_pil, mask_pil = max_dimension_resize(image_pil, mask_pil, max_dim)

    image_np = pil_to_np_array(image_pil)
    mask_np = pil_to_np_array(mask_pil)

    print('Visualizing mask overlap...')

    p = visualize_sample(image_np, mask_np, image_np * mask_np, nrow=3, size_factor=10, interactive=interactive)
    p.join()

    return image_np, mask_np


def save_image_from_np_array(output_image, output_name, overwrite=False):
    pil_image = Image.fromarray((output_image.transpose(1, 2, 0) * 255.0).astype('uint8'))

    output_path = f'{output_name}.jpg'
    if os.path.isfile(output_path) and not overwrite:
        output_path = f'{output_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
    print(f'\nSaving output image to: "{output_path}"\n')

    pil_image.save(output_path)
