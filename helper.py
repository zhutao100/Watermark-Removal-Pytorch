import datetime
import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision.utils import make_grid


def pil_to_np_array(pil_image):
    """Convert PIL image to numpy array.
    
    Args:
        pil_image (PIL.Image): Input PIL image
        
    Returns:
        numpy.ndarray: Normalized numpy array with values in [0, 1]
    """
    ar = np.array(pil_image)
    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]
    return ar.astype(np.float32) / 255.


def np_to_torch_array(np_array, device=None):
    """Convert numpy array to torch tensor with batch dimension.
    
    Args:
        np_array (numpy.ndarray): Input numpy array
        device (torch.device, optional): Device to place tensor on
        
    Returns:
        torch.Tensor: Torch tensor with added batch dimension
    """
    tensor = torch.from_numpy(np_array).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def torch_to_np_array(torch_array):
    """Convert torch tensor to numpy array, removing batch dimension.
    
    Args:
        torch_array (torch.Tensor): Input torch tensor with batch dimension
        
    Returns:
        numpy.ndarray: Numpy array without batch dimension
    """
    return torch_array.detach().cpu().numpy().squeeze(0)


def read_image(path):
    """Read an image from file.
    
    Args:
        path (str): Path to the image file
        
    Returns:
        PIL.Image: Loaded PIL image
    """
    pil_image = Image.open(path)
    return pil_image


def crop_image(image, crop_factor=64):
    """Crop image to be divisible by crop_factor.
    
    Args:
        image (PIL.Image): Input PIL image
        crop_factor (int, optional): Factor to make dimensions divisible by. Defaults to 64.
        
    Returns:
        PIL.Image: Cropped image
    """
    shape = (image.size[0] - image.size[0] % crop_factor, image.size[1] - image.size[1] % crop_factor)
    bbox = [int((image.size[1] - shape[1]) / 2), int((image.size[0] - shape[0]) / 2),
            int((image.size[1] + shape[1]) / 2), int((image.size[0] + shape[0]) / 2)]
    return image.crop(bbox)


def get_image_grid(images, nrow=3):
    """Create a grid of images.
    
    Args:
        images (list): List of numpy arrays representing images
        nrow (int, optional): Number of images per row. Defaults to 3.
        
    Returns:
        numpy.ndarray: Grid image as numpy array
    """
    torch_images = [torch.from_numpy(x) for x in images]
    grid = make_grid(torch_images, nrow)
    return grid.numpy()


def visualize_sample(*images_np, nrow=3, size_factor=10, interactive=False):
    """Visualize sample images in a separate process.
    
    Args:
        *images_np: Variable number of numpy arrays representing images
        nrow (int, optional): Number of images per row. Defaults to 3.
        size_factor (int, optional): Size factor for the figure. Defaults to 10.
        interactive (bool, optional): Whether to use interactive mode. Defaults to False.
        
    Returns:
        multiprocessing.Process: Process handling the visualization
    """
    c = max(x.shape[0] for x in images_np)
    images_np = [x if (x.shape[0] == c) else np.concatenate([x, x, x], axis=0) for x in images_np]
    grid = get_image_grid(images_np, nrow)
    p = mp.Process(target=show_image_grid, args=(grid, len(images_np), size_factor, interactive, ))
    p.start()
    return p


def show_image_grid(grid, images_np_len, size_factor, interactive):
    """Show image grid in matplotlib.
    
    Args:
        grid (numpy.ndarray): Grid image to display
        images_np_len (int): Number of images in the grid
        size_factor (int): Size factor for the figure
        interactive (bool): Whether to use interactive mode
    """
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
    """Resize image and mask maintaining aspect ratio with maximum dimension constraint.
    
    Args:
        image_pil (PIL.Image): Input image
        mask_pil (PIL.Image): Input mask
        max_dim (int): Maximum dimension allowed
        
    Returns:
        tuple: Resized image and mask as PIL images
    """
    w, h = image_pil.size
    if w > max_dim:
        h = int((h / w) * max_dim)
        w = int(max_dim)
    if h > max_dim:
        w = int((w / h) * max_dim)
        h = int(max_dim)
    print(f'Output size: {w} x {h}')
    return image_pil.resize((w, h)), mask_pil.resize((w, h))


def preprocess_images(image_path, mask_path, max_dim, interactive=False, silent=False):
    """Preprocess image and mask for watermark removal.
    
    Args:
        image_path (str): Path to the watermarked image
        mask_path (str): Path to the mask image
        max_dim (int): Maximum dimension for output
        interactive (bool, optional): Interactive mode for visualization. Defaults to False.
        silent (bool, optional): Silent mode. Defaults to False.
        
    Returns:
        tuple: Preprocessed image and mask as numpy arrays
    """
    image_pil = read_image(image_path).convert('RGB')
    mask_pil = read_image(mask_path).convert('RGB')

    image_pil, mask_pil = max_dimension_resize(image_pil, mask_pil, max_dim)

    image_np = pil_to_np_array(image_pil)
    mask_np = pil_to_np_array(mask_pil)

    if not silent:
        print('Visualizing mask overlap...')
        p = visualize_sample(image_np, mask_np, image_np * mask_np, nrow=3, size_factor=10, interactive=interactive)
        p.join()

    return image_np, mask_np


def save_image_from_np_array(output_image, output_name, overwrite=False, timestamp=False):
    """Save numpy array as image file.
    
    Args:
        output_image (numpy.ndarray): Image data to save
        output_name (str): Base name for output file
        overwrite (bool, optional): Overwrite existing files. Defaults to False.
        timestamp (bool, optional): Add timestamp to filename. Defaults to False.
    """
    pil_image = Image.fromarray((output_image.transpose(1, 2, 0) * 255.0).astype('uint8'))

    output_path = f'{output_name}.jpg'
    if (os.path.isfile(output_path) and not overwrite) or timestamp:
        output_path = f'{output_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
    print(f'\nSaving output image to: "{output_path}"\n')

    pil_image.save(output_path, quality=95)
