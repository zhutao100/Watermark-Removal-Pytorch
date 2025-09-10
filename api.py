import os
from pathlib import Path

import torch
from torch import optim
from tqdm.auto import tqdm

from model.generator import SkipEncoderDecoder, input_noise
import helper

DEFAULT_REG_NOISE = 0.03
DEFAULT_INPUT_DEPTH = 32
DEFAULT_LEARNING_RATE = 0.01


def form_output_postfix(reg_noise, input_depth, lr):
    """Generate a postfix for output filenames based on non-default parameters.
    
    Args:
        reg_noise (float): Regularization noise value
        input_depth (int): Input depth value
        lr (float): Learning rate value
        
    Returns:
        str: Formatted postfix string
    """
    output_postfix = ''
    if input_depth != DEFAULT_INPUT_DEPTH:
        output_postfix += f'_input-depth-{input_depth}'
    if reg_noise != DEFAULT_REG_NOISE:
        output_postfix += f'_reg-noise-{reg_noise}'
    if lr != DEFAULT_LEARNING_RATE:
        output_postfix += f'_lr-{lr}'
    return output_postfix


def remove_watermark(image_path, mask_path, max_dim, reg_noise,
                     input_depth, lr, show_step, training_steps, tqdm_length=100,
                     save_intermediate_results=False, overwrite=False, interactive=False,
                     visualize_intermediate_results=False, timestamp=False, silent=False):
    """Remove watermark from an image using deep image priors.
    
    Args:
        image_path (str): Path to the watermarked image
        mask_path (str): Path to the mask image
        max_dim (float): Maximum dimension for output image
        reg_noise (float): Regularization noise parameter
        input_depth (int): Input depth for the generator
        lr (float): Learning rate for optimization
        show_step (int): Interval for visualizing results
        training_steps (int): Number of training iterations
        tqdm_length (int, optional): Length of tqdm progress bar. Defaults to 100.
        save_intermediate_results (bool, optional): Save intermediate results. Defaults to False.
        overwrite (bool, optional): Overwrite existing output files. Defaults to False.
        interactive (bool, optional): Render images in matplotlib interactive mode. Defaults to False.
        visualize_intermediate_results (bool, optional): Visualize intermediate results. Defaults to False.
        timestamp (bool, optional): Add timestamps to output filenames. Defaults to False.
        silent (bool, optional): Run in silent mode with no image window pop-ups. Defaults to False.
    """
    DTYPE = torch.FloatTensor

    # Improved device detection logic
    device = 'cpu'  # Default to CPU
    if torch.cuda.is_available():
        device = 'cuda'
        print("Setting Device to CUDA...")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("Setting Device to MPS...")
    else:
        print('\nSetting device to "cpu", since torch is not built with "cuda" or "mps" support...')
        print('It is recommended to use GPU if possible...')

    # Cross-platform compatible path handling
    output_prefix = Path(image_path).stem
    output_postfix = form_output_postfix(reg_noise, input_depth, lr)

    image_np, mask_np = helper.preprocess_images(image_path, mask_path, max_dim, interactive, silent)

    print('Building the model...')
    generator = SkipEncoderDecoder(
        input_depth,
        num_channels_down=[128] * 5,
        num_channels_up=[128] * 5,
        num_channels_skip=[128] * 5
    ).type(DTYPE).to(device)

    objective = torch.nn.MSELoss().type(DTYPE).to(device)
    optimizer = optim.Adam(generator.parameters(), lr)

    image_var = helper.np_to_torch_array(image_np).type(DTYPE).to(device)
    mask_var = helper.np_to_torch_array(mask_np).type(DTYPE).to(device)

    generator_input = input_noise(input_depth, image_np.shape[1:]).type(DTYPE).to(device)

    generator_input_saved = generator_input.detach().clone()
    noise = generator_input.detach().clone()

    print('\nStarting training...\n')

    progress_bar = tqdm(range(training_steps), desc='Completed', ncols=tqdm_length)

    visualize_processes = []
    intermediate_dir = f'{output_prefix}-intermediate'
    if save_intermediate_results:
        os.makedirs(intermediate_dir, exist_ok=True)
    for step in progress_bar:
        optimizer.zero_grad()

        generator_input = generator_input_saved
        if reg_noise > 0:
            generator_input += noise.normal_() * reg_noise

        output = generator(generator_input)

        loss = objective(output * mask_var, image_var * mask_var)
        loss.backward()

        if show_step and step % show_step == 0:
            output_image = helper.torch_to_np_array(output)
            if save_intermediate_results:
                intermediate_name = f'{intermediate_dir}/step-{step}{output_postfix}'
                helper.save_image_from_np_array(output_image, intermediate_name, overwrite, timestamp)

            if visualize_intermediate_results and not silent:
                p = helper.visualize_sample(image_np, output_image, nrow=2, size_factor=10, interactive=interactive)
                visualize_processes.append(p)

        progress_bar.set_postfix(Loss=loss.item())

        optimizer.step()

    for p in visualize_processes:
        p.join()

    output_image = helper.torch_to_np_array(output)
    if not silent:
        p_output = helper.visualize_sample(output_image, nrow=1, size_factor=10, interactive=interactive)
        p_output.join()

    output_name = f'{output_prefix}-output_step-{training_steps}{output_postfix}'
    helper.save_image_from_np_array(output_image, output_name, overwrite, timestamp)
