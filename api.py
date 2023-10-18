from model.generator import SkipEncoderDecoder, input_noise
from torch import optim
from tqdm.auto import tqdm
import helper
import os
import torch
import platform

if platform.system() == 'Darwin':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

DEFAULT_REG_NOISE = 0.03
DEFAULT_INPUT_DEPTH = 32
DEFAULT_LEARNING_RATE = 0.01


def form_output_postfix(reg_noise, input_depth, lr):
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
    DTYPE = torch.FloatTensor
    has_set_device = False
    if torch.cuda.is_available():
        device = 'cuda'
        has_set_device = True
        print("Setting Device to CUDA...")
    try:
        if torch.backends.mps.is_available():
            device = 'mps'
            has_set_device = True
            print("Setting Device to MPS...")
    except Exception as e:
        print(f"Your version of pytorch might be too old, which does not support MPS. Error: \n{e}")
        pass
    if not has_set_device:
        device = 'cpu'
        print('\nSetting device to "cpu", since torch is not built with "cuda" or "mps" support...')
        print('It is recommended to use GPU if possible...')

    output_prefix = image_path.split('/')[-1].split('.')[-2]
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
