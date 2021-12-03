from model.generator import SkipEncoderDecoder, input_noise
from torch import optim
from tqdm.auto import tqdm
import helper
import os
import torch


def remove_watermark(image_path, mask_path, max_dim, reg_noise,
                     input_depth, lr, show_step, training_steps, tqdm_length=100,
                     save_intermediate_results=False, overwrite=False, interactive=False,
                     visualize_intermediate_results=False, add_input_depth_postfix=True,
                     timestamp=False, silent=False):
    DTYPE = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    if not torch.cuda.is_available():
        print('\nSetting device to "cpu", since torch is not built with "cuda" support...')
        print('It is recommended to use GPU if possible...')

    output_prefix = image_path.split('/')[-1].split('.')[-2]
    input_depth_postfix = f'_input-depth-{input_depth}'

    image_np, mask_np = helper.preprocess_images(image_path, mask_path, max_dim, interactive, silent)

    print('Building the model...')
    generator = SkipEncoderDecoder(
        input_depth,
        num_channels_down=[128] * 5,
        num_channels_up=[128] * 5,
        num_channels_skip=[128] * 5
    ).type(DTYPE)

    objective = torch.nn.MSELoss().type(DTYPE)
    optimizer = optim.Adam(generator.parameters(), lr)

    image_var = helper.np_to_torch_array(image_np).type(DTYPE)
    mask_var = helper.np_to_torch_array(mask_np).type(DTYPE)

    generator_input = input_noise(input_depth, image_np.shape[1:]).type(DTYPE)

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
            generator_input = generator_input_saved + (noise.normal_() * reg_noise)

        output = generator(generator_input)

        loss = objective(output * mask_var, image_var * mask_var)
        loss.backward()

        if show_step and step % show_step == 0:
            output_image = helper.torch_to_np_array(output)
            if save_intermediate_results:
                intermediate_name = f'{intermediate_dir}/step{step}' + \
                    input_depth_postfix if add_input_depth_postfix else ''
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

    output_name = f'{output_prefix}-output_step-{training_steps}' + \
        input_depth_postfix if add_input_depth_postfix else ''
    helper.save_image_from_np_array(output_image, output_name, overwrite, timestamp)
