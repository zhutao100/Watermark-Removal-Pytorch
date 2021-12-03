import argparse
from api import remove_watermark

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Removing Watermark')
    parser.add_argument('--image-path', '-i', type=str,
                        default='./data/watermark-unavailable/watermarked/watermarked0.png',
                        help='Path to the "watermarked" image.')
    parser.add_argument('--mask-path', '-m', type=str, default='./data/watermark-unavailable/masks/mask0.png',
                        help='Path to the "watermark" image.')
    parser.add_argument('--input-depth', type=int, default=32,
                        help=("Max channel dimension of the noise input. "
                              "Set it based on gpu/device memory you have available."))
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--training-steps', type=int, default=3000, help='Number of training iterations.')
    parser.add_argument('--show-step', type=int, default=200, help='Interval for visualizing results.')
    parser.add_argument('--reg-noise', type=float, default=0.03, help='Hyper-parameter for regularized noise input.')
    parser.add_argument('--max-dim', type=float, default=512, help='Max dimension of the final output image')
    parser.add_argument('--overwrite', '-f', action='store_true', help='Overwrite existing output file')
    parser.add_argument('--intermediate-results', action='store_true', help='Store intermediate results')
    parser.add_argument('--interactive', action='store_true', help='Render images in matplotlib interactive mode')
    parser.add_argument('--timestamp', action='store_true', help='Always add timestamps to output filenames')
    parser.add_argument('--silent', action='store_true', help='Silent mode; no image window pop-ups.')

    args = parser.parse_args()

    remove_watermark(
        image_path=args.image_path,
        mask_path=args.mask_path,
        max_dim=args.max_dim,
        show_step=args.show_step,
        reg_noise=args.reg_noise,
        input_depth=args.input_depth,
        lr=args.lr,
        training_steps=args.training_steps,
        save_intermediate_results=args.intermediate_results,
        overwrite=args.overwrite,
        interactive=args.interactive,
        timestamp=args.timestamp,
        silent=args.silent,
    )
