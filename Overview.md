## Project Overview: Watermark Removal using Deep Image Priors with PyTorch

This is a sophisticated deep learning project that implements watermark removal using the **Deep Image Prior** technique, which leverages the inherent structure of CNNs for image restoration without requiring any training data.

### Core Technology & Innovation

**Deep Image Prior Concept**: The project implements the groundbreaking idea that CNN architectures alone can provide sufficient image priors for restoration tasks, eliminating the need for training on large datasets. This is based on the paper [Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior).

**Key Innovation**: The project's main contribution is solving the practical scenario where the watermark is **not available separately** - a common real-world challenge. The author provides a manual overlay solution that transforms watermark removal into an image inpainting task.

### Architecture & Implementation

**Model Architecture**: 
- Uses a `SkipEncoderDecoder` CNN with 5 layers
- Reduced from ~3M to ~500K parameters for faster inference
- Implements depthwise separable convolutions for efficiency
- Uses skip connections and concatenation operations

**Key Components**:
- `api.py`: Main API wrapper with `remove_watermark()` function
- `helper.py`: Image preprocessing, visualization, and utility functions
- `inference.py`: Command-line interface for batch processing
- `model/generator.py`: Core neural network architecture
- `model/modules.py`: Custom PyTorch modules (Conv2dBlock, Concat, etc.)

### Two Usage Scenarios

**Scenario 1: Watermark Available**
- Requires exact watermark image with matching scale/position
- Uses Hadamard product: `Watermarked Image = Original Image × Watermark`
- Training minimizes L2 loss between `Generated Image × Watermark` and `Watermarked Image`

**Scenario 2: Watermark Unavailable (Practical Solution)**
- Only watermarked image is available
- User manually creates overlay masks highlighting watermarked regions
- Transforms problem into image inpainting task
- No need for watermark detection models or adversarial training

### Data Organization

```
data/
├── watermark-available/     # Original images + watermark
│   ├── image1.png, image2.png, image3.png
│   └── watermark.png
└── watermark-unavailable/   # Watermarked images + masks
    ├── watermarked/         # Watermarked images
    └── masks/              # Manual overlay masks
```

### Technical Features

- **Multi-platform support**: CUDA, MPS (Apple Silicon), CPU fallback
- **Flexible parameters**: Learning rate, input depth, regularization noise, training steps
- **Visualization**: Real-time progress tracking and intermediate results
- **Image processing**: Automatic resizing, preprocessing, and format conversion
- **Output options**: Timestamped filenames, overwrite protection, silent mode

### Applications Beyond Watermark Removal

The technique can be used for general image inpainting/editing:
- Removing unwanted objects from images
- Image restoration and enhancement
- Artistic image manipulation

### Project Structure

- **License**: MIT License
- **Dependencies**: PyTorch, PIL, matplotlib, tqdm, numpy
- **Cross-platform**: Works on Windows, macOS, Linux with GPU/CPU acceleration
- **Documentation**: Comprehensive README with examples and visual results

This project represents an elegant solution to a practical image processing problem, combining deep learning theory with practical engineering to create an effective, user-friendly watermark removal system.