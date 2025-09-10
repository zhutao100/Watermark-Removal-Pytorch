# Watermark Removal using Deep Image Priors with PyTorch

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5137544.svg)](https://doi.org/10.5281/zenodo.5137544) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
<img src='final_outputs.webp' alt="Final Results" style="float: center; border-radius: 6px;">
</div>

```python
from api import remove_watermark

remove_watermark(
    image_path = IMAGE_NAME,
    mask_path = MASK_NAME,
    max_dim = MAX_DIM,
    show_step = SHOW_STEPS,
    reg_noise = REG_NOISE,
    input_depth = INPUT_DEPTH,
    lr = LR,
    training_steps = TRAINING_STEPS,
    tqdm_length = 900
)
```

## 📋 Project Overview

This project implements the groundbreaking **Deep Image Prior** technique for watermark removal using PyTorch. Based on the research paper [Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior), this approach leverages the inherent structure of CNNs for image restoration without requiring any training data.

### 🚀 Key Innovation

The project's primary contribution is solving the **practical real-world scenario** where the watermark is **not available separately** - a common challenge not addressed in the original paper. The solution transforms watermark removal into an image inpainting task through manual overlay creation.

## 🧠 Technical Foundation

### Deep Image Prior Concept
CNN architectures alone can provide sufficient image priors for restoration tasks, eliminating the need for training on large datasets. The generator network learns to reconstruct clean images by optimizing against the structure of the network itself rather than learned representations.

### Mathematical Foundation
Watermarked images can be represented as the Hadamard product:
```
Watermarked Image = Original Image × Watermark
```

The training objective minimizes L2 loss:
```
Loss = ||Generated Image × Watermark - Watermarked Image||²
```

## 🏗️ Architecture & Implementation

### Neural Network Architecture
- **Model**: `SkipEncoderDecoder` with 5-layer U-Net style architecture
- **Parameters**: ~500,000 (optimized from ~3M for faster inference)
- **Features**: Skip connections, concatenation operations, batch normalization
- **Activation**: LeakyReLU (0.2) with Sigmoid output

### Key Components
```
├── api.py              # Main API wrapper and inference pipeline
├── helper.py           # Image preprocessing, visualization utilities
├── inference.py        # Command-line interface for batch processing
├── notebook.ipynb      # Interactive Jupyter notebook for experimentation
└── model/
    ├── generator.py    # Core neural network architecture
    └── modules.py      # Custom PyTorch modules (Conv2dBlock, Concat, etc.)
```

### Technical Features
- **Multi-platform Support**: CUDA, MPS (Apple Silicon), CPU fallback
- **Flexible Configuration**: Learning rate, input depth, regularization noise, training steps
- **Real-time Visualization**: Progress tracking and intermediate results display
- **Image Processing**: Automatic resizing, preprocessing, and format conversion
- **Output Options**: Timestamped filenames, overwrite protection, silent mode

## 📁 Data Organization

```
data/
├── watermark-available/     # Scenario 1: Watermark available
│   ├── image1.png, image2.png, image3.png
│   └── watermark.png
└── watermark-unavailable/   # Scenario 2: Watermark unavailable
    ├── watermarked/         # Watermarked images
    └── masks/              # Manual overlay masks (mask0.png, mask1.png, etc.)
```

## 🎯 Two Usage Scenarios

### Scenario 1: Watermark Available
**Requirements:**
- Exact watermark image with matching scale, position, and spatial transformations
- Watermark must precisely match the applied watermark in the image

**Process:**
1. Provide watermarked image and corresponding watermark mask
2. Generator learns to reconstruct original image using L2 loss optimization
3. Output: Clean, watermark-free image

### Scenario 2: Watermark Unavailable (Practical Solution)
**Challenge:**
- Only watermarked image is available
- No prior information about watermark location or content
- Must work without training on datasets

**Innovative Solution:**
1. **Manual Overlay Creation**: User manually highlights watermarked regions using simple tools (MS Paint, GIMP, etc.)
2. **Mask Generation**: Create binary masks covering watermarked areas
3. **Inpainting**: Transform problem into image inpainting task
4. **Processing**: Use same optimization pipeline as Scenario 1

**Advantages:**
- ✅ No watermark detection model required
- ✅ No need for large training datasets
- ✅ Avoids adversarial training complexity
- ✅ Produces high-quality results with minimal artifacts
- ✅ Works with any watermark type or pattern

## 🚀 Quick Start

### Installation
```bash
pip install torch torchvision pillow matplotlib tqdm numpy
```

### Basic Usage (API)
```python
from api import remove_watermark

# Simple watermark removal
remove_watermark(
    image_path="watermarked_image.jpg",
    mask_path="watermark_mask.png",
    max_dim=512,
    training_steps=3000,
    lr=0.01
)
```

### Command Line Interface
```bash
python inference.py \
    --image-path ./data/watermark-unavailable/watermarked/watermarked0.png \
    --mask-path ./data/watermark-unavailable/masks/mask0.png \
    --max-dim 512 \
    --training-steps 3000 \
    --lr 0.01 \
    --show-step 200
```

### Available Parameters
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--image-path` | Path to watermarked image | Required |
| `--mask-path` | Path to watermark/mask image | Required |
| `--max-dim` | Maximum output dimension | 512 |
| `--input-depth` | Noise input channels | 32 |
| `--lr` | Learning rate | 0.01 |
| `--training-steps` | Number of optimization steps | 3000 |
| `--show-step` | Visualization interval | 200 |
| `--reg-noise` | Noise regularization | 0.03 |
| `--overwrite` | Overwrite existing outputs | False |
| `--silent` | Suppress visualizations | False |

## 📊 Performance & Results

### Sample Results
<div align="center">
<img src='outputs/watermark-unavailable/progress.gif' alt="Generator's Progress" style="float: center; border-radius: 6px;">
</div>

**Key Achievements:**
- High-quality reconstruction with minimal artifacts
- Real-time progress visualization
- Consistent results across different watermark types
- Efficient processing (optimized architecture)

### Experiment Gallery
- **Experiment 0**: [Result](outputs/watermark-unavailable/output0.webp)
- **Experiment 1**: [Result](outputs/watermark-unavailable/output1.webp)
- **Experiment 2**: [Result](outputs/watermark-unavailable/output2.webp)
- **Experiment 3**: [Result](outputs/watermark-unavailable/output3.webp)
- **Experiment 4**: [Result](outputs/watermark-unavailable/output4.webp)
- **Experiment 5**: [Result](outputs/watermark-unavailable/output5.webp)
- **Experiment 6**: [Result](outputs/watermark-unavailable/output6.webp)

## 🎨 Beyond Watermark Removal

### General Image Inpainting
This technique extends to various image editing applications:
- **Object Removal**: Unwanted elements from photos
- **Image Restoration**: Damaged or corrupted image repair
- **Artistic Editing**: Creative image manipulation
- **Photo Enhancement**: Quality improvement and refinement

### Sample Image Editing Results
<div align="center">
<img src='outputs/image-editing/edit1.png' alt="Image Editing Example 1" style="float: center; border-radius: 6px;">
</div>

<div align="center">
<img src='outputs/image-editing/edit2.png' alt="Image Editing Example 2" style="float: center; border-radius: 6px;">
</div>

<div align="center">
<img src='outputs/image-editing/edit3.png' alt="Image Editing Example 3" style="float: center; border-radius: 6px;">
</div>

<div align="center">
<img src='outputs/image-editing/edit4.png' alt="Image Editing Example 4" style="float: center; border-radius: 6px;">
</div>

## ⚠️ Important Considerations

### Mask Quality Guidelines
- **Thinner masks generally produce better results**
- **Avoid covering large portions** of the image
- **Be precise** with watermark boundaries
- **Use high contrast** between masked and unmasked regions

### Technical Limitations
- The model learns image statistics from a single image
- Large masked areas may produce visible artifacts
- Results depend on mask quality and precision
- No training on diverse datasets affects generalization

### Best Practices
1. **Create precise masks** that cover only watermarked areas
2. **Use appropriate image sizes** (512x512 recommended for balance)
3. **Monitor loss progression** to ensure proper convergence
4. **Adjust parameters** based on specific image characteristics

## 🔧 Advanced Configuration

### Model Architecture Details
```python
# SkipEncoderDecoder configuration
SkipEncoderDecoder(
    input_depth=32,                    # Noise input channels
    num_channels_down=[128] * 5,       # Downsample channels
    num_channels_up=[128] * 5,         # Upsample channels  
    num_channels_skip=[128] * 5        # Skip connection channels
)
```

### Optimization Parameters
- **Optimizer**: Adam with default parameters
- **Loss Function**: Mean Squared Error (MSE)
- **Regularization**: Gaussian noise injection
- **Learning Rate**: 0.01 (adjust based on convergence)

## 📚 Documentation & Resources

- [Original Deep Image Prior Paper](https://dmitryulyanov.github.io/deep_image_prior)
- [Implementation Tutorial](https://brainbust.medium.com/watermark-removal-using-deep-image-priors-d37f87a9ca1)
- [Jupyter Notebook](notebook.ipynb) for interactive experimentation
- [Command Line Interface](inference.py) for batch processing

## 🤝 Contributing

This project builds upon the excellent work of the Deep Image Prior authors. Contributions are welcome for:
- Bug fixes and improvements
- Additional use case examples
- Performance optimizations
- Documentation enhancements

## 📄 License

[MIT License](LICENSE)

---

*This project demonstrates the power of combining deep learning theory with practical engineering to solve real-world image processing challenges.*