# NutriNet – Vision Transformer (ViT) for Food Classification

Implemented the ViT architecture from scratch in PyTorch, based on "An Image is Worth 16x16 Words." Built a food image classification pipeline with patch embedding, positional encoding, and transformer encoder blocks (MSA, LayerNorm, MLP, residuals). Trained on a multi-class dataset (Food 101) using Adam optimizer with learning rate warmup and cross-entropy loss. Added a nutritional estimation module providing calorie content and performed comparisons with CNN baselines, showcasing ViT's effectiveness in food recognition.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [Technologies Used](#technologies-used)
- [License](#license)

## Project Overview

NutriNet is a comprehensive food classification system that leverages the power of Vision Transformer (ViT) architecture for accurate food recognition and nutritional analysis. This project implements the groundbreaking "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" paper from scratch using PyTorch.

### Key Features:
- **Custom ViT Implementation**: Built from scratch with patch embedding, positional encoding, and multi-head self-attention
- **Food Classification**: Trained on the Food-101 dataset with 101 different food categories
- **Nutritional Analysis**: Integrated calorie estimation and nutritional information
- **Performance Comparison**: Benchmarked against traditional CNN architectures
- **Production Ready**: Includes a Streamlit web application for real-time predictions

### Architecture Highlights:
- Multi-Scale Attention (MSA) mechanisms
- Layer Normalization and MLP blocks
- Residual connections for improved gradient flow
- Adam optimizer with learning rate warmup scheduling
- Cross-entropy loss optimization

## Repository Structure

```
NutriNet-Vision-Transformer-ViT/
├── Data/                          # Dataset and data processing scripts
├── app/                           # gradio web application
├── extras/                        # Additional utilities and resources
├── 01_PyTorch_workflow.ipynb      
├── 02_Neural_Networks_classification_with_PyTorch.ipynb
├── 03_PyTorch_computer_vision.ipynb
├── 04_pytorch_custom_datasets.ipynb
├── 05_pytorch_going_modular_cell_mode.ipynb
├── 06_pytorch_transfer_learning.ipynb
├── 07_pytorch_experiment_tracking.ipynb
├── 08_pytorch_paper_replicating.ipynb
├── 09_pytorch_model_deployment.ipynb
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- At least 8GB RAM

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/omkar00004/NutriNet-Vision-Transformer-ViT.git
   cd NutriNet-Vision-Transformer-ViT
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision torchaudio
   pip install numpy pandas matplotlib seaborn
   pip install streamlit pillow
   pip install jupyter notebook
   ```

4. **Download the Food-101 dataset:**
   ```bash
   # The dataset will be automatically downloaded when running the training scripts
   # Or manually download from: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
   ```

## Usage

### Training the Model

1. **Start with the PyTorch workflow:**
   ```bash
   jupyter notebook 01_PyTorch_workflow.ipynb
   ```

2. **Progress through the notebooks:**
   - Follow the numbered sequence from `01_` to `09_`
   - Each notebook builds upon the previous concepts
   - The Vision Transformer implementation is in `08_pytorch_paper_replicating_video.ipynb`

### Running the Web Application

1. **Launch the Streamlit app:**
   ```bash
   cd app
   streamlit run app.py
   ```

2. **Access the application:**
   - Open your browser to `http://localhost:8501`
   - Upload food images for classification
   - View nutritional information and confidence scores

### Model Inference

```python
import torch
from PIL import Image
# Load your trained model
model = torch.load('path/to/your/model.pth')
model.eval()

# Predict on new image
image = Image.open('path/to/food/image.jpg')
prediction = model(image)
print(f"Predicted food class: {prediction}")
```

## Contributing

We welcome contributions to NutriNet! Here's how you can help:

### Development Setup

1. **Fork the repository** on GitHub
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes** and add tests if applicable
4. **Commit your changes:**
   ```bash
   git commit -m "Add amazing feature"
   ```
5. **Push to your branch:**
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions and classes
- Include unit tests for new functionality
- Update documentation as needed
- Ensure all tests pass before submitting

### Areas for Contribution

- Model optimization and efficiency improvements
- Additional food datasets integration
- Enhanced nutritional analysis features
- Mobile app development
- Performance benchmarking

## Technologies Used

- **Deep Learning Framework**: PyTorch
- **Computer Vision**: Torchvision, PIL
- **Web Application**: Streamlit
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Development**: Jupyter Notebook
- **Version Control**: Git
- **Architecture**: Vision Transformer (ViT)
- **Optimization**: Adam optimizer with learning rate scheduling
- **Loss Function**: Cross-entropy loss
- **Dataset**: Food-101 (101 food categories)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ by [Omkar](https://github.com/omkar00004)**

*For questions, suggestions, or collaborations, feel free to open an issue or reach out directly!*
