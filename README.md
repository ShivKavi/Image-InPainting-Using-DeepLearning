# Image-InPainting-Using-DeepLearning
Image Inpainting using GMCNN | Deep learning-based image restoration system that fills missing or damaged regions in images with high realism and structural consistency. Built with PyTorch.

# Contributions
Mokesh Prathaban - Model architecture design and implementation
Rushitha Alva - Dataset preprocessing and augmentation and Training and hyperparameter tuning
Vijay kumar Reddy Marripati - Performance evaluation and benchmarking

# Motivation
Traditional inpainting techniques, such as patch-based methods, struggle with large missing areas and complex textures. Deep learning-based approaches, including GANs (Generative Adversarial Networks) and Diffusion Models, have shown superior performance in generating realistic image completions.

Challenges in traditional methods:
Inability to handle large missing regions – Results in artifacts and unrealistic blending.
Lack of structural consistency – Fails to preserve edges and object shapes.
High computational cost – Training large generative models requires significant resources.

To overcome these challenges, our approach integrates:
Multi-Column Encoder: Captures both global and local features for better reconstruction.
PatchGAN Discriminator: Ensures realistic local texture generation.
Advanced Loss Functions: Combines L1 loss, adversarial loss, and perceptual loss for improved quality.

# Key Features
##Deep learning-based image inpainting for high-quality restoration.
##GAN-based models to enhance texture and structural consistency.
##Loss function optimization for sharp and realistic results.
##Multi-dataset training to improve generalization across diverse images.
##Benchmarking against state-of-the-art models for performance evaluation.

# Technologies Used
Deep Learning Frameworks: PyTorch, TensorFlow, Keras
Datasets: Places2
GAN-based Models: GMCNN
Training Environment: Google Colab Pro (NVIDIA A100 GPU)

# Installation & Setup

1️) Clone the Repository
git clone https://github.com/your-username/Image-Inpainting.git

2️) Install Dependencies
pip install -r requirements.txt

3️) Run the Jupyter Notebook
jupyter notebook InPaintFP.ipynb

# Evaluation Metrics

To assess the performance of the inpainting model, we use the following metrics:

Peak Signal-to-Noise Ratio (PSNR) – Measures image reconstruction quality.
Structural Similarity Index (SSIM) – Evaluates structural fidelity.
Inference Speed – Assesses real-time feasibility.

# Results

Kindly check the Results folder for the screenshots of the results

# References

Pathak, D., et al. (2016). "Context Encoders: Feature Learning by Inpainting." CVPR.
Yu, J., et al. (2019). "Free-Form Image Inpainting with Gated Convolution." ICCV.
Iizuka, S., et al. (2017). "Globally and Locally Consistent Image Completion." SIGGRAPH.
Liu, G., et al. (2018). "Image Inpainting for Irregular Holes Using Partial Convolutions." ECCV.
