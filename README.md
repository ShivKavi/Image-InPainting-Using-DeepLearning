## Image-InPainting-Using-DeepLearning
Image Inpainting using GMCNN | Deep learning-based image restoration system that fills missing or damaged regions in images with high realism and structural consistency. Built with PyTorch.

## Contributions

- **Mokesh Prathaban**  
  Model architecture design and implementation.

- **Rushitha Alva**  
  Dataset preprocessing, augmentation, training, and hyperparameter tuning.

- **Vijay Kumar Reddy Marripati**  
  Performance evaluation and benchmarking.
 


## Motivation
Traditional inpainting techniques, such as patch-based methods, struggle with large missing areas and complex textures. Deep learning-based approaches, including GANs (Generative Adversarial Networks) and Diffusion Models, have shown superior performance in generating realistic image completions.

#Challenges in traditional methods:
Inability to handle large missing regions – Results in artifacts and unrealistic blending.
Lack of structural consistency – Fails to preserve edges and object shapes.
High computational cost – Training large generative models requires significant resources.

To overcome these challenges, our approach integrates:
Multi-Column Encoder: Captures both global and local features for better reconstruction.
PatchGAN Discriminator: Ensures realistic local texture generation.
Advanced Loss Functions: Combines L1 loss, adversarial loss, and perceptual loss for improved quality.

## Key Features
- Multi-column encoder for capturing fine-grained and global features.
- PatchGAN and GMCNN discriminator for improving local texture realism.
- Loss functions: L1 Loss, Adversarial Loss, and Perceptual Loss (LPIPS).
- Evaluation metrics: PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).
- Trained on the Places2 dataset (10,000 images, 256x256 resolution).
- Optimized for GPU training with Automatic Mixed Precision (AMP).

## Technologies Used
Deep Learning Frameworks: PyTorch, TensorFlow  
Datasets: Places2  
GAN-based Models: GMCNN  
Training Environment: Google Colab Pro (NVIDIA A100 GPU)

## Installation & Setup

1️) Clone the Repository
git clone https://github.com/your-username/Image-Inpainting.git

2️) Install Dependencies
pip install -r requirements.txt

3️) Run the Jupyter Notebook
jupyter notebook InPaintFP.ipynb

## Evaluation Metrics

To assess the performance of the inpainting model, we use the following metrics:
- Peak Signal-to-Noise Ratio (PSNR): Measures image reconstruction quality.
- Structural Similarity Index (SSIM): Evaluates structural fidelity.
- Inference Speed: Assesses real-time feasibility.

## Results
Kindly check the Results folder for the screenshots of the results

## References
- Pathak, D., et al. (2016). "Context Encoders: Feature Learning by Inpainting." CVPR.  
- Yu, J., et al. (2019). "Free-Form Image Inpainting with Gated Convolution." ICCV.  
- Iizuka, S., et al. (2017). "Globally and Locally Consistent Image Completion." SIGGRAPH.  
- Liu, G., et al. (2018). "Image Inpainting for Irregular Holes Using Partial Convolutions." ECCV.

