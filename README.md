# Generative Adversarial Networks (GANs) Implemented from Scratch  

This repository contains my implementations of various **Generative Adversarial Networks (GANs)**, re-created from scratch based on **Aladdin Persson's GAN tutorial series**. Each implementation explores different architectures, training techniques, and improvements for generating high-quality images.  

## 📌 Implemented Models  

### 1️⃣ Simple GAN  
A basic Generative Adversarial Network (GAN) with a simple **fully connected neural network** for generating images from noise.  
📂 **Folder:** `1_SimpleGAN`  

### 2️⃣ Deep Convolutional GAN (DCGAN)  
Introduces **convolutional layers** for more stable training and higher-quality image generation.  
📂 **Folder:** `2_DCGAN`  

### 3️⃣ Wasserstein GAN (WGAN & WGAN-GP)  
Enhances GAN training stability by replacing the traditional loss function with the **Wasserstein loss**. The gradient penalty (WGAN-GP) further improves convergence.  
📂 **Folder:** `3_WassersteinGAN`  

### 4️⃣ Pix2Pix  
An **image-to-image translation** GAN that learns a **mapping between paired images** (e.g., sketches to realistic images). Uses a **U-Net generator** and a **PatchGAN discriminator**.  
📂 **Folder:** `4_Pix2Pix`  

### 5️⃣ CycleGAN  
An unpaired image-to-image translation model that allows for **style transfer** between two domains (e.g., horses ↔ zebras, summer ↔ winter).  
📂 **Folder:** `5_CycleGAN`  

### 6️⃣ Progressive Growing GAN (ProGAN)  
A **progressively growing GAN**, where the generator and discriminator **start small and gradually add layers**, improving training stability and image resolution.  
📂 **Folder:** `6_ProGAN`  

### 7️⃣ Super-Resolution GAN (SRGAN)  
A GAN trained for **super-resolution**, enhancing low-resolution images into high-quality outputs.  
📂 **Folder:** `7_SRGAN`  

### 8️⃣ Enhanced Super-Resolution GAN (ESRGAN)  
An improvement over SRGAN, introducing **Residual-in-Residual Dense Blocks (RRDB)** and **perceptual loss** for state-of-the-art super-resolution.  
📂 **Folder:** `8_ESRGAN`  

---

## 📖 References  

- **Aladdin Persson’s YouTube Playlist on GANs:** [Watch Here](https://www.youtube.com/playlist?list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va)  
