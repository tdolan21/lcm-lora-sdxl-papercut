# LCM-SDXL + LoRa

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF6F61?style=for-the-badge&logo=Gradio&logoColor=white)
![Diffusers](https://img.shields.io/badge/Diffusers-5B4638?style=for-the-badge&logo=Diffusers&logoColor=white)
![PIL](https://img.shields.io/badge/PIL-%230A0A0A.svg?&style=for-the-badge&logo=PIL&logoColor=white)

This application utilizes advanced AI models, including Stable Diffusion and various adapters, to generate and modify images based on textual prompts. It integrates Gradio for a user-friendly interface, allowing users to interact with different functionalities like image generation, inpainting, and image modification.

![Image Generated By DallE-3](assets/stable.png)

Full post and instructional available [here](https://tims-tutorials.vercel.app/blog/lcm_sdxl)

## Adapters Included
+ Papercut
+ Pixel-Art-XL ([download](https://huggingface.co/nerijs/pixel-art-xl/resolve/main/pixel-art-xl.safetensors?download=true))
+ AnimatedDiff GIF generation
  
### Setting up Conda Environment

Creating a dedicated Conda environment for your project is considered a best practice as it helps manage dependencies and avoid conflicts.

```bash
conda create --name torch python=3.10
conda activate torch
```

You should make sure that you have followed the correct steps located [here](https://pytorch.org/get-started/locally/) to install torch with GPU support
for your correct platform. If you have not done this, you will not be able to use the GPU.

I am on linux for example, so this would be my install command using conda.

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
```
Build the correct command for your machine on the website and make sure its installed properly by testing if torch has access to the gpu.

You do not need to locally install CUDA or cudnn, pytorch ships with prebuilt binaries for each version. These steps alone are enough for torch to see your GPU.


## Installation
First make sure that you hav the safetensors file available above downloaded and placed in the root directory of the application.

To set up this application, ensure you have Python installed. Then, install the required libraries using pip:

```bash
git clone https://github.com/tdolan21/lcm-lora-sdxl-papercut.git
cd lcm-lora-sdxl-papercut
pip install -r requirements.txt
```
Once you have the requirements installed you can start the application with:

```bash
gradio app.py
```
### Features
+ Generate paper mache images with dimensionality.
+ Generate pixel art in 16 bit stylings
+ Generate GIF with varying frame rate from custom prompt


