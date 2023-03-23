# Image Diffusion with CLIP
This repository contains Python code for using a CLIP model to perform image diffusion

<p>This repository contains Python code for using a CLIP model to perform image diffusion. Specifically, it uses the Stable Diffusion algorithm to transform an input image according to a given prompt, while also using a negative prompt to ensure that the output is not too dissimilar from the input.</p>
<p>The code requires several dependencies, including the <a href="https://pillow.readthedocs.io/en/stable/" target="_new">Pillow</a> and <a href="https://pytorch.org/" target="_new">PyTorch</a> libraries, as well as the <a href="https://github.com/stabilityai/clip_interrogator" target="_new">CLIP Interrogator</a> and <a href="https://github.com/lucidrains/diffusion" target="_new">Diffusers</a> packages. To run the code, you will also need to have a pre-trained Stable Diffusion model, which can be downloaded from the <a href="https://modelhub.ai/stability-ai/stable-diffusion-2" target="_new">Stability AI Model Hub</a>.</p>
<p>The main entry point for the code is the <code>interrogate_images</code> function, which takes two image paths as input and returns two lists of prompts generated by the CLIP model. These prompts can then be used to perform image diffusion using the <code>StableDiffusionImg2ImgPipeline</code> class from the Diffusers package. The resulting image is saved as a PNG file.</p>
<p>This project is an implementation of a stable diffusion model, which is a machine learning algorithm that generates new images based on textual prompts. It is an analog to the popular MidJourney Remix, which uses a similar approach to generate images based on input text. This implementation uses the Stable Diffusion pipeline, which is designed to generate high-quality images and is particularly effective at image-to-image translation tasks. The pipeline is pretrained on a large dataset of images and prompts, and can be fine-tuned on specific tasks with additional data.</p>
<p>This code was developed by <a href="https://stability.ai/" target="_new">Stability AI</a> and is provided under the MIT License. Feel free to use it for your own image diffusion projects!</p>
