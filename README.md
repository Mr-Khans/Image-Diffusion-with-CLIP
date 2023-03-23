# Image Diffusion with CLIP
This repository contains Python code for using a CLIP model to perform image diffusion
![alt text for screen readers](/example/1.png)

<p>This repository contains Python code for using a CLIP model to perform image diffusion. Specifically, it uses the Stable Diffusion algorithm to transform an input image according to a given prompt, while also using a negative prompt to ensure that the output is not too dissimilar from the input.</p>
<p>The code requires several dependencies, including the <a href="https://pillow.readthedocs.io/en/stable/" target="_new">Pillow</a> and <a href="https://pytorch.org/" target="_new">PyTorch</a> libraries, as well as the <a href="https://github.com/stabilityai/clip_interrogator" target="_new">CLIP Interrogator</a> and <a href="https://github.com/lucidrains/diffusion" target="_new">Diffusers</a> packages. To run the code, you will also need to have a pre-trained Stable Diffusion model, which can be downloaded from the <a href="https://modelhub.ai/stability-ai/stable-diffusion-2" target="_new">Stability AI Model Hub</a>.</p>

<table><thead><tr><th>Example 1</th><th>Example 2</th></tr></thead><tbody><tr><td><img src="/example/1.png" style="height: 400px; width:400px;"/> </td><td><img src="/example/2.png" style="height: 400px; width:400px;"/></td></tr></tbody></table>

<p>The main entry point for the code is the <code>interrogate_images</code> function, which takes two image paths as input and returns two lists of prompts generated by the CLIP model. These prompts can then be used to perform image diffusion using the <code>StableDiffusionImg2ImgPipeline</code> class from the Diffusers package. The resulting image is saved as a PNG file.</p>

<p>This project is an implementation of a stable diffusion model, which is a machine learning algorithm that generates new images based on textual prompts. It is an analog to the popular MidJourney Remix, which uses a similar approach to generate images based on input text. This implementation uses the Stable Diffusion pipeline, which is designed to generate high-quality images and is particularly effective at image-to-image translation tasks. The pipeline is pretrained on a large dataset of images and prompts, and can be fine-tuned on specific tasks with additional data.</p>
<div class="flex flex-grow flex-col gap-3"><div class="min-h-[20px] flex flex-col items-start gap-4 whitespace-pre-wrap"><div class="markdown prose w-full break-words dark:prose-invert light"><p>To run the script with two image paths, open the command line interface and navigate to the directory where the script is located. Then run the following command:</p><pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg></button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-bash">python run.py path/to/image1.png path/to/image2.png
</code></div></div></pre><p>Replace <code>run.py</code>, <code>path/to/image1.png</code>, and <code>path/to/image2.png</code> with the actual names and paths of the script and images, respectively.</p><p>Note: Make sure that you have the required libraries and dependencies installed before running the script. You can refer to the <code>requirements.txt</code> file for a list of required libraries and their versions.</p></div></div></div>


<p>This code was developed by <a href="https://stability.ai/" target="_new">Stability AI</a> and is provided under the MIT License. Feel free to use it for your own image diffusion projects!</p>
