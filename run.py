from PIL import Image
from sys import argv
from typing import List, Tuple
from clip_interrogator import Config, Interrogator
from torch import autocast
import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionUpscalePipeline, DiffusionPipeline, StableDiffusionDepth2ImgPipeline, DPMSolverMultistepScheduler
from diffusers import PNDMScheduler, DDIMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers import LMSDiscreteScheduler
from PIL import Image
import random
from clip_interrogator import Config, Interrogator

prompt_1 = "a man in black jacket and is holding cell phone, boromir an anime world, trade offer meme, headshot profile picture, one onion ring, espn, facebook lizard tongue, he got big french musctache, point finger with ring on it, tessgarman, icon, uhq, sun down, kombi"
prompt_2 = "a green monsterfruit with pear tree in the background, shrek as neo from matrix, he looks like human minion, fbx, square, she has jiggly fat round belly, avatar image, old male, murky dusty deep, photo pinterest, sfw version, standing class, giga chad capaybara, video game"
  

def proof_text(prompt: str) -> str:
    """
    This function takes in a string prompt and returns a processed version of the prompt.
    
    Args:
    - prompt: A string representing the input prompt.
    
    Returns:
    - A string representing the processed version of the input prompt.
    """
    #words = prompt.split()
    #prompt_del_dublicate = " ".join(sorted(set(words), key=words.index))
    words = prompt.split()[:15]
    prompt_del_token = " ".join(words)
    return prompt_del_token


def interrogate_images(image_path_1: str, image_path_2: str) -> Tuple[List[str], List[str]]:
    """
    This function opens two images located at `image_path_1` and `image_path_2`, converts them to the RGB format, and uses the specified clip model to generate prompts for each image.

    Args:
    - image_path_1: A string representing the path to the first image.
    - image_path_2: A string representing the path to the second image.

    Returns:
    - A tuple containing two lists of strings representing the prompts generated for each image.

    """
    image_1 = Image.open(image_path_1).convert('RGB')
    image_2 = Image.open(image_path_2).convert('RGB')

    ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k"))
    prompt_1 = ci.interrogate_fast(image_1)
    prompt_2 = ci.interrogate_fast(image_2)
    return prompt_1, prompt_2

def initialize_image(image_path: str) -> Image:
    """
    This function takes in the path of an image, opens it, resizes it to 512x512, and converts it to RGB format.

    Args:
    - image_path: A string representing the path to the image.

    Returns:
    - A PIL Image object representing the initialized image.
    """
    init_image = Image.open(image_path)
    init_image = init_image.resize((512, 512)).convert("RGB")
    return init_image


if __name__ == "__main__":

  image_path_1 = argv[1]
  image_path_2 = argv[2]

  #model_id = 'stabilityai/stable-diffusion-2'
  model_id = "dreamlike-art/dreamlike-photoreal-2.0"

  scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

  pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
      model_id,
      revision="fp16",
      torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
      scheduler=scheduler
    ).to("cuda")
  pipe.enable_attention_slicing()

  prompt_1, prompt_2 = interrogate_images(image_path_1, image_path_2)

  init_image_1 = initialize_image(image_path_1)
  init_image_2 = initialize_image(image_path_2)

  prompt_1 = proof_text(prompt_1)
  prompt_2 = proof_text(prompt_2)

  #print(prompt_1)
  #print(prompt_2)
  seed = 121 #random.randint(0, 2147483647)
  num_inference_steps = int(50)
  strength = 1
  guidance_scale = 17

  with autocast("cuda"):
    image = pipe(
        prompt =  prompt_2 + "ighting, hyperdetailed, 8 k realistic, global illumination, radiant light, frostbite 3 engine, cryengine, trending on artstation, digital art ",
        negative_prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy",
        image = init_image_1,
        num_inference_steps =  num_inference_steps,
        strength = strength,
        guidance_scale = guidance_scale,
        generator = torch.Generator("cuda").manual_seed(seed)).images
    image[0].save("sd_1.png")
    print(f"prompt: {prompt_1}\nstep: {num_inference_steps}\nstrength: {strength}\nguidance_scale: {guidance_scale}\nseed: {seed}")

  num_inference_steps_2 = int(35)
  strength_2 = 0.65
  guidance_scale_2 = 7

  with autocast("cuda"):
    image = pipe(
        prompt = "ultra detailed, photorealistic, " + prompt_1 + " high detail, high quality, 8K, photo realism",
        negative_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
        image = init_image_2,
        num_inference_steps = num_inference_steps_2,
        strength = strength_2,
        guidance_scale = guidance_scale_2,
        generator = torch.Generator("cuda").manual_seed(seed)).images
    print(f"prompt: {prompt_2}\nstep: {num_inference_steps_2}\nstrength: {strength_2}\nguidance_scale: {guidance_scale_2}\nseed: {seed}")

    image[0].save("sd_2.png")