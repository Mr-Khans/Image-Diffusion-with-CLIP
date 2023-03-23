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
from diffusers import LMSDiscreteScheduler
from PIL import Image
from clip_interrogator import Config, Interrogator



def proof_text(prompt: str): -> str:
    """
    This function takes in a string prompt and returns a processed version of the prompt.
    
    Args:
    - prompt: A string representing the input prompt.
    
    Returns:
    - A string representing the processed version of the input prompt.
    """
  words = prompt.split()
  prompt_del_dublicate = " ".join(sorted(set(words), key=words.index))
  words = prompt_del_dublicate.split()[:50]
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
    prompt_1 = ci.interrogate(image_1)
    prompt_2 = ci.interrogate(image_2)
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

  model_id = 'stabilityai/stable-diffusion-2'

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

  print(prompt_1)
  print(prompt_2)

  with autocast("cuda"):
    image = pipe(
        prompt = prompt_2,
        negative_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
        image = init_image_1,
        num_inference_steps = int(25),
        strength = 0.1,
        guidance_scale = 7).images
    image[0].save("sd_1.png")

  with autocast("cuda"):
    image = pipe(
        prompt = prompt_1,
        negative_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
        image = init_image_2,
        num_inference_steps = int(25),
        strength = 0.1,
        guidance_scale = 7).images
    image[0].save("sd_2.png")