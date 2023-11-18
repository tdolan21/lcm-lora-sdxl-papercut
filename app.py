import gradio as gr
import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image
from diffusers import AutoPipelineForInpainting, LCMScheduler
from diffusers import DiffusionPipeline, LCMScheduler
from PIL import Image, ImageEnhance
import io


def generate_image(prompt, num_inference_steps, guidance_scale):
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    adapter_id = "latent-consistency/lcm-lora-sdxl"

    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    # Load and fuse lcm lora
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()

    # Generate the image
    image = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
  
    return image

def inpaint_image(prompt, init_image, mask_image, num_inference_steps, guidance_scale):
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
    pipe.fuse_lora()

    if init_image is not None:
        init_image_path = init_image.name  # Get the file path
        init_image = Image.open(init_image_path).resize((1024, 1024))
    else:
        raise ValueError("Initial image not provided or invalid")

    if mask_image is not None:
        mask_image_path = mask_image.name  # Get the file path
        mask_image = Image.open(mask_image_path).resize((1024, 1024))
    else:
        raise ValueError("Mask image not provided or invalid")

    # Generate the inpainted image
    generator = torch.manual_seed(42)
    image = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    return image

def generate_image_with_adapter(prompt, num_inference_steps, guidance_scale):
    pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    torch_dtype=torch.float16
    ).to("cuda")

    # set scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Load and fuse lcm lora
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
    pipe.load_lora_weights("TheLastBen/Papercut_SDXL", weight_name="papercut.safetensors", adapter_name="papercut")

    # Combine LoRAs
    pipe.set_adapters(["lcm", "papercut"], adapter_weights=[1.0, 0.8])
    pipe.fuse_lora()
    generator = torch.manual_seed(0)
    # Generate the image
    image = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]
    pipe.unfuse_lora()
    return image

def pixel_art_image(prompt, negative_prompt, num_inference_steps, guidance_scale, num_images):
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
    pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    pipe.load_lora_weights(lcm_lora_id, adapter_name="lora")
    pipe.load_lora_weights("./pixel-art-xl.safetensors", adapter_name="pixel")

    pipe.set_adapters(["lora", "pixel"], adapter_weights=[1.0, 1.2])
    pipe.to(device="cuda", dtype=torch.float16)

    prompt = prompt
    negative_prompt = negative_prompt

    num_images = int(num_images)

    for i in range(num_images):
        img = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
    
    img.save(f"lcm_lora_{i}.png")

    return img



def modify_image(image, brightness, contrast):
    # Function to modify brightness and contrast
    image = Image.open(io.BytesIO(image))
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    return image

with gr.Blocks(gr.themes.Soft()) as demo:
    with gr.Row():
        image_output = gr.Image(label="Generated Image")

    with gr.Row():
        with gr.Accordion(label="Configuration Options"):
            prompt_input = gr.Textbox(label="Prompt", placeholder="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
            steps_input = gr.Slider(minimum=1, maximum=10, label="Inference Steps", value=4)
            guidance_input = gr.Slider(minimum=0, maximum=2, label="Guidance Scale", value=1)
            generate_button = gr.Button("Generate Image")

    with gr.Row():
        with gr.Accordion(label="Papercut Image Generation", open=False):
            adapter_prompt_input = gr.Textbox(label="Prompt", placeholder="papercut, a cute fox")
            adapter_steps_input = gr.Slider(minimum=1, maximum=10, label="Inference Steps", value=4)
            adapter_guidance_input = gr.Slider(minimum=0, maximum=2, label="Guidance Scale", value=1)
            adapter_generate_button = gr.Button("Generate Papercut Image")
        
    with gr.Row():
        with gr.Accordion(label="Pixel-Art-XL Image Generation", open=False):
            pixel_prompt_input = gr.Textbox(label="Prompt", placeholder="pixel, a cute corgi")
            pixel_negative_prompt_input = gr.Textbox(label="Negative Prompt", placeholder="3d render, realistic")
            pixel_steps_input = gr.Slider(minimum=1, maximum=10, label="Inference Steps", value=8)
            pixel_guidance_input = gr.Slider(minimum=0, maximum=3, label="Guidance Scale", value=1.2)
            pixel_num_images_input = gr.Number(label="Number of Images", value=9)
            pixel_generate_button = gr.Button("Generate PixelArt Image")


    with gr.Row():
        with gr.Accordion(label="Inpainting", open=False):
            inpaint_prompt_input = gr.Textbox(label="Prompt for Inpainting", placeholder="a castle on top of a mountain, highly detailed, 8k")
            init_image_input = gr.File(label="Initial Image")
            mask_image_input = gr.File(label="Mask Image")
            inpaint_steps_input = gr.Slider(minimum=1, maximum=10, label="Inference Steps", value=4)
            inpaint_guidance_input = gr.Slider(minimum=0, maximum=2, label="Guidance Scale", value=1)
            inpaint_button = gr.Button("Inpaint Image")

    with gr.Row():
        with gr.Accordion(label="Image Modification (Experimental)", open=False):
            brightness_slider = gr.Slider(minimum=0.5, maximum=1.5, step=1, label="Brightness")
            contrast_slider = gr.Slider(minimum=0.5, maximum=1.5, step=1, label="Contrast")
            modify_button = gr.Button("Modify Image")

    with gr.Row():
        with gr.Accordion(label="Additional Information", open=False):
            gr.Markdown(
                        "+ The first Accordion is Default LCM Generation"
                        + "\n"
                        + "+ The second Accordion is Papercut Generation"
                        + "\n"
                        + "+ The third Accordion is Pixel-Art-XL Generation"
                        + "\n"
                        + "+ The fourth Accordion is Inpainting"
                        + "\n"
                        + "+ The fifth Accordion is Image Modification"
                        )

    
    # Button for default LCM Image Generation
    generate_button.click(
        generate_image,
        inputs=[prompt_input, steps_input, guidance_input],
        outputs=image_output
    )
    # Button for Image Modification
    modify_button.click(
        modify_image,
        inputs=[image_output, brightness_slider, contrast_slider],
        outputs=image_output
    )
    # Button for Inpainting
    inpaint_button.click(
        inpaint_image,
        inputs=[inpaint_prompt_input, init_image_input, mask_image_input, inpaint_steps_input, inpaint_guidance_input],
        outputs=image_output
    )
    # Button for Papercut
    adapter_generate_button.click(
        generate_image_with_adapter,
        inputs=[adapter_prompt_input, adapter_steps_input, adapter_guidance_input],
        outputs=image_output
    )
    
    # Link the button to the pixel_art_image function
    pixel_generate_button.click(
            pixel_art_image,
            inputs=[
                pixel_prompt_input, 
                pixel_negative_prompt_input, 
                pixel_steps_input, 
                pixel_guidance_input,
                pixel_num_images_input 
            ],
            outputs=image_output
        )
   


demo.launch()
