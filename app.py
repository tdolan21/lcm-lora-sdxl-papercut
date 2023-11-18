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
        with gr.Accordion(label="Papercut Image Generation"):
            adapter_prompt_input = gr.Textbox(label="Prompt", placeholder="papercut, a cute fox")
            adapter_steps_input = gr.Slider(minimum=1, maximum=10, label="Inference Steps", value=4)
            adapter_guidance_input = gr.Slider(minimum=0, maximum=2, label="Guidance Scale", value=1)
            adapter_generate_button = gr.Button("Generate Image with Adapter")

    with gr.Row():
        with gr.Accordion(label="Inpainting"):
            inpaint_prompt_input = gr.Textbox(label="Prompt for Inpainting", placeholder="a castle on top of a mountain, highly detailed, 8k")
            init_image_input = gr.File(label="Initial Image")
            mask_image_input = gr.File(label="Mask Image")
            inpaint_steps_input = gr.Slider(minimum=1, maximum=10, label="Inference Steps", value=4)
            inpaint_guidance_input = gr.Slider(minimum=0, maximum=2, label="Guidance Scale", value=1)
            inpaint_button = gr.Button("Inpaint Image")

    with gr.Row():
        with gr.Accordion(label="Image Modification (Experimental)"):
            brightness_slider = gr.Slider(minimum=0.5, maximum=1.5, step=1, label="Brightness")
            contrast_slider = gr.Slider(minimum=0.5, maximum=1.5, step=1, label="Contrast")
            modify_button = gr.Button("Modify Image")

    

    generate_button.click(
        generate_image,
        inputs=[prompt_input, steps_input, guidance_input],
        outputs=image_output
    )

    modify_button.click(
        modify_image,
        inputs=[image_output, brightness_slider, contrast_slider],
        outputs=image_output
    )
    inpaint_button.click(
        inpaint_image,
        inputs=[inpaint_prompt_input, init_image_input, mask_image_input, inpaint_steps_input, inpaint_guidance_input],
        outputs=image_output
    )
    adapter_generate_button.click(
        generate_image_with_adapter,
        inputs=[adapter_prompt_input, adapter_steps_input, adapter_guidance_input],
        outputs=image_output
    )

demo.launch()