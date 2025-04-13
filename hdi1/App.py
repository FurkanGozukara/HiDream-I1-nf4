import torch
import gradio as gr
import logging
import os
import argparse
import sys
import subprocess
from pathlib import Path
import time

from .nf4 import *

# Resolution options
RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]

# Parse resolution string to get height and width
def parse_resolution(resolution_str):
    return tuple(map(int, resolution_str.split("(")[0].strip().split(" × ")))

# Create outputs directory if it doesn't exist
def ensure_output_dir():
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Open outputs folder based on OS
def open_outputs_folder():
    output_dir = ensure_output_dir()
    if sys.platform == "win32":
        subprocess.Popen(["explorer", output_dir])
    else:  # For Linux and macOS
        subprocess.Popen(["xdg-open", output_dir])
    return "Opened outputs folder"

def update_steps(model):
    return MODEL_CONFIGS[model]["num_inference_steps"]

def gen_img_helper(model, prompt, res, seed, num_steps, num_images):
    global pipe, current_model
    output_images = []
    seeds_used = []
    output_dir = ensure_output_dir()

    # 1. Check if the model matches loaded model, load the model if not
    if model != current_model:
        print(f"Unloading model {current_model}...")
        del pipe
        torch.cuda.empty_cache()
        
        print(f"Loading model {model}...")
        pipe, _ = load_models(model)
        current_model = model
        print("Model loaded successfully!")

    # 2. Generate images in batch
    res = parse_resolution(res)
    for i in range(num_images):
        image, used_seed = generate_image(pipe, model, prompt, res, seed, num_steps)
        
        # Save the image
        timestamp = int(time.time())
        filename = f"{output_dir}/image_{timestamp}_{used_seed}.png"
        image.save(filename)
        print(f"Saved image to {filename}")
        
        # Add to results
        output_images.append(image)
        seeds_used.append(used_seed)
        
        # Use the last seed + 1 for the next image if not random
        if seed != -1:
            seed = used_seed + 1

    # Return only the first image to display
    return output_images[0], seeds_used[0]


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="HiDream-I1-nf4 Dashboard")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    args = parser.parse_args()

    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    # Initialize with default model
    print("Loading default model (fast)...")
    current_model = "fast"
    pipe, _ = load_models(current_model)
    print("Model loaded successfully!")

    # Ensure outputs directory exists
    ensure_output_dir()

    # Create Gradio interface
    with gr.Blocks(title="HiDream-I1-nf4 Dashboard") as demo:
        gr.Markdown("# HiDream NF4 SECourses Improved App V1 : https://www.patreon.com/posts/126589906/")
        
        with gr.Row():
            with gr.Column():
                model_type = gr.Radio(
                    choices=list(MODEL_CONFIGS.keys()),
                    value="fast",
                    label="Model Type",
                    info="Select model variant"
                )
                
                prompt = gr.Textbox(
                    label="Prompt", 
                    placeholder="A cat holding a sign that says \"Hi-Dreams.ai\".", 
                    lines=3
                )
                
                resolution = gr.Radio(
                    choices=RESOLUTION_OPTIONS,
                    value=RESOLUTION_OPTIONS[0],
                    label="Resolution",
                    info="Select image resolution"
                )
                
                seed = gr.Number(
                    label="Seed (use -1 for random)", 
                    value=-1, 
                    precision=0
                )
                
                num_steps = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=MODEL_CONFIGS["fast"]["num_inference_steps"],
                    step=1,
                    label="Number of Steps",
                    info="More steps = higher quality but slower"
                )
                
                num_images = gr.Slider(
                    minimum=1,
                    maximum=1000,
                    value=1,
                    step=1,
                    label="Number of Images",
                    info="Generate multiple images at once"
                )
                
                generate_btn = gr.Button("Generate Image")
                seed_used = gr.Number(label="Seed Used", interactive=False)
                
            with gr.Column():
                output_image = gr.Image(label="Generated Image", type="pil", height=512)
                open_folder_btn = gr.Button("Open Outputs Folder")
        
        # Update steps when model changes
        model_type.change(
            fn=update_steps,
            inputs=[model_type],
            outputs=[num_steps]
        )
        
        generate_btn.click(
            fn=gen_img_helper,
            inputs=[model_type, prompt, resolution, seed, num_steps, num_images],
            outputs=[output_image, seed_used]
        )
        
        open_folder_btn.click(
            fn=open_outputs_folder,
            inputs=[],
            outputs=[gr.Textbox(visible=False)]
        )

    demo.launch(inbrowser=True, share=args.share)
