import torch
import gradio as gr
import logging

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


def gen_img_helper(model, llama_model, prompt, res, seed):
    global pipe, current_model, current_llama_model

    # Check if the model or LLaMA model has changed
    if model != current_model or llama_model != current_llama_model:
        print(f"Unloading model {current_model}...")
        del pipe
        torch.cuda.empty_cache()
        
        print(f"Loading model {model} with LLaMA model {llama_model}...")
        pipe, _ = load_models(model, llama_model)
        current_model = model
        current_llama_model = llama_model
        print("Model loaded successfully!")

    # 2. Generate image
    res = parse_resolution(res)
    return generate_image(pipe, model, prompt, res, seed)


if __name__ == "__main__":
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    # Initialize with default model
    print("Loading default model (fast)...")
    current_model = "fast"
    current_llama_model = "int4"
    pipe, _ = load_models(current_model, current_llama_model)
    print("Model loaded successfully!")

    # Create Gradio interface
    with gr.Blocks(title="HiDream-I1-nf4 Dashboard") as demo:
        gr.Markdown("# HiDream-I1-nf4 Dashboard")
        
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    model_type = gr.Radio(
                        choices=list(MODEL_CONFIGS.keys()),
                        value="fast",
                        label="Model Type",
                        info="Select model variant"
                    )
                    
                    llama_model = gr.Radio(
                        choices=["int4", "int8"],
                        value="int4",
                        label="LLaMA Model",
                        info="Select which LLaMA model to use: INT4 (hugging-quants) or INT8 (clowman)"
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
                
                generate_btn = gr.Button("Generate Image")
                seed_used = gr.Number(label="Seed Used", interactive=False)
                
            with gr.Column():
                output_image = gr.Image(label="Generated Image", type="pil")
        
        generate_btn.click(
            fn=gen_img_helper,
            inputs=[model_type, llama_model, prompt, resolution, seed],
            outputs=[output_image, seed_used]
        )

    demo.launch()
