from .nf4 import *

import argparse
import time
import logging


if __name__ == "__main__":
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("prompt", type=str, help="Prompt to generate image from")
    
    parser.add_argument("-m", "--model", type=str, default="dev",
                        help="Model to use",
                        choices=["dev", "full", "fast"])
    
    parser.add_argument("-lm", "--llama-model", type=str, default="int4",
                        help="LLaMA model to use",
                        choices=["int4", "int8"])
    
    parser.add_argument("-s", "--seed", type=int, default=-1, 
                        help="Seed for generation")
    
    parser.add_argument("-r", "--res", type=str, default="1024x1024", 
                        help="Resolution for generation", 
                        choices=["1024x1024", "768x1360", "1360x768", "880x1168", "1168x880", "1248x832", "832x1248"])
    
    parser.add_argument("-o", "--output", type=str, default="output.png")
    
    args = parser.parse_args()
    model_type = args.model
    llama_model = args.llama_model
    
    # Initialize with default model
    print(f"Loading model {model_type} with LLaMA model {llama_model}...")
    pipe, _ = load_models(model_type, llama_model)
    print("Model loaded successfully!")
    
    st = time.time()
    
    resolution = tuple(map(int, args.res.strip().split("x")))
    image, seed = generate_image(pipe, model_type, args.prompt, resolution, args.seed)
    image.save(args.output)
    
    print(f"Image saved to {args.output}, elapsed time: {time.time() - st:.2f} seconds")
    print(f"Seed used: {seed}")
