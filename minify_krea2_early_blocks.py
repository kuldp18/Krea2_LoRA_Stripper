#This script drops everything after a certain depth. It defaults to keeping blocks 0 through 10 (11 total blocks), which captures the early concept-generation phase of Krea 2 while shedding the later styling phase.

import argparse
from safetensors.torch import load_file, save_file

def minify_krea2_early_blocks(input_path, output_path, max_block):
    print(f"Loading Krea 2 LoRA from: {input_path}")
    tensors = load_file(input_path)
    
    kept_tensors = {}
    
    for k, v in tensors.items():
        # 1. Preserve text-conditioning layers
        if "txtfusion" in k:
            kept_tensors[k] = v
            continue
            
        # 2. Preserve early structural blocks
        if "diffusion_model.blocks." in k:
            try:
                block_idx = int(k.split("diffusion_model.blocks.")[1].split(".")[0])
                if block_idx <= max_block:
                    kept_tensors[k] = v
            except (IndexError, ValueError):
                continue
                
    print(f"Preserved txtfusion and blocks 0 through {max_block}.")
    save_file(kept_tensors, output_path)
    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minify Krea 2 LoRA via Early Block Isolation")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-m", "--max-block", type=int, default=10, help="Max block to keep (default: 10)")
    
    args = parser.parse_args()
    minify_krea2_early_blocks(args.input, args.output, args.max_block)