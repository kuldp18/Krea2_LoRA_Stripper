#This script analyzes the L2 norm of each block and keeps only the heaviest weights. It defaults to keeping 8 out of 28 blocks, which targets the most mathematically active ~30% of the network.

import argparse
import torch
from safetensors.torch import load_file, save_file

def minify_krea2_top_k(input_path, output_path, keep_top_k):
    print(f"Loading Krea 2 LoRA from: {input_path}")
    tensors = load_file(input_path)
    
    kept_tensors = {}
    block_magnitudes = {}
    
    # 1. Preserve text-conditioning layers
    for k, v in tensors.items():
        if "txtfusion" in k:
            kept_tensors[k] = v
            
        # 2. Calculate L2 norm for spatial blocks
        elif "diffusion_model.blocks." in k:
            try:
                block_idx = int(k.split("diffusion_model.blocks.")[1].split(".")[0])
                if block_idx not in block_magnitudes:
                    block_magnitudes[block_idx] = 0.0
                block_magnitudes[block_idx] += torch.norm(v.float()).item()
            except ValueError:
                continue

    if not block_magnitudes:
        print("Error: No diffusion_model.blocks found. Is this a Krea 2 LoRA?")
        return

    # 3. Filter for Top-K
    sorted_blocks = sorted(block_magnitudes.items(), key=lambda x: x[1], reverse=True)
    top_blocks = [str(idx) for idx, mag in sorted_blocks[:keep_top_k]]
    
    print(f"Top {keep_top_k} active blocks retained: {', '.join(top_blocks)}")
    
    # 4. Extract
    for k, v in tensors.items():
        if "diffusion_model.blocks." in k:
            try:
                block_idx_str = k.split("diffusion_model.blocks.")[1].split(".")[0]
                if block_idx_str in top_blocks:
                    kept_tensors[k] = v
            except IndexError:
                continue
                
    save_file(kept_tensors, output_path)
    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minify Krea 2 LoRA via L2 Norm Analysis")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-k", "--keep", type=int, default=8, help="Top-K blocks to keep (default: 8)")
    
    args = parser.parse_args()
    minify_krea2_top_k(args.input, args.output, args.keep)