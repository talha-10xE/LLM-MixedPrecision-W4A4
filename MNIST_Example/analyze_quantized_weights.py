import torch
import numpy as np
from collections import OrderedDict
import os

# --- Configuration ---
MODEL_PATH_ORIGINAL = "mnist_model_fp32_baseline.pth"
MODEL_PATH_QUANTIZED = "mnist_model_int4_w8x8.pth"
TILE_SIZE = 8

def unpack_int4_tile(packed_data_segment, num_elements):
    """
    Unpacks a segment of UINT8 data back into signed INT4 values.
    """
    # 1. Unpack into unsigned 4-bit values (0-15)
    
    # Extract low nibble
    low_nibbles = packed_data_segment & 0x0F
    # Extract high nibble
    high_nibbles = (packed_data_segment >> 4) & 0x0F
    
    # Interleave the results: [low0, high0, low1, high1, ...]
    unpacked_uint4 = torch.stack((low_nibbles, high_nibbles), dim=1).flatten()
    
    # Trim to the exact number of elements expected 
    unpacked_uint4 = unpacked_uint4[:num_elements]
    
    # 2. Convert unsigned 4-bit [0, 15] back to signed INT4 [-8, 7]
    # W_int = W_uint4 - 8 (8 is the offset used in packing)
    W_int = (unpacked_uint4.to(torch.int8) - 8)

    return W_int

def analyze_quantized_model():
    """
    Loads the quantized model, prints tensor details, and verifies the 
    quantization/packing process for a sample tile against the original FP32 weights.
    """
    try:
        # Load the quantized model
        quantized_state = torch.load(MODEL_PATH_QUANTIZED, map_location='cpu')
        print(f"--- Loaded Custom Quantized Model from: {MODEL_PATH_QUANTIZED} ---")
    except FileNotFoundError:
        print(f"Error: Quantized model file '{MODEL_PATH_QUANTIZED}' not found.")
        print("Please ensure you have run quantizer.py successfully.")
        return
    except Exception as e:
        print(f"An error occurred while loading the quantized model: {e}")
        return

    try:
        # Load the original FP32 model for comparison
        original_state = torch.load(MODEL_PATH_ORIGINAL, map_location='cpu')
        print(f"--- Loaded Original FP32 Model from: {MODEL_PATH_ORIGINAL} ---")
    except FileNotFoundError:
        print(f"Error: Original model file '{MODEL_PATH_ORIGINAL}' not found.")
        return
    except Exception as e:
        print(f"An error occurred while loading the original model: {e}")
        return


    print("\n" + "="*80)
    print("STRUCTURED ANALYSIS & QUANTIZATION ERROR VERIFICATION")
    print("="*80)
    
    # Keep track of the name required to fetch the original weight (e.g., 'fc1')
    first_weight_base_name = None 

    # --- Print Shapes and Contents (Unchanged from previous version for context) ---
    for name in quantized_state.keys():
        if name.endswith('.packed_int4') and first_weight_base_name is None:
            # The quantized key is 'fc1.weight.packed_int4'.
            # We want to extract 'fc1' to find 'fc1.weight' in the original state.
            try:
                # Find the index of '.weight.' and extract the part before it
                base_name = name.split('.weight.')[0] 
                first_weight_base_name = base_name
            except:
                # Fallback if the name format is unexpected
                pass
            
        
        # This part of the code prints the analysis output
        if name.endswith('.packed_int4'):
            print(f"\n[WEIGHTS] {name}")
            tensor = quantized_state[name]
            print(f"  -> Shape: {tuple(tensor.shape)}")
            print(f"  -> Type: {tensor.dtype} (UINT8, holding 2 INT4 values per byte)")
            print(f"  -> Content (First 16 bytes): {tensor[:16].tolist()}...")

        elif name.endswith('.scales'):
            print(f"\n[SCALES] {name}")
            tensor = quantized_state[name]
            print(f"  -> Shape: {tuple(tensor.shape)}")
            print(f"  -> Type: {tensor.dtype} (FP32)")
            print(f"  -> Content (First 8 scales): [{', '.join(f'{s:.6f}' for s in tensor[:8].tolist())}, ...]")
            print(f"  -> Scale Count: {tensor.numel()} scales (one per 8x8 tile)")

        elif name.endswith('.bias'):
            print(f"\n[BIAS] {name}")
            tensor = quantized_state[name]
            print(f"  -> Shape: {tuple(tensor.shape)}")
            print(f"  -> Type: {tensor.dtype} (FP32)")
            print(f"  -> Content (First 8): [{', '.join(f'{s:.3f}' for s in tensor[:8].tolist())}, ...]")

        elif name.endswith('.original_shape') or name.endswith('.transposed_shape'):
            print(f"\n[SHAPE INFO] {name}")
            tensor = quantized_state[name]
            print(f"  -> Shape: {tuple(tensor.shape)}")
            print(f"  -> Value: {tensor.tolist()}")

        else:
            print(f"\n[OTHER] {name} (Shape: {tuple(quantized_state[name].shape)})")

if __name__ == "__main__":
    # Temporarily set torch print options for better output readability
    torch.set_printoptions(precision=4, sci_mode=False)
    analyze_quantized_model()