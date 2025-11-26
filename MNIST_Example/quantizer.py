import torch
import numpy as np
from collections import OrderedDict

# --- Configuration ---
MODEL_PATH_IN = "mnist_model_fp32_baseline.pth"
MODEL_PATH_OUT = "mnist_model_int4_w8x8.pth"
TILE_SIZE = 8
INT4_MAX = 7 # Symmetric signed INT4 range is [-8, 7]
CLIP_PERCENTILE = 99.0 # Use 99th percentile for outlier clipping

def symmetric_quantize_and_pack(W_fp32):
    """
    Applies 8x8 symmetric quantization with 99th percentile clipping to the 
    weight matrix W_fp32.
    
    The weight matrix is assumed to be PRE-TRANSPOSED (In_Features, Out_Features).
    
    Returns:
    - packed_int4_weights: (torch.uint8) Packed INT4 data (2 INT4 values per byte).
    - scale_factors: (torch.float32) Scale factors for each 8x8 tile.
    """
    R_in, C_out = W_fp32.shape
    
    # --- EDGE CASE FIX: Zero-Padding ---
    # Calculate the required padded dimensions (next multiple of TILE_SIZE)
    R_padded = ((R_in + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    C_padded = ((C_out + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE

    if R_padded != R_in or C_padded != C_out:
        # Create a new tensor initialized to zeros for padding
        W_padded = torch.zeros(R_padded, C_padded, dtype=W_fp32.dtype)
        # Copy the original weights into the top-left corner
        W_padded[:R_in, :C_out] = W_fp32
        W_to_quantize = W_padded
        print(f"  -> Padded from ({R_in}, {C_out}) to ({R_padded}, {C_padded}) for tiling.")
    else:
        W_to_quantize = W_fp32
    
    # Recalculate tile counts based on padded dimensions
    num_tiles_r = R_padded // TILE_SIZE
    num_tiles_c = C_padded // TILE_SIZE
    
    # Initialize lists to hold the scales and the packed quantized data
    scale_factors = []
    quantized_tiles = []

    # Iterate over the padded weight matrix in 8x8 blocks
    for r_block in range(num_tiles_r):
        for c_block in range(num_tiles_c):
            # Define tile boundaries
            r_start, r_end = r_block * TILE_SIZE, (r_block + 1) * TILE_SIZE
            c_start, c_end = c_block * TILE_SIZE, (c_block + 1) * TILE_SIZE
            
            # W_to_quantize is now guaranteed to have dimensions R_padded x C_padded
            W_tile = W_to_quantize[r_start:r_end, c_start:c_end]
            
            # --- Step 1: Calculate 99th Percentile Clip Threshold (T_99) ---
            # We use the 99th percentile of the absolute values to set the range.
            abs_W = torch.abs(W_tile)
            # Flatten to 1D and use torch.quantile for percentile calculation
            T_99 = torch.quantile(abs_W.flatten(), CLIP_PERCENTILE / 100.0)
            
            # If the calculated threshold is close to zero, prevent division by zero
            if T_99.item() < 1e-6:
                 S = torch.tensor(1.0)
                 T_99 = torch.tensor(0.0) # Ensure no clipping if range is zero
            else:
                 # --- Step 2: Calculate Scale Factor (S) ---
                 # S = T_99 / 7 (since max INT4 is 7 for symmetric)
                 S = T_99 / INT4_MAX 

            scale_factors.append(S.item())
            
            # --- Step 3: Quantize and Clip ---
            # 1. Clip the FP32 tile values to the determined range [-T_99, T_99]
            W_clipped = torch.clamp(W_tile, -T_99, T_99)
            
            # 2. Quantize: W_int = round(W_clipped / S)
            W_int = torch.round(W_clipped / S)
            
            # 3. Clip W_int to the final INT4 range [-8, 7]
            W_int_final = torch.clamp(W_int, -INT4_MAX - 1, INT4_MAX).to(torch.int8)

            quantized_tiles.append(W_int_final.flatten())

    # Concatenate all 8x8 tiles into one long vector of INT4 data
    all_int4_data = torch.cat(quantized_tiles)

    # --- Step 4: Pack the INT4 data into UINT8 (2 INT4 values per byte) ---
    
    # We need to ensure we have an even number of elements for pairing
    if all_int4_data.numel() % 2 != 0:
        # Should not happen if all dimensions are multiples of 8, but for robustness:
        print("Warning: Odd number of elements. Padding with zero.")
        padding = torch.zeros(1, dtype=torch.int8)
        all_int4_data = torch.cat([all_int4_data, padding])

    # INT4 values must be in the range [-8, 7]. We want to pack two of them.
    # W_int has shape (N)
    # We pair (W_int[0], W_int[1]), (W_int[2], W_int[3]), ...
    
    # The packing strategy: (Low nibble, High nibble)
    # Byte = (W_int_1 & 0x0F) | ((W_int_2 & 0x0F) << 4)
    # Since W_int is signed, we need to handle the negative values correctly.
    # The 4-bit representation of -8 is (1000)_2, 7 is (0111)_2.
    
    # 1. Cast to unsigned 4-bit (0-15) for packing convenience
    # Add 8 to shift [-8, 7] to [0, 15].
    W_uint4 = (all_int4_data + 8).to(torch.uint8)

    # 2. Reshape W_uint4 to (N/2, 2)
    W_uint4_paired = W_uint4.view(-1, 2)

    # 3. Pack: low nibble is the first element, high nibble is the second
    packed_weights = (W_uint4_paired[:, 0] & 0x0F) | (W_uint4_paired[:, 1] << 4)
    
    # Reshape scale_factors list to tensor
    scale_tensor = torch.tensor(scale_factors, dtype=torch.float32)
    
    return packed_weights, scale_tensor


def quantize_and_save_model():
    """
    Main function to load the FP32 model, quantize weights using the
    defined symmetric W4-8x8 scheme, and save the result.
    """
    try:
        # Load the state dictionary from the trained model file
        state_dict = torch.load(MODEL_PATH_IN, map_location='cpu')
        print(f"--- Loaded FP32 Weights from: {MODEL_PATH_IN} ---")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH_IN}' not found.")
        print("Please ensure you have generated the baseline model.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # Dictionary to hold the quantized components
    quantized_state = OrderedDict()
    
    print(f"\n--- Quantizing Weights and Saving to: {MODEL_PATH_OUT} ---")

    for name, W_fp32 in state_dict.items():
        if name.endswith('.weight'):
            print(f"Quantizing {name} (Shape: {W_fp32.shape})...")
            
            # --- CRITICAL STEP: Transpose the weight matrix for memory efficiency ---
            # PyTorch: (Out_Features, In_Features) -> Saved: (In_Features, Out_Features)
            W_transposed = W_fp32.T
            
            # Quantize the transposed weights (padding happens inside this function)
            packed_weights, scale_factors = symmetric_quantize_and_pack(W_transposed)
            
            # Save the quantized data and scales
            quantized_state[f'{name}.packed_int4'] = packed_weights
            quantized_state[f'{name}.scales'] = scale_factors
            
            # Also save the original shape and the transposed shape for reference
            # The kernel needs the original shape to know the actual dimensions
            # of the output tensor.
            quantized_state[f'{name}.original_shape'] = torch.tensor(W_fp32.shape)
            quantized_state[f'{name}.transposed_shape'] = torch.tensor(W_transposed.shape)

            # Store biases as FP32 (typically kept at high precision)
            bias_name = name.replace('.weight', '.bias')
            if bias_name in state_dict:
                quantized_state[bias_name] = state_dict[bias_name]
                
            # Log compression efficiency
            original_size = W_fp32.numel() * 4 # FP32 size (4 bytes/element)
            quantized_size = packed_weights.numel() * 1 # INT4 packed size (1 byte/byte)
            scale_size = scale_factors.numel() * 4 # FP32 size (4 bytes/scale)
            
            total_quant_storage = quantized_size + scale_size
            compression = original_size / total_quant_storage
            
            print(f"  -> Scales: {scale_factors.numel()} total scales saved (1 per 8x8 tile).")
            print(f"  -> Weights compressed from {original_size/1024:.2f} KB to {total_quant_storage/1024:.2f} KB ({compression:.2f}x compression ratio).")
            
        elif name.endswith('.bias'):
            # Skip biases here, they are handled with weights above
            continue 
        else:
             # Save other parameters (e.g., batch norm running stats, if they existed)
            quantized_state[name] = W_fp32

    # Save the custom state dictionary
    torch.save(quantized_state, MODEL_PATH_OUT)
    print(f"\n--- Custom Quantized Model Saved Successfully! ---")


if __name__ == "__main__":
    quantize_and_save_model()