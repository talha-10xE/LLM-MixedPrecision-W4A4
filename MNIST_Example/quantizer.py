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
    - unpadded_shape: (tuple) The original, unpadded shape (R_in, C_out) of the transposed tensor.
    """
    R_in, C_out = W_fp32.shape
    
    # --- EDGE CASE FIX: Zero-Padding ---
    # Calculate the required padded dimensions (next multiple of TILE_SIZE)
    R_padded = ((R_in + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    C_padded = ((C_out + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE

    if R_padded != R_in or C_padded != C_out:
        # Create a new tensor initialized to zeros for padding
        W_padded = torch.zeros((R_padded, C_padded), dtype=W_fp32.dtype, device=W_fp32.device)
        # Copy the original weights into the top-left corner
        W_padded[:R_in, :C_out] = W_fp32
    else:
        W_padded = W_fp32

    # --- Quantization Parameters ---
    R_p, C_p = W_padded.shape
    num_tiles_r = R_p // TILE_SIZE
    num_tiles_c = C_p // TILE_SIZE

    # Pre-allocate lists for results
    int4_tiles = []
    scale_factors = []

    # Iterate over 8x8 tiles
    for i in range(num_tiles_r):
        for j in range(num_tiles_c):
            r_start, r_end = i * TILE_SIZE, (i + 1) * TILE_SIZE
            c_start, c_end = j * TILE_SIZE, (j + 1) * TILE_SIZE
            
            # Extract the 8x8 tile
            W_tile = W_padded[r_start:r_end, c_start:c_end]
            
            # 1. Determine the maximum range for the tile (using 99th percentile clipping)
            # Find the absolute value at the 99th percentile
            abs_W_tile = torch.abs(W_tile)
            
            # Using torch.quantile is safer than a manual sort for 99th percentile
            # We use the max abs value if the tile is all zero (to prevent S=0/NaN/Inf scale)
            # but since we are padding with zeros, a non-zero tile size is guaranteed for non-zero weights.
            max_val = torch.quantile(abs_W_tile.flatten(), CLIP_PERCENTILE / 100.0)
            
            # Handle the case where the max_val is extremely small (near zero)
            # Set a minimum clipping value to avoid division by zero or massive quantization error
            max_val = max(max_val, 1e-6) 
            
            # 2. Calculate the scale factor (S)
            # S = Max_Value / Max_Quantization_Range
            S = max_val / INT4_MAX
            
            # 3. Quantize the FP32 tile to signed INT4
            # Q = round(W / S)
            Q_tile = torch.round(W_tile / S).to(torch.int8)
            
            # 4. Clip the quantized values to the target range [-8, 7]
            Q_tile = torch.clamp(Q_tile, -INT4_MAX - 1, INT4_MAX) # Clips to [-8, 7]
            
            # 5. Pack the signed INT4 into UINT8
            # Convert signed INT4 [-8, 7] to unsigned 4-bit [0, 15] by adding offset 8
            W_uint4 = (Q_tile + 8).to(torch.uint8)
            
            # Pack two 4-bit values into one 8-bit byte
            # Group every two elements: [a, b, c, d, ...] -> [(b << 4) | a, (d << 4) | c, ...]
            # The indices for pairing are [0, 1], [2, 3], etc.
            W_uint4_flat = W_uint4.flatten()
            
            # Check if total number of elements is even
            if W_uint4_flat.numel() % 2 != 0:
                # This should only happen if the padded size is uneven, which is impossible with TILE_SIZE=8.
                # However, for safety, we pad with one more zero if needed.
                W_uint4_flat = torch.cat((W_uint4_flat, torch.tensor([0], dtype=torch.uint8, device=W_padded.device)))
            
            low_nibbles = W_uint4_flat[::2]
            high_nibbles = W_uint4_flat[1::2]
            
            packed_byte = (high_nibbles << 4) | low_nibbles
            
            # Store results
            int4_tiles.append(packed_byte)
            scale_factors.append(S)

    # Concatenate all tiles into single tensors
    packed_weights = torch.cat(int4_tiles).contiguous()
    scale_factors = torch.tensor(scale_factors, dtype=torch.float32)

    return packed_weights, scale_factors, (R_in, C_out) # Return UNPADDED shape

def quantize_model(model_path_in, model_path_out):
    """
    Loads the FP32 model, quantizes its weight tensors, and saves the custom state dict.
    """
    print(f"--- Quantizing Model Weights: {model_path_in} -> {model_path_out} ---")
    
    try:
        # Load the FP32 state dict (CPU is sufficient for quantization)
        state_dict = torch.load(model_path_in, map_location='cpu')
    except FileNotFoundError:
        print(f"Error: FP32 baseline model not found at {model_path_in}. Please run main.py first.")
        return

    quantized_state = OrderedDict()
    
    # Iterate through all parameters in the state dict
    for name, W_fp32 in state_dict.items():
        
        if name.endswith('.weight'):
            print(f"Processing layer: {name} (Original shape: {list(W_fp32.shape)})")
            
            # 1. Transpose the weight matrix: PyTorch is (Out x In), we quantize (In x Out)
            W_transposed = W_fp32.T
            
            # 2. Quantize and pack the data
            packed_weights, scale_factors, unpadded_shape = symmetric_quantize_and_pack(W_transposed)
            
            # Store the packed data and metadata
            quantized_state[f'{name}.packed_int4'] = packed_weights
            quantized_state[f'{name}.scales'] = scale_factors
            
            # Store the original/final shape for the model layer (Out x In)
            quantized_state[f'{name}.original_shape'] = torch.tensor(list(W_fp32.shape))
            
            # CRITICAL FIX: Store the UNPADDED transposed shape (In x Out) for dequantization slicing
            quantized_state[f'{name}.unpadded_transposed_shape'] = torch.tensor(unpadded_shape)

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
    print(f"\nSuccessfully saved quantized model to {MODEL_PATH_OUT}")

if __name__ == '__main__':
    quantize_model(MODEL_PATH_IN, MODEL_PATH_OUT)