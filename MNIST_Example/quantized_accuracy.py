import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import sys
from collections import OrderedDict
import numpy as np

# --- Configuration ---
MODEL_PATH_ORIGINAL = "mnist_model_fp32_baseline.pth"
MODEL_PATH_QUANTIZED = "mnist_model_int4_w8x8.pth"
DATA_DIR = "dataset"
BATCH_SIZE = 64
TILE_SIZE = 8
TILE_ELEMENTS = TILE_SIZE * TILE_SIZE # 64
TILE_BYTES = TILE_ELEMENTS // 2 # 32

# Ensure parent directory is in path to import dataset/model
# We will use the SimpleDenseNet class definition directly here
# to make the accuracy script self-contained.

# -------------------------------------------------------------------
# 1. Model Definition (Copied from model.py)
# -------------------------------------------------------------------
class SimpleDenseNet(nn.Module):
    """
    A simple fully connected (dense) neural network for MNIST classification.
    Layer structure: 784 -> 300 -> 70 -> 10.
    """
    def __init__(self):
        super(SimpleDenseNet, self).__init__()
        
        # 28*28 = 784 input features (flattened image)
        self.fc1 = nn.Linear(784, 300)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(300, 70)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(70, 10) # 10 output classes (digits 0-9)
        
    def forward(self, x):
        # x shape: (batch_size, 784)
        
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        # Output layer (no activation, CrossEntropyLoss handles logits)
        x = self.fc3(x) 
        
        return x

# -------------------------------------------------------------------
# 2. Data Loading (Adapted from main.py and dataset.py)
# -------------------------------------------------------------------
def load_test_data():
    """Loads the MNIST test dataset using the same transforms as training."""
    
    # Data transforms for preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Ensure images are single channel
        transforms.Resize((28, 28)),                
        transforms.ToTensor(),                      # Convert PIL Image to Tensor (float32, [0, 1])
        transforms.Normalize((0.5,), (0.5,))        # Normalize to [-1, 1]
    ])

    try:
        # PyTorch's ImageFolder handles the directory structure (class names are subdirectories)
        test_dataset = ImageFolder(root=os.path.join(DATA_DIR, "test"), transform=transform)
    except FileNotFoundError:
        print(f"Error: Could not find data in {DATA_DIR}/test. Please ensure the dataset has been prepared.")
        sys.exit(1)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    return test_loader

# -------------------------------------------------------------------
# 3. De-Quantization Logic
# -------------------------------------------------------------------

def unpack_int4_tile(packed_data_segment, num_elements=TILE_ELEMENTS):
    """
    Unpacks a segment of UINT8 data back into signed INT4 values [-8, 7].
    
    Returns a 1D tensor of signed INT4 values.
    """
    # 1. Unpack into unsigned 4-bit values (0-15)
    low_nibbles = packed_data_segment & 0x0F
    high_nibbles = (packed_data_segment >> 4) & 0x0F
    
    # Interleave the results: [low0, high0, low1, high1, ...]
    unpacked_uint4 = torch.stack((low_nibbles, high_nibbles), dim=1).flatten()
    
    # Trim to the exact number of elements expected (always 64 for 8x8 tile)
    unpacked_uint4 = unpacked_uint4[:num_elements]
    
    # 2. Convert unsigned 4-bit [0, 15] back to signed INT4 [-8, 7]
    W_int = (unpacked_uint4.to(torch.int8) - 8)

    return W_int

def dequantize_state_dict(quantized_state):
    """
    Converts a state dict containing packed INT4 weights and scales
    back into a standard PyTorch state dict with FP32 weights.
    
    CRITICAL FIX: This now iterates tile-by-tile, slices the 1D packed
    tensor, unpacks it, and places the resulting 8x8 FP32 tile into the
    correct position in the final dense matrix. This avoids the faulty
    1D reshape step that caused scrambling.
    """
    dequantized_state = OrderedDict()
    weight_comparisons = []
    
    # Load the original FP32 state dict for comparison purposes
    try:
        original_state = torch.load(MODEL_PATH_ORIGINAL, map_location='cpu')
    except FileNotFoundError:
        print(f"Error: FP32 model '{MODEL_PATH_ORIGINAL}' not found. Cannot perform weight comparison.")
        original_state = {}

    for key, value in quantized_state.items():
        
        if key.endswith('.packed_int4'):
            # The full key is like 'fc1.weight.packed_int4'
            base_key = key.replace('.packed_int4', '') 
            final_weight_key = base_key
            
            # Check if all required metadata is present
            scales_key = f'{base_key}.scales'
            original_shape_key = f'{base_key}.original_shape'
            unpadded_transposed_shape_key = f'{base_key}.unpadded_transposed_shape'

            if not all(k in quantized_state for k in [scales_key, original_shape_key, unpadded_transposed_shape_key]):
                print(f"Error: Missing metadata for {base_key}")
                continue # Skip this layer
                
            # Fetch corresponding scales and shape info
            scales = quantized_state[scales_key]
            original_shape = quantized_state[original_shape_key].tolist()
            R_in_unpadded, C_out_unpadded = quantized_state[unpadded_transposed_shape_key].tolist()
            
            # We must calculate the padded shape
            R_in_padded = ((R_in_unpadded + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
            C_out_padded = ((C_out_unpadded + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
            
            num_tiles_r = R_in_padded // TILE_SIZE
            num_tiles_c = C_out_padded // TILE_SIZE
            
            # Initialize the final padded, transposed (In x Out) FP32 tensor
            W_dequant_transposed_padded = torch.zeros(R_in_padded, C_out_padded, dtype=torch.float32)
            
            # Value is the 1D packed INT4 (uint8) tensor
            packed_weights = value 
            
            byte_offset = 0
            scale_idx = 0
            
            # --- TILE-BY-TILE ASSEMBLY ---
            for i in range(num_tiles_r): # Iterate over In_Features dimension (rows)
                r_start, r_end = i * TILE_SIZE, (i + 1) * TILE_SIZE
                for j in range(num_tiles_c): # Iterate over Out_Features dimension (columns)
                    
                    c_start, c_end = j * TILE_SIZE, (j + 1) * TILE_SIZE
                    
                    # 1. Slice the packed 1D tensor to get the current tile's data (32 bytes)
                    packed_data_segment = packed_weights[byte_offset:byte_offset + TILE_BYTES]
                    
                    # 2. Unpack the 32 bytes into 64 signed INT4 values
                    W_int_1d = unpack_int4_tile(packed_data_segment, TILE_ELEMENTS)
                    
                    # 3. Reshape the 1D signed INT4 into the 8x8 tile
                    W_int_tile = W_int_1d.reshape(TILE_SIZE, TILE_SIZE) 
                    
                    # 4. Dequantize: Q * S
                    S = scales[scale_idx]
                    W_dequant_tile = W_int_tile.to(torch.float32) * S
                    
                    # 5. Place the 8x8 de-quantized tile into the correct position
                    W_dequant_transposed_padded[r_start:r_end, c_start:c_end] = W_dequant_tile
                    
                    byte_offset += TILE_BYTES
                    scale_idx += 1
            
            # 6. Slice the padded tensor back to its original (unpadded) transposed shape (In x Out)
            W_dequant_transposed_unpadded = W_dequant_transposed_padded[:R_in_unpadded, :C_out_unpadded]
            
            # 7. Transpose from (In x Out) back to PyTorch's (Out x In)
            W_dequant_original = W_dequant_transposed_unpadded.T.contiguous()
            
            # Verify shape
            if list(W_dequant_original.shape) != original_shape:
                raise ValueError(f"Shape mismatch after dequantization for {final_weight_key}. Expected {original_shape}, got {list(W_dequant_original.shape)}")

            # Add the reconstructed FP32 weight tensor to the new state dict
            dequantized_state[final_weight_key] = W_dequant_original
            
            print(f"  [Dequantized] {final_weight_key}: Reconstructed shape {list(W_dequant_original.shape)}")
            
            # --- DEBUGGING STEP (Requested by user) ---
            if final_weight_key in original_state:
                W_orig = original_state[final_weight_key].to(W_dequant_original.device)
                
                # IMPORTANT: Ensure the original weight tensor is the same size before comparison
                if W_orig.shape != W_dequant_original.shape:
                     print(f"  Warning: Original shape {W_orig.shape} != Dequant shape {W_dequant_original.shape}. Skipping comparison.")
                else:
                    diff = torch.abs(W_orig - W_dequant_original)
                    
                    max_abs_diff = diff.max().item()
                    mean_abs_diff = diff.mean().item()
                    
                    weight_comparisons.append((final_weight_key, max_abs_diff, mean_abs_diff))

        elif key.endswith(('.bias')):
            # Bias is already FP32, just copy it
            dequantized_state[key] = value
        
        # Ignore scale/shape tensors as they were used above or are intermediate data

    # Print debugging comparison results
    print("\n--- DEBUG: Weight Difference Analysis (Dequantized vs. Original) ---")
    if weight_comparisons:
        print(f"{'Layer':<15} | {'Max Abs Diff (L-inf)':<20} | {'Mean Abs Diff (L1)':<20}")
        print("-" * 58)
        for name, max_diff, mean_diff in weight_comparisons:
            print(f"{name:<15} | {max_diff:<20.8f} | {mean_diff:<20.8f}")
    else:
        print("No weight layers found for comparison.")

    return dequantized_state

# -------------------------------------------------------------------
# 4. Evaluation Logic (Adapted from main.py)
# -------------------------------------------------------------------

def evaluate_model(model, test_loader, device):
    """Evaluates the model on the test set and returns accuracy."""
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad(): # Disable gradient calculation
        for images, labels in test_loader:
            
            images = images.to(device) # Move to device
            images = images.view(images.size(0), -1) # Flatten to (batch_size, 784)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # Get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# -------------------------------------------------------------------
# 5. Main Execution
# -------------------------------------------------------------------

def main():
    
    # 1. Setup Device and Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    test_loader = load_test_data()

    # 2. --- Evaluate FP32 Baseline Model ---
    try:
        print("\n--- 1. Evaluating FP32 Baseline Model ---")
        model_fp32 = SimpleDenseNet().to(device)
        state_fp32 = torch.load(MODEL_PATH_ORIGINAL, map_location=device)
        model_fp32.load_state_dict(state_fp32)
        
        acc_fp32 = evaluate_model(model_fp32, test_loader, device)
        print(f"Original FP32 Model Accuracy: {acc_fp32:.4f}%")
        
    except FileNotFoundError:
        print(f"Error: FP32 model '{MODEL_PATH_ORIGINAL}' not found. Please run main.py first to train the baseline.")
        return
    except Exception as e:
        print(f"An error occurred while loading/evaluating FP32 model: {e}")
        return

    # 3. --- Evaluate De-Quantized Model ---
    acc_dequantized = None
    try:
        print("\n--- 2. Evaluating De-Quantized (INT4 -> FP32) Model ---")
        
        # a. Load Quantized State
        quantized_state = torch.load(MODEL_PATH_QUANTIZED, map_location='cpu')
        
        # b. De-Quantize Weights
        dequantized_state = dequantize_state_dict(quantized_state)
        
        # c. Load De-Quantized Weights into a fresh FP32 model
        model_dequantized = SimpleDenseNet().to(device)
        model_dequantized.load_state_dict(dequantized_state) 
        
        # d. Evaluate
        acc_dequantized = evaluate_model(model_dequantized, test_loader, device)
        print(f"\nDe-Quantized INT4 Model Accuracy: {acc_dequantized:.4f}%")
        
    except FileNotFoundError:
        print(f"Error: Quantized model '{MODEL_PATH_QUANTIZED}' not found. Please run quantizer.py first.")
        return
    except (KeyError, ValueError) as e:
        print(f"An error occurred during dequantization or loading: {e}")
        print("This usually means the dequantization function produced an incorrect key or shape.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading/evaluating De-Quantized model: {e}")
        return
        
    # 4. --- Comparison ---
    if acc_dequantized is not None:
        print("\n--- 3. Comparison ---")
        loss_percentage = 100 * (acc_fp32 - acc_dequantized) / acc_fp32
        absolute_drop = acc_fp32 - acc_dequantized
        
        print(f"Absolute Accuracy Drop: {absolute_drop:.4f} percentage points")
        print(f"Relative Accuracy Loss: {loss_percentage:.2f}% (of the original accuracy)")

if __name__ == "__main__":
    main()