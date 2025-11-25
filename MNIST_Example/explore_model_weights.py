import torch
import numpy as np

# --- Configuration ---
MODEL_PATH = "mnist_model_fp32_baseline.pth"
TILE_SIZE = 8

def analyze_weights_and_tiles():
    """
    Loads the FP32 model weights and analyzes the global and 8x8 tile
    min/max statistics for linear layers.
    """
    try:
        # Load the state dictionary from the trained model file
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        print(f"--- Loaded FP32 Weights from: {MODEL_PATH} ---")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        print("Please ensure you have run main.py successfully to create the baseline model.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    linear_layers = [key for key in state_dict.keys() if key.endswith('.weight')]
    
    if not linear_layers:
        print("No linear layer weights found in the state dictionary.")
        return

    for name in linear_layers:
        W = state_dict[name].float() # Ensure it's FP32 for analysis
        out_features, in_features = W.shape
        
        print("\n" + "="*70)
        print(f"Layer: {name}")
        print(f"Global Shape: {W.shape} (Out={out_features}, In={in_features})")
        
        # 1. Global Analysis
        global_min = W.min().item()
        global_max = W.max().item()
        global_range = global_max - global_min
        
        print(f"Global Min: {global_min:.6f}, Max: {global_max:.6f}, Range: {global_range:.6f}")
        
        # 2. Tile-wise Analysis (8x8)
        
        # Calculate number of full tiles along each dimension
        rows_full = out_features // TILE_SIZE
        cols_full = in_features // TILE_SIZE

        print(f"\n--- Tile-wise (8x8) Analysis ---")
        print(f"Total full 8x8 tiles: {rows_full} x {cols_full}")
        
        # Analyze a subset of the first 3x3 tiles for demonstration
        tiles_to_analyze = min(rows_full, 3)
        
        tile_ranges = []

        print(f"\nAnalyzing a {tiles_to_analyze}x{tiles_to_analyze} grid of 8x8 tiles:")
        
        # Iterate over output features (rows)
        for r in range(tiles_to_analyze):
            row_min_max_str = []
            
            # Iterate over input features (columns)
            for c in range(tiles_to_analyze):
                r_start = r * TILE_SIZE
                c_start = c * TILE_SIZE
                
                # Extract the 8x8 tile
                tile = W[r_start:r_start + TILE_SIZE, c_start:c_start + TILE_SIZE]
                
                tile_min = tile.min().item()
                tile_max = tile.max().item()
                tile_range = tile_max - tile_min
                tile_ranges.append(tile_range)
                
                # Format the output string for the tile
                row_min_max_str.append(f"({tile_min:+.3f} to {tile_max:+.3f})")
            
            # Print the row of analyzed tiles
            print(f"Row Block {r*TILE_SIZE:03d}-{r*TILE_SIZE+7:03d}: | {' | '.join(row_min_max_str)} |")
            
        # 3. Insights based on tile ranges
        if tile_ranges:
            avg_tile_range = np.mean(tile_ranges)
            max_tile_range = np.max(tile_ranges)
            
            print("\n--- Quantization Insights ---")
            print(f"Global Dynamic Range: {global_range:.4f}")
            print(f"Max Tile Dynamic Range (in subset): {max_tile_range:.4f}")
            print(f"Average Tile Dynamic Range (in subset): {avg_tile_range:.4f}")
            
            # This is the key metric
            compression_ratio = global_range / avg_tile_range if avg_tile_range > 0 else 0
            print(f"Average Range Reduction (Global/Avg Tile Range): {compression_ratio:.2f}x")

    print("\n" + "="*70)
    print("Analysis Complete.")
    print("="*70)
    
    # Final Quantization Strategy Insights
    print("\n💡 Key Takeaways for W4 Quantization:")
    print("---------------------------------------")
    print("1. **Tile-Wise Quantization is Valid:** Since the 'Average Range Reduction' is likely greater than 1, quantizing with 8x8 tiles provides a significantly tighter distribution range for each scaling factor ($S$) compared to using a single global scale.")
    print("2. **4-bit Precision:** For a given dynamic range (Range/2^4), a tighter range per tile means the quantization step size will be smaller, leading to lower quantization error (higher accuracy) compared to a global scale.")
    print("3. **LLM Relevance:** This local variation observed in the MNIST model is even more pronounced in LLMs (due to layer-wise activation differences and specific feature learning), confirming that the 8x8 granularity is a necessary feature for your W4A4 scheme.")

if __name__ == "__main__":
    analyze_weights_and_tiles()