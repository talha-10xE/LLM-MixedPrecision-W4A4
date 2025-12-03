import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" 
PROMPT = "India vs USA vs CHina? which country is better?"
SAVE_DIR = "./local_qwen_model" 

# --- Check Device and Setup ---
if torch.cuda.is_available():
    # DTYPE is the target precision for computation (Float16)
    DTYPE = torch.float16 
    device = torch.device("cuda")
    print(f"✅ CUDA available. Using device: {torch.cuda.get_device_name(0)}")
    print(f"Model will be loaded in: {DTYPE}")
else:
    DTYPE = torch.float32
    device = torch.device("cpu")
    print("❌ CUDA not available. Falling back to CPU. Performance will be slower.")
    print("Model will be loaded in: torch.float32")


# --- 1. Quantization Helper Function ---
def quantize_per_channel_int8_no_zeropoint(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes a weight tensor to Int8 (symmetric, no zero point) per output channel (dim 0).
    The scales are initialized as float32 on CPU for demonstration purposes.
    """
    
    # 1. Calculate max absolute value per output channel
    max_vals, _ = torch.abs(weight).max(dim=1, keepdim=True)
    
    # 2. Calculate scales: S = max_val / 127. 
    # CRUCIAL CHANGE: We explicitly cast to float32 here and keep it on CPU 
    # to demonstrate the float32 -> float16 conversion later.
    scale = (max_vals / 127.0).clamp(min=1e-8).to(torch.float32)

    # 3. Quantize weight: W_int8 = round(W_float32 / S_float32)
    # The intermediate computation uses float32, then casts the result to int8.
    quant_weight = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)

    # Reshape scale back to a 1D vector [out_features] for simpler storage/casting.
    return quant_weight, scale.squeeze(1)


# --- 2. Custom Quantized Linear Layer Definition ---
class CustomLMHead(nn.Module):
    """
    A custom final linear layer that stores weights in Int8 and dequantizes 
    them to Float16 (DTYPE) at runtime for computation.
    """
    def __init__(self, in_features, out_features, quant_weight, scales):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Store quantized weight as an Int8 tensor (1 byte/param)
        self.quant_weight = nn.Parameter(quant_weight, requires_grad=False)
        
        # Store scales as a DTYPE (Float32 initially) tensor (4 bytes/param)
        self.scales = nn.Parameter(scales, requires_grad=False)
        
        print(f"CustomLMHead created: in={in_features}, out={out_features}, stored as Int8/Float32 (init).")

    def forward(self, x):
        # 1. Dequantize the weights to the input DTYPE (float16)
        # W_dequant = W_int8_float16 * S_float16
        # .to(x.dtype) casts the Int8 tensor to Float16 for the computation
        # The scale must also be cast to the input DTYPE (x.dtype which is float16)
        dequantized_weight = self.quant_weight.to(x.dtype) * self.scales.to(x.dtype).view(-1, 1)
        
        # 2. Perform the linear computation.
        return F.linear(x, dequantized_weight, None)

# --- 3. Verification Function ---
def print_layer_info(layer: CustomLMHead, stage: str):
    """Prints the device, dtype, and size of the internal parameters."""
    print(f"\n===== PARAMETER CHECK: {stage} =====")
    
    # Check the quantized weight (should be Int8)
    q_weight = layer.quant_weight
    q_size = q_weight.numel() * q_weight.element_size() / (1024*1024)
    print(f"quant_weight (Weights):")
    print(f"  - Device: {q_weight.device}")
    print(f"  - DType:  {q_weight.dtype} (Expected: torch.int8, should NOT change)")
    print(f"  - Memory: {q_size:.2f} MB")

    # Check the scales (should be Float32 before, Float16 after)
    scales = layer.scales
    s_size = scales.numel() * scales.element_size() / (1024*1024)
    expected_dtype = "torch.float32" if stage.startswith("BEFORE") and scales.dtype == torch.float32 else str(DTYPE)
    print(f"scales (Scales):")
    print(f"  - Device: {scales.device}")
    print(f"  - DType:  {scales.dtype} (Expected: {expected_dtype})")
    print(f"  - Memory: {s_size:.6f} MB")
    print("===================================")


# --- 4. Download and Load Model & Tokenizer ---
print(f"\n🚀 Loading model '{MODEL_ID}'...")
start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",
    low_cpu_mem_usage=True
)
# Ensure the entire model is on the device
model.to(device)

load_time = time.time() - start_time
print(f"✅ Model loaded successfully in {load_time:.2f} seconds.")

# --- 5. Quantize and Replace the lm_head layer ---

original_lm_head = model.lm_head
original_weights = original_lm_head.weight.data.cpu().float() # Move to CPU and float32 for clean quantization
in_features = original_lm_head.in_features
out_features = original_lm_head.out_features

# Quantize the weights
print("\n🛠️ Quantizing original weights to Int8 (scales kept in Float32 on CPU)...")
quant_weight, scales = quantize_per_channel_int8_no_zeropoint(original_weights)

# Create the custom layer instance
custom_layer = CustomLMHead(in_features, out_features, quant_weight, scales)
# IMPORTANT: The custom layer and its parameters start on CPU/Int8/Float32.

# --- VERIFICATION STEP 1: BEFORE .to() ---
print_layer_info(custom_layer, "BEFORE .to(device, dtype)")

# --- VERIFICATION STEP 2: Apply .to() ---
print(f"\n➡️ Applying custom_layer.to(device={device}, dtype={DTYPE})...")
# This is the crucial step: moving tensors to device and applying dtype change only to floats.
custom_layer.to(device=device, dtype=DTYPE) 

# --- VERIFICATION STEP 3: AFTER .to() ---
print_layer_info(custom_layer, "AFTER .to(device, dtype)")

# --- 6. Final Setup and Inference ---

# Replace the model's layer with the custom instance
model.lm_head = custom_layer 

print("\n--- Model Architecture AFTER Quantization Surgery ---")
print(model.lm_head)
print("----------------------------------------")


# Run Inference (Generate Text)
# Prepare Input for Inference (Tokenize)
messages = [
    {"role": "system", "content": "You are a helpful and concise assistant."},
    {"role": "user", "content": PROMPT}
]
input_text = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)


print(f"\n🧠 Generating response to prompt: '{PROMPT}'")
start_time = time.time()

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode and Print Output
generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
generation_time = time.time() - start_time

print("--- Generated Response ---")
print(generated_text)
print("--------------------------")
print(f"⚡ Generation complete in {generation_time:.2f} seconds.")
print(f"Tokens/Second: {len(output_ids[0]) / generation_time:.2f}")