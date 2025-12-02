import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" 
PROMPT = "India vs USA vs China? which country is better in terms of living a good life?"
SAVE_DIR = "./local_qwen_model" # Directory where weights will be saved

# --- 1. Custom Linear Layer Definition ---
# This custom layer will replace the original lm_head.
# We inherit from nn.Linear so we can easily transfer the original weights.
class CustomLMHead(nn.Module):
    """
    A custom final linear layer for the language model head.
    The actual linear transformation is done by the standard torch.nn.Linear module.
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        # Define the actual linear layer that mirrors the original structure
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        print(f"CustomLMHead created: in={in_features}, out={out_features}, bias={bias}")

    def forward(self, x):
        # You can add custom logic here if needed (e.g., logging, custom activation)
        return self.linear(x)

# --- Check Device and Setup ---
if torch.cuda.is_available():
    # Use bfloat16 for reduced memory usage
    DTYPE = torch.float16 
    device = torch.device("cuda")
    print(f"✅ CUDA available. Using device: {torch.cuda.get_device_name(0)}")
    print(f"Model will be loaded in: {DTYPE}")
else:
    DTYPE = torch.float32
    device = torch.device("cpu")
    print("❌ CUDA not available. Falling back to CPU. Performance will be slower.")
    print("Model will be loaded in: torch.float32")


# --- 2. Download and Load Model & Tokenizer ---
print(f"\n🚀 Loading model '{MODEL_ID}'...")
start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",
    low_cpu_mem_usage=True
)
model.to(device)

load_time = time.time() - start_time
print(f"✅ Model loaded successfully in {load_time:.2f} seconds.")

print("\n--- Model Architecture BEFORE Surgery ---")
print(model)
print("-----------------------------------------")


# --- 3. Save Model to Current Directory ---
# The safetensors file is downloaded to the Hugging Face cache.
# We save it locally using save_pretrained().
print(f"\n💾 Saving model weights and config to local directory: {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR)
print(f"✅ Model saved. Check the '{SAVE_DIR}' directory for 'model.safetensors'.")

# --- 4. Replace the lm_head layer ---

# A. Extract parameters from the original layer
original_lm_head = model.lm_head
original_weights = original_lm_head.weight.data
original_bias = original_lm_head.bias.data if original_lm_head.bias is not None else None

# B. Get the dimensions for the new layer
in_features = original_lm_head.in_features
out_features = original_lm_head.out_features
has_bias = original_lm_head.bias is not None

# C. Create the custom layer instance
custom_layer = CustomLMHead(in_features, out_features, bias=has_bias)

# D. Transfer the original weights to the new layer
with torch.no_grad():
    custom_layer.linear.weight.data.copy_(original_weights)
    if has_bias and original_bias is not None:
        custom_layer.linear.bias.data.copy_(original_bias)

# FIX: Move the custom layer to the same device (GPU) AND the correct DTYPE (float16).
# Using .to(device=device, dtype=DTYPE) ensures both conditions are met, resolving the dtype mismatch error.
custom_layer.to(device=device, dtype=DTYPE)

# E. Replace the model's layer with the custom instance
# Since lm_head is a direct attribute of the overall model, we can use simple assignment:
model.lm_head = custom_layer 

print("\n--- Model Architecture AFTER Surgery ---")
# Now printing the whole model shows the change, but let's just confirm the lm_head:
print(model)
print("----------------------------------------")


# --- 5. Run Inference (Continue with the original script) ---
# Check that inference still works after the surgery.
# Note: For this specific model (Qwen2ForCausalLM), the embedding and lm_head 
# weights are tied by default in the architecture, but we have explicitly 
# replaced the layer here while preserving its weights.

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


# Run Inference (Generate Text)
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

# --- Clean up the local model save directory (Optional) ---
# If you want to clean up the saved files, uncomment the block below.
# import shutil
# if os.path.exists(SAVE_DIR):
#     print(f"\n🧹 Cleaning up saved directory: {SAVE_DIR}")
#     shutil.rmtree(SAVE_DIR)
#     print("Cleanup complete.")