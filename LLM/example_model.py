import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# --- Configuration ---
# A very small, instruction-tuned model suitable for 8GB VRAM
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" 
PROMPT = "India vs USA vs CHina? which country is better?"

# --- Check Device and Setup ---
if torch.cuda.is_available():
    # Use bfloat16 for reduced memory usage and faster performance on most modern NVIDIA GPUs
    # RTX 2070 supports float16 (half-precision), but often bfloat16 is a better compromise
    # For older or less-compatible GPUs, you might switch to torch.float16 or even 4-bit quantization (see section below)
    DTYPE = torch.float16 
    print(f"✅ CUDA available. Using device: {torch.cuda.get_device_name(0)}")
    print(f"Model will be loaded in: {DTYPE}")
else:
    DTYPE = torch.float32
    print("❌ CUDA not available. Falling back to CPU. Performance will be slower.")
    print("Model will be loaded in: torch.float32")

# --- 1. Download and Load Model & Tokenizer ---
print(f"\n🚀 Loading model '{MODEL_ID}'...")
start_time = time.time()

# AutoModelForCausalLM loads the model architecture and weights
# AutoTokenizer loads the vocabulary and tokenization rules
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto", # Automatically map model layers to available devices (GPU/CPU)
    low_cpu_mem_usage=True # Helps with memory management during loading
)

print(model)

load_time = time.time() - start_time
print(f"✅ Model loaded successfully in {load_time:.2f} seconds.")

# --- 2. Prepare Input for Inference (Tokenize) ---
# Follow the model's chat template for instruction-tuned models
messages = [
    {"role": "system", "content": "You are a helpful and concise assistant."},
    {"role": "user", "content": PROMPT}
]

# Convert the chat history into the model's expected prompt format
input_text = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

# Encode the prompt text into input IDs and move them to the GPU
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)


# --- 3. Run Inference (Generate Text) ---
print(f"\n🧠 Generating response to prompt: '{PROMPT}'")
start_time = time.time()

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=256, # Limit the length of the generated response
        do_sample=True, # Enable sampling for more creative/less repetitive output
        temperature=0.7, # Control randomness
        top_p=0.95, # Nucleus sampling
        pad_token_id=tokenizer.eos_token_id # Important for generation stability
    )

# --- 4. Decode and Print Output ---
generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
generation_time = time.time() - start_time

print("--- Generated Response ---")
print(generated_text)
print("--------------------------")
print(f"⚡ Generation complete in {generation_time:.2f} seconds.")
print(f"Tokens/Second: {len(output_ids[0]) / generation_time:.2f}")