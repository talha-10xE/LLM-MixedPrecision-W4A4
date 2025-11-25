# LLM-MixedPrecision-W4A4

This repository implements a high-performance, accuracy-preserving 4-bit (W4A4) quantization scheme for deep neural networks, focusing on Large Language Models (LLMs). We combine the speed of custom integer kernels with the accuracy of mixed-precision decomposition to achieve state-of-the-art efficiency.

## 🎯 The Core Innovation: Hybrid W4A4 Strategy

We address the two main challenges of 4-bit quantization—speed and accuracy—by creating two parallel compute pipelines with custom cuda kernels:

1. Speed Pipeline (W4A4): For the majority (>99%) of regular data, we use high-speed INT4 × INT4 matrix multiplication.
2. Accuracy Pipeline (W16A16): For high-magnitude outliers, we switch to full FP16 × FP16 (W16A16) computation to eliminate quantization error.

## 🚀 Project Roadmap and Implementation Plan

The project will proceed in three distinct phases, building complexity from simple dense layers to full LLM architectures.

### Phase 1: Proof of Concept & Custom Kernel Development (MNIST)

Goal: Validate the quantization logic, accuracy retention, and build the foundational custom MatMul kernel.

Target: A small Dense Network (e.g., 3-4 linear layers) trained on the MNIST 28x28 image classification dataset.

Weights Quantization: Static INT4 Quantization with 8x8 Tile-wise Granularity.

Activations Quantization: Dynamic Quantization with Mixed Precision Decomposition (Outlier separation).

Output: Measure INT4 accuracy against the FP16 baseline and verify the speedup in the W4A4 path along with memory footprint improvement.

### Phase 2: Scaling to LLMs (Linear Layers Only)

Goal: Integrate the custom kernel into a Hugging Face Transformer architecture and demonstrate feasibility on a large model.

Target: Apply the quantization scheme only to the Linear Layers (q_proj, k_proj, v_proj, o_proj, and MLP layers) of a small-scale LLM (e.g., Llama-3B).

Process: Load the pre-trained LLM, quantize weights, save the custom PyTorch dictionary, and modify the model forward pass to use our custom W4A4 kernels.

### Phase 3: Full Architecture Quantization

Goal: Apply the hybrid W4A4 scheme to all remaining floating-point operations.

Target: Quantize non-linear/non-MatMul layers, specifically Attention Mechanisms (e.g., Key/Query/Value scaling and masking operations), and other components like Layer Normalization.

Final Output: A fully optimized, W4A4-inference-ready LLM with minimal accuracy degradation.
