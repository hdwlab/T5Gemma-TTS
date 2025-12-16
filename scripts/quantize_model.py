"""
Quantize T5GemmaVoice model to 8-bit or 4-bit using bitsandbytes.

Usage:
    python scripts/quantize_model.py \
        --model_name Aratako/T5Gemma-TTS-2b-2b \
        --output_dir ./quantized_model_8bit \
        --bits 8

    python scripts/quantize_model.py \
        --model_name Aratako/T5Gemma-TTS-2b-2b \
        --output_dir ./quantized_model_4bit \
        --bits 4

Requirements:
    - CUDA GPU (bitsandbytes requires CUDA)
    - pip install bitsandbytes accelerate
"""

import argparse
import logging
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Layers to skip quantization (embeddings, output heads, normalization)
# These are critical for model quality
MODULES_TO_NOT_QUANTIZE = [
    # Embeddings
    "backbone.model.encoder.embed_tokens",
    "backbone.model.decoder.embed_tokens",
    "audio_embedding",
    # Output prediction layers
    "predict_layer",
    # lm_head (if exists, though it's pruned in this model)
    "lm_head",
    # Decoder is very sensitive, thus skip all decoder layers
    "backbone.model.decoder",
    "decoder_module",
]


def copy_custom_code_files(model_name: str, output_dir: str) -> None:
    """Copy custom code files from HF repo to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Files needed for trust_remote_code
    custom_code_patterns = ["modeling_", "configuration_"]

    try:
        # List files in the repo
        repo_files = list_repo_files(model_name)

        # Find and download custom code files
        for filename in repo_files:
            is_custom = any(p in filename for p in custom_code_patterns)
            if is_custom and filename.endswith(".py"):
                logger.info(f"Copying custom code file: {filename}")
                downloaded_path = hf_hub_download(model_name, filename)
                dest_path = output_path / filename
                shutil.copy2(downloaded_path, dest_path)

    except Exception as e:
        logger.warning(f"Could not copy custom code files: {e}")
        logger.warning(
            "You may need to manually copy modeling_*.py and configuration_*.py"
        )


def get_quantization_config(bits: int, double_quant: bool = True) -> BitsAndBytesConfig:
    """Create BitsAndBytesConfig for specified bit width."""
    if bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=MODULES_TO_NOT_QUANTIZE,
        )
    elif bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=MODULES_TO_NOT_QUANTIZE,
        )
    else:
        raise ValueError(f"Unsupported bit width: {bits}. Use 4 or 8.")


def main():
    parser = argparse.ArgumentParser(description="Quantize T5GemmaVoice model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Aratako/T5Gemma-TTS-2b-2b",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=8,
        help="Quantization bit width (4 or 8)",
    )
    parser.add_argument(
        "--no_double_quant",
        action="store_true",
        help="Disable double quantization for 4-bit (slightly larger but faster)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bitsandbytes quantization")

    logger.info(f"Loading model: {args.model_name}")
    logger.info(f"Quantization: {args.bits}-bit")
    logger.info(f"Modules to skip: {MODULES_TO_NOT_QUANTIZE}")

    # Create quantization config
    quant_config = get_quantization_config(
        bits=args.bits,
        double_quant=not args.no_double_quant,
    )

    # Load model with quantization
    logger.info("Loading and quantizing model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Check which layers were quantized
    quantized_modules = []
    non_quantized_modules = []
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            has_quant = hasattr(module.weight, "quant_state")
            is_int = str(module.weight.dtype) in ["torch.int8", "torch.uint8"]
            if has_quant or is_int:
                quantized_modules.append(name)
            else:
                non_quantized_modules.append(name)

    logger.info(f"Quantized modules: {len(quantized_modules)}")
    logger.info(f"Non-quantized modules: {len(non_quantized_modules)}")

    # Save the quantized model
    logger.info(f"Saving quantized model to: {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)

    # Copy custom code files (modeling_*.py, configuration_*.py) for trust_remote_code
    logger.info("Copying custom code files for trust_remote_code...")
    copy_custom_code_files(args.model_name, args.output_dir)

    logger.info("Quantization complete!")


if __name__ == "__main__":
    main()
