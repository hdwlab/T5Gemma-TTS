"""
Gradio demo for HF-format T5GemmaVoice checkpoints.

Usage:
    python inference_gradio.py --model_dir ./t5gemma_voice_hf --port 7860
"""

import argparse
import os
import random
from functools import lru_cache
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch

from data.tokenizer import AudioTokenizer
from duration_estimator import estimate_duration
from inference_tts_utils import (
    get_audio_info,
    get_sample_rate,
    inference_one_sample,
    normalize_text_with_lang,
    transcribe_audio,
)

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: Optional[int]) -> int:
    """
    Seed all RNGs. If seed is None, draw a fresh random seed and return it.
    """
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
        print(f"[Info] No seed provided; using random seed {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


def _device_str_from_map(device_map, fallback: str) -> str:
    if not device_map:
        return fallback
    for dev in device_map.values():
        if dev in ("cpu", "disk"):
            continue
        if isinstance(dev, int):
            return f"cuda:{dev}"
        if isinstance(dev, torch.device):
            return str(dev)
        if isinstance(dev, str):
            return dev
    return fallback


# ---------------------------------------------------------------------------
# Model / tokenizer loaders (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_resources(
    model_dir: str,
    xcodec2_model_name: Optional[str],
    xcodec2_sample_rate: Optional[int],
    use_torch_compile: bool,
    cpu_codec: bool,
    whisper_device: str,
):
    """Load model and tokenizers from HF-format directory or repo."""
    if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
        raise ImportError("Please install transformers before running the demo.")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    # Check if model is quantized (bitsandbytes) - if not, move to device manually
    is_quantized = getattr(model, "is_loaded_in_8bit", False) or getattr(
        model, "is_loaded_in_4bit", False
    )
    has_device_map = bool(getattr(model, "hf_device_map", None))
    if not is_quantized and device != "cpu" and not has_device_map:
        model = model.to(device)
    if has_device_map:
        device = _device_str_from_map(model.hf_device_map, device)
    model.eval()
    cfg = model.config

    tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    xcodec2_model_name = xcodec2_model_name or getattr(cfg, "xcodec2_model_name", None)

    audio_tokenizer = AudioTokenizer(
        backend="xcodec2",
        device=torch.device("cpu") if cpu_codec else torch.device(device),
        model_name=xcodec2_model_name,
        sample_rate=xcodec2_sample_rate,
    )

    # Apply torch.compile for faster inference (2nd run onwards)
    can_compile = (
        use_torch_compile
        and torch.cuda.is_available()
        and not cpu_codec
    )
    if can_compile:
        print("[Info] Applying torch.compile to model and codec (this may take a minute on first inference)...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            audio_tokenizer.codec = torch.compile(audio_tokenizer.codec, mode="reduce-overhead")
            print("[Info] torch.compile applied successfully.")
        except Exception as e:
            print(f"[Warning] torch.compile failed, falling back to eager mode: {e}")

    codec_audio_sr = getattr(cfg, "codec_audio_sr", audio_tokenizer.sample_rate)
    if xcodec2_sample_rate is not None:
        codec_audio_sr = xcodec2_sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)

    return {
        "model": model,
        "cfg": cfg,
        "text_tokenizer": text_tokenizer,
        "audio_tokenizer": audio_tokenizer,
        "device": device,
        "codec_audio_sr": codec_audio_sr,
        "codec_sr": codec_sr,
        "whisper_device": whisper_device,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    reference_speech: Optional[str],
    reference_text: Optional[str],
    target_text: str,
    target_duration: Optional[float],
    top_k: int,
    top_p: float,
    min_p: float,
    temperature: float,
    seed: Optional[int],
    resources: dict,
    cut_off_sec: int = 100,
    lang: Optional[str] = None,
    num_samples: int = 1,
) -> list:
    """
    Run TTS and return a list of (sample_rate, waveform) tuples for Gradio playback.
    When num_samples=1, returns a single tuple for backward compatibility.
    """
    used_seed = seed_everything(None if seed is None else int(seed))

    model = resources["model"]
    cfg = resources["cfg"]
    text_tokenizer = resources["text_tokenizer"]
    audio_tokenizer = resources["audio_tokenizer"]
    device = resources["device"]
    codec_audio_sr = resources["codec_audio_sr"]
    codec_sr = resources["codec_sr"]

    silence_tokens = []
    repeat_prompt = 0  # fixed; UI removed
    stop_repetition = 3  # keep sensible default from CLI HF script
    sample_batch_size = 1

    no_reference_audio = reference_speech is None or str(reference_speech).strip().lower() in {"", "none", "null"}
    has_reference_text = reference_text not in (None, "", "none", "null")

    if no_reference_audio and has_reference_text:
        raise ValueError("reference_text was provided but reference_speech is missing.")

    if no_reference_audio:
        prefix_transcript = ""
    elif not has_reference_text:
        print("[Info] No reference text; transcribing reference speech with Whisper.")
        prefix_transcript = transcribe_audio(reference_speech, resources["whisper_device"])
        print(f"[Info] Whisper transcription: {prefix_transcript}")
    else:
        prefix_transcript = reference_text

    # Language + normalization (Japanese only)
    lang = None if lang in {None, "", "none", "null"} else str(lang)
    target_text, lang_code = normalize_text_with_lang(target_text, lang)
    if prefix_transcript:
        prefix_transcript, _ = normalize_text_with_lang(prefix_transcript, lang_code)

    if target_duration is None:
        target_generation_length = estimate_duration(
            target_text=target_text,
            reference_speech=None if no_reference_audio else reference_speech,
            reference_transcript=None if no_reference_audio else prefix_transcript,
            target_lang=lang_code,
            reference_lang=lang_code,
        )
        print(f"[Info] target_duration not provided, estimated as {target_generation_length:.2f} seconds.")
    else:
        target_generation_length = float(target_duration)

    if not no_reference_audio:
        info = get_audio_info(reference_speech)
        prompt_end_frame = int(cut_off_sec * get_sample_rate(info))
    else:
        prompt_end_frame = 0

    decode_config = {
        "top_k": int(top_k),
        "top_p": float(top_p),
        "min_p": float(min_p),
        "temperature": float(temperature),
        "stop_repetition": stop_repetition,
        "codec_audio_sr": codec_audio_sr,
        "codec_sr": codec_sr,
        "silence_tokens": silence_tokens,
        "sample_batch_size": sample_batch_size,
    }

    concat_audio, gen_audio = inference_one_sample(
        model=model,
        model_args=cfg,
        text_tokenizer=text_tokenizer,
        audio_tokenizer=audio_tokenizer,
        audio_fn=None if no_reference_audio else reference_speech,
        target_text=target_text,
        lang=lang_code,
        device=device,
        decode_config=decode_config,
        prompt_end_frame=prompt_end_frame,
        target_generation_length=target_generation_length,
        prefix_transcript=prefix_transcript,
        multi_trial=[],
        repeat_prompt=repeat_prompt,
        return_frames=False,
        num_samples=num_samples,
    )

    print(f"[Info] Seed used for this run: {used_seed}")

    # Handle single vs multiple samples
    if num_samples == 1:
        # Single sample: gen_audio is a tensor
        audio = gen_audio[0].detach().cpu()
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        waveform = audio.numpy()
        max_abs = float(np.max(np.abs(waveform)))
        rms = float(np.sqrt(np.mean(waveform**2)))
        print(f"[Info] Generated audio stats -> max_abs: {max_abs:.6f}, rms: {rms:.6f}")
        return [(codec_audio_sr, waveform)]
    else:
        # Multiple samples: gen_audio is a list of tensors
        results = []
        for i, audio in enumerate(gen_audio):
            wav = audio[0].detach().cpu()
            if wav.ndim == 2 and wav.shape[0] == 1:
                wav = wav.squeeze(0)
            waveform = wav.numpy()
            max_abs = float(np.max(np.abs(waveform)))
            rms = float(np.sqrt(np.mean(waveform**2)))
            print(f"[Info] Sample {i+1} audio stats -> max_abs: {max_abs:.6f}, rms: {rms:.6f}")
            results.append((codec_audio_sr, waveform))
        return results


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo(resources, server_port: int, share: bool, max_num_samples: int = 256, cols_per_row: int = 8):
    description = (
        "Reference speech is optional. If provided without reference text, Whisper (large-v3-turbo) "
        "will auto-transcribe and use it as the prompt text.\n\n"
        "**Batch Generation**: Set 'Number of Samples' > 1 to generate multiple variations in parallel "
        "(same input, different random samples). This is efficient as the encoder runs only once."
    )

    with gr.Blocks() as demo:
        gr.Markdown("## T5Gemma-TTS (HF)")
        gr.Markdown(description)

        with gr.Row():
            reference_speech_input = gr.Audio(
                label="Reference Speech (optional)",
                type="filepath",
            )
            reference_text_box = gr.Textbox(
                label="Reference Text (optional, leave blank to auto-transcribe)",
                lines=2,
            )

        target_text_box = gr.Textbox(
            label="Target Text",
            value="こんにちは、私はAIです。これは音声合成のテストです。",
            lines=3,
        )

        with gr.Row():
            target_duration_box = gr.Textbox(
                label="Target Duration (seconds, optional)",
                value="",
                placeholder="Leave blank for auto estimate",
            )
            seed_box = gr.Textbox(
                label="Random Seed (optional)",
                value="",
                placeholder="Leave blank to use a random seed each run",
            )

        with gr.Row():
            top_k_box = gr.Slider(label="top_k", minimum=0, maximum=100, step=1, value=30)
            top_p_box = gr.Slider(label="top_p", minimum=0.1, maximum=1.0, step=0.05, value=0.9)
            min_p_box = gr.Slider(label="min_p (0 = disabled)", minimum=0.0, maximum=1.0, step=0.05, value=0.0)
            temperature_box = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, step=0.05, value=0.8)

        with gr.Row():
            num_samples_slider = gr.Slider(
                label="Number of Samples (batch generation)",
                minimum=1,
                maximum=max_num_samples,
                step=1,
                value=1,
                info="Generate multiple variations in parallel with different random samples",
            )

        generate_button = gr.Button("Generate")

        # Create multiple audio outputs in a grid layout (cols_per_row columns)
        output_audios = []
        num_rows = (max_num_samples + cols_per_row - 1) // cols_per_row
        with gr.Column():
            for row_idx in range(num_rows):
                with gr.Row():
                    for col_idx in range(cols_per_row):
                        i = row_idx * cols_per_row + col_idx
                        if i >= max_num_samples:
                            break
                        audio = gr.Audio(
                            label=f"#{i+1}",
                            type="numpy",
                            interactive=False,
                            visible=(i == 0),
                            show_label=True,
                            scale=1,
                            min_width=80,
                        )
                        output_audios.append(audio)

        def gradio_inference(
            reference_speech,
            reference_text,
            target_text,
            target_duration,
            top_k,
            top_p,
            min_p,
            temperature,
            seed,
            num_samples,
        ):
            dur = float(target_duration) if str(target_duration).strip() not in {"", "None", "none"} else None
            seed_val = None
            if str(seed).strip() not in {"", "None", "none"}:
                seed_val = int(float(seed))

            num_samples = int(num_samples)
            results = run_inference(
                reference_speech=reference_speech,
                reference_text=reference_text,
                target_text=target_text,
                target_duration=dur,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                temperature=temperature,
                seed=seed_val,
                resources=resources,
                num_samples=num_samples,
            )

            # Prepare outputs for all audio components
            outputs = []
            for i in range(max_num_samples):
                if i < len(results):
                    # Return audio data and make visible
                    outputs.append(gr.update(value=results[i], visible=True))
                else:
                    # Hide unused components
                    outputs.append(gr.update(value=None, visible=False))
            return outputs

        generate_button.click(
            fn=gradio_inference,
            inputs=[
                reference_speech_input,
                reference_text_box,
                target_text_box,
                target_duration_box,
                top_k_box,
                top_p_box,
                min_p_box,
                temperature_box,
                seed_box,
                num_samples_slider,
            ],
            outputs=output_audios,
        )

    demo.launch(server_name="0.0.0.0", server_port=server_port, share=share, debug=True)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gradio demo for HF T5GemmaVoice")
    parser.add_argument("--model_dir", type=str, default="./t5gemma_voice_hf", help="HF model directory or repo id")
    parser.add_argument("--xcodec2_model_name", type=str, default=None, help="Override xcodec2 model name from config")
    parser.add_argument("--xcodec2_sample_rate", type=int, default=None, help="Override xcodec2 sample rate from config")
    parser.add_argument("--cpu_whisper", action="store_true", help="Run Whisper transcription on CPU to save VRAM")
    parser.add_argument("--cpu_codec", action="store_true", help="Run XCodec2 tokenizer on CPU to save VRAM")
    parser.add_argument("--low_vram", action="store_true", help="Preset: cpu_whisper + cpu_codec")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile (faster startup, slower inference)")
    args = parser.parse_args()

    if args.low_vram:
        args.cpu_whisper = True
        args.cpu_codec = True
        args.no_compile = True

    if args.cpu_whisper or args.low_vram:
        whisper_device = "cpu"
    elif torch.cuda.is_available():
        whisper_device = "cuda"
    elif torch.backends.mps.is_available():
        whisper_device = "mps"
    else:
        whisper_device = "cpu"

    resources = _load_resources(
        args.model_dir,
        args.xcodec2_model_name,
        args.xcodec2_sample_rate,
        use_torch_compile=not args.no_compile,
        cpu_codec=args.cpu_codec,
        whisper_device=whisper_device,
    )
    build_demo(resources=resources, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
