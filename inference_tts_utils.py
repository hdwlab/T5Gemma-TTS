"""Inference helpers for T5Gemma TTS."""

import logging
import re
import time
from typing import Any, Dict, Optional, Protocol, Tuple, Union, cast

import torch
import torchaudio

from .data.tokenizer import tokenize_audio
from .duration_estimator import detect_language

# ==== PyTorch 2.9+ audio backend compatibility helpers ====
# Starting with PyTorch 2.9, torchaudio internally depends on TorchCodec for
# audio loading/saving operations. In environments where TorchCodec is
# unavailable or incompatible (e.g., DGX Spark), these helpers provide a
# soundfile-based fallback.


def _torch_ge_29() -> bool:
    """Check whether the installed PyTorch version is >= 2.9.

    Returns:
        bool: True if PyTorch version is >= 2.9.

    """
    try:
        v = torch.__version__.split("+")[0]
        major, minor = map(int, v.split(".")[:2])
        return (major, minor) >= (2, 9)
    except Exception:
        return False


class _AudioInfoSoundFile(Protocol):
    """Audio info interface for soundfile backend."""

    samplerate: int


class _AudioInfoTorchAudio(Protocol):
    """Audio info interface for torchaudio backend."""

    sample_rate: int


def get_audio_info(audio_path: str) -> Union[_AudioInfoSoundFile, _AudioInfoTorchAudio]:
    """Get audio file info (sample rate).

    Returns an object with `samplerate` attribute (soundfile style).
    For PyTorch < 2.9, returns torchaudio.info result which has `sample_rate`.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        object: Audio info object from soundfile or torchaudio.

    """
    if _torch_ge_29():
        import soundfile as sf

        return sf.info(audio_path)
    else:
        return torchaudio.info(audio_path)


def get_sample_rate(info: Union[_AudioInfoSoundFile, _AudioInfoTorchAudio]) -> int:
    """Extract sample rate from audio info object (works for both backends).

    Args:
        info (object): Audio info object.

    Returns:
        int: Sample rate in Hz.

    """
    sr = cast(Optional[int], getattr(info, "samplerate", None))
    if sr is not None:
        return sr
    sr = cast(Optional[int], getattr(info, "sample_rate", None))
    if sr is not None:
        return sr
    raise ValueError("Audio info object has no samplerate/sample_rate attribute.")


# Text normalization (only applied when language is Japanese)
_REPLACE_MAP = {
    r"\t": "",
    r"\[n\]": "",
    r" ": "",
    r"　": "",
    r"[;▼♀♂《》≪≫①②③④⑤⑥]": "",
    r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]": "",
    r"[\uff5e\u301C]": "ー",
    r"？": "?",
    r"！": "!",
    r"[●◯〇]": "○",
    r"♥": "♡",
}
_FULLWIDTH_ALPHA_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(
            list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)),
            list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)),
            strict=True,
        )
    }
)
_HALFWIDTH_KATAKANA_CHARS = "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ"
_FULLWIDTH_KATAKANA_CHARS = (
    "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチ"
    "ツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン"
)
_HALFWIDTH_KATAKANA_TO_FULLWIDTH = str.maketrans(
    _HALFWIDTH_KATAKANA_CHARS, _FULLWIDTH_KATAKANA_CHARS
)
_FULLWIDTH_DIGITS_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(range(0xFF10, 0xFF1A), range(0x30, 0x3A), strict=True)
    }
)


def _normalize_japanese_text(text: str) -> str:
    """Normalize Japanese text to match dataset conventions.

    Args:
        text (str): Input text.

    Returns:
        str: Normalized text.

    """
    for pattern, replacement in _REPLACE_MAP.items():
        text = re.sub(pattern, replacement, text)

    text = text.translate(_FULLWIDTH_ALPHA_TO_HALFWIDTH)
    text = text.translate(_FULLWIDTH_DIGITS_TO_HALFWIDTH)
    text = text.translate(_HALFWIDTH_KATAKANA_TO_FULLWIDTH)

    # Collapse long ellipses
    text = re.sub(r"…{3,}", "……", text)
    return text


def normalize_text_with_lang(text: str, lang: Optional[str]) -> Tuple[str, Optional[str]]:
    """Normalize text when language is Japanese.

    Returns normalized text and the resolved language code to avoid re-detecting downstream.

    Args:
        text (str): Input text.
        lang (Optional[str]): Language code (optional).

    Returns:
        Tuple[str, Optional[str]]: Normalized text and resolved language code.

    """
    resolved_lang = lang.lower() if isinstance(lang, str) else None
    if not text:
        return text, resolved_lang
    if resolved_lang is None:
        resolved_lang = detect_language(text)
    if resolved_lang and resolved_lang.startswith("ja"):
        return _normalize_japanese_text(text), resolved_lang
    return text, resolved_lang


@torch.no_grad()
def inference_one_sample(
    model: Any,
    model_args: Any,
    text_tokenizer: Any,
    audio_tokenizer: Any,
    audio_fn: Optional[str],
    target_text: str,
    lang: Optional[str],
    device: str,
    decode_config: Dict[str, Any],
    prompt_end_frame: int,
    target_generation_length: float,
    prefix_transcript: Optional[str] = None,
    quiet: bool = False,
    repeat_prompt: Union[int, str] = 0,
    multi_trial: Optional[list] = None,
    return_frames: bool = False,
    num_samples: int = 1,
) -> tuple:
    """Run single-sample inference.

    Args:
        model (Any): T5Gemma TTS model instance.
        model_args (Any): Model config/args.
        text_tokenizer (Any): Text tokenizer.
        audio_tokenizer (Any): Audio tokenizer.
        audio_fn (Optional[str]): Reference audio path.
        target_text (str): Target text to synthesize.
        lang (Optional[str]): Language code (optional).
        device (str): Device string (e.g., "cpu", "cuda:0").
        decode_config (Dict[str, Any]): Decoding configuration.
        prompt_end_frame (int): Prompt frame cutoff.
        target_generation_length (float): Target generation length in seconds.
        prefix_transcript (Optional[str]): Prefix transcript (optional).
        quiet (bool): Disable logging if True.
        repeat_prompt (Union[int, str]): Prompt repetition strategy.
        multi_trial (Optional[list]): Multi-trial config (unused here).
        return_frames (bool): Return codec frames if True.
        num_samples (int): Number of samples to generate.

    Returns:
        tuple: Generated audio (and optional frames), following the original API.

    """
    multi_trial = multi_trial or []

    # XCodec2 always uses a single codebook.
    if int(getattr(model_args, "n_codebooks", 1)) != 1:
        raise ValueError("XCodec2 backend supports only n_codebooks=1.")
    n_codebooks = 1
    # empty_token = int(getattr(model_args, "empty_token", 0))
    codec_sr = int(decode_config["codec_sr"])
    y_sep_token = getattr(model_args, "y_sep_token", None)
    x_sep_token = getattr(model_args, "x_sep_token", None)
    eos_token = getattr(model_args, "eos", getattr(model_args, "eog", None))
    add_eos_token = getattr(model_args, "add_eos_to_text", 0)
    add_bos_token = getattr(model_args, "add_bos_to_text", 0)

    silence_tokens = decode_config["silence_tokens"]
    if isinstance(silence_tokens, str):
        silence_tokens = eval(silence_tokens)

    has_reference_audio = audio_fn is not None and str(audio_fn).lower() not in {"", "none", "null"}

    if has_reference_audio:
        assert audio_fn is not None
        encoded_frames = tokenize_audio(
            audio_tokenizer,
            audio_fn,
            offset=0,
            num_frames=prompt_end_frame if prompt_end_frame > 0 else -1,
        )
    else:
        # No reference: start with zero-length prompt; BOS (empty_token) will be added in the model.
        encoded_frames = torch.empty(
            (1, n_codebooks, 0),
            dtype=torch.long,
            device=audio_tokenizer.device,
        )

    if encoded_frames.ndim == 2:
        # Allow [T, 1] or [1, T] inputs for convenience.
        encoded_frames = encoded_frames.unsqueeze(0)
    if encoded_frames.ndim != 3 or encoded_frames.shape[0] != 1:
        raise ValueError(f"Unexpected prompt shape {encoded_frames.shape}")
    # Normalize to [B, 1, frames]
    if encoded_frames.shape[2] == 1:
        encoded_frames = encoded_frames.transpose(1, 2).contiguous()
    if encoded_frames.shape[1] != 1:
        raise ValueError(f"Expected a single codebook axis, got shape {encoded_frames.shape}")

    single_encoded_frames = encoded_frames.clone()

    # VoiceStar support removed; keep variable for backward compatibility.
    effective_delay_inc = 0
    if isinstance(repeat_prompt, int) and repeat_prompt > 0:
        cur_repeat_prompt = repeat_prompt
        while cur_repeat_prompt > 0:
            encoded_frames = torch.cat([encoded_frames, single_encoded_frames], dim=2)
            cur_repeat_prompt -= 1
    elif isinstance(repeat_prompt, str) and repeat_prompt.lower() == "max":
        repeat_prompt = 0
        while (
            encoded_frames.shape[2]
            + codec_sr * target_generation_length
            + effective_delay_inc
            + single_encoded_frames.shape[2]
            < model_args.audio_max_length * codec_sr
        ):
            encoded_frames = torch.cat([encoded_frames, single_encoded_frames], dim=2)
            repeat_prompt += 1

    # Only insert y_sep when we actually have a reference prompt.
    if y_sep_token is not None and has_reference_audio and encoded_frames.shape[2] > 0:
        encoded_frames = torch.cat(
            [
                encoded_frames,
                torch.full(
                    (1, n_codebooks, 1),
                    y_sep_token,
                    dtype=torch.long,
                    device=encoded_frames.device,
                ),
            ],
            dim=2,
        )

    original_audio = encoded_frames.transpose(2, 1).contiguous()  # [B, T, K]
    prompt_frames = original_audio.shape[1]

    # ---- Text Processing (text-only) ----
    # Normalize Japanese text when requested or detected; reuse lang to avoid double detection
    target_text, lang = normalize_text_with_lang(target_text, lang)
    if prefix_transcript:
        prefix_transcript, _ = normalize_text_with_lang(prefix_transcript, lang)

    def encode_text_hf(text):
        if isinstance(text, list):
            text = " ".join(text)
        return text_tokenizer.encode(text.strip(), add_special_tokens=False)

    text_tokens = encode_text_hf(target_text)
    if prefix_transcript:
        prefix_tokens = encode_text_hf(prefix_transcript)
        if x_sep_token is not None:
            text_tokens = prefix_tokens + [x_sep_token] + text_tokens
        else:
            text_tokens = prefix_tokens + text_tokens

    if add_eos_token:
        text_tokens.append(add_eos_token)
    if add_bos_token:
        text_tokens = [add_bos_token] + text_tokens

    text_tokens = torch.LongTensor(text_tokens).unsqueeze(0)
    text_tokens_lens = torch.LongTensor([text_tokens.shape[-1]])

    if not quiet:
        logging.info(
            "original audio length: %d codec frames, which is %.2f sec.",
            original_audio.shape[1],
            original_audio.shape[1] / codec_sr,
        )

    if getattr(model_args, "parallel_pattern", 0) != 0:
        tgt_y_lens = torch.LongTensor(
            [int(original_audio.shape[1] + codec_sr * target_generation_length + 2)]
        )
    else:
        tgt_y_lens = torch.LongTensor(
            [
                int(
                    original_audio.shape[1]
                    + codec_sr * target_generation_length
                    + effective_delay_inc
                )
            ]
        )

    assert decode_config["sample_batch_size"] <= 1
    stime = time.time()
    assert multi_trial == []
    if not quiet:
        logging.info("running inference with num_samples=%d", num_samples)

    concat_frames, gen_frames = model.inference_tts(
        text_tokens.to(device),
        text_tokens_lens.to(device),
        original_audio.to(device),
        tgt_y_lens=tgt_y_lens.to(device),
        top_k=decode_config["top_k"],
        top_p=decode_config["top_p"],
        min_p=decode_config["min_p"],
        temperature=decode_config["temperature"],
        stop_repetition=decode_config["stop_repetition"],
        silence_tokens=silence_tokens,
        prompt_frames=prompt_frames,
        num_samples=num_samples,
    )

    inference_time = time.time() - stime
    num_generated_tokens = gen_frames.shape[-1]
    tokens_per_sec = (
        num_generated_tokens * num_samples / inference_time if inference_time > 0 else 0.0
    )
    audio_duration = num_generated_tokens / codec_sr
    real_time_factor = audio_duration * num_samples / inference_time if inference_time > 0 else 0.0

    if not quiet:
        logging.info("inference on %d sample(s) took: %.4f sec.", num_samples, inference_time)
        logging.info(
            "generated encoded_frames.shape: %s, which is %.2f sec per sample.",
            gen_frames.shape,
            audio_duration,
        )
    print(
        f"[Speed] {tokens_per_sec:.2f} tokens/s | RTF: {real_time_factor:.2f}x | "
        f"Generated {num_generated_tokens} tokens x {num_samples} samples in {inference_time:.2f}s"
    )

    def _strip_sep_and_eos_per_sample(
        frames: torch.Tensor, sep_token: Optional[int], eos_token: Optional[int]
    ):
        """Strip sep/eos tokens per sample, returning list of variable-length tensors."""
        results = []
        for b in range(frames.shape[0]):
            sample_frames = frames[b : b + 1]  # [1, K, T]
            mask = torch.ones_like(sample_frames, dtype=torch.bool)
            if sep_token is not None:
                mask &= sample_frames.ne(sep_token)
            if eos_token is not None:
                mask &= sample_frames.ne(eos_token)
            if mask.all():
                results.append(sample_frames)
            else:
                # Find valid length for this sample
                valid_len = int(mask[0, 0].sum().item())
                cleaned = sample_frames[0, 0][mask[0, 0]][:valid_len]
                results.append(cleaned.view(1, 1, -1))
        return results

    # Process frames per sample to handle variable lengths
    concat_frames_list = _strip_sep_and_eos_per_sample(concat_frames, y_sep_token, eos_token)
    gen_frames_list = _strip_sep_and_eos_per_sample(gen_frames, y_sep_token, eos_token)

    # Decode each sample
    concat_samples = []
    gen_samples = []

    for i in range(num_samples):
        # Decode generated audio
        gen_sample = audio_tokenizer.decode(gen_frames_list[i])
        gen_samples.append(gen_sample)

        # Decode concatenated audio if reference exists
        if has_reference_audio:
            try:
                concat_sample = audio_tokenizer.decode(concat_frames_list[i])
                concat_samples.append(concat_sample)
            except Exception as exc:
                logging.warning(
                    "Failed to decode concatenated prompt audio for sample %d: %s", i, exc
                )
                concat_samples.append(gen_sample)
        else:
            concat_samples.append(gen_sample)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # For backward compatibility: if num_samples=1, return single tensors
    if num_samples == 1:
        if return_frames:
            return (
                concat_samples[0],
                gen_samples[0],
                concat_frames_list[0].detach().cpu(),
                gen_frames_list[0].detach().cpu(),
            )
        return concat_samples[0], gen_samples[0]

    # For multiple samples, return lists
    if return_frames:
        return (
            concat_samples,
            gen_samples,
            [f.detach().cpu() for f in concat_frames_list],
            [f.detach().cpu() for f in gen_frames_list],
        )
    return concat_samples, gen_samples
