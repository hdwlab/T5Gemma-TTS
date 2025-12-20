from __future__ import annotations

import argparse
import hashlib
import logging
import random
import re
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

# Ensure project root is on sys.path (so data.tokenizer can be imported)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.tokenizer import AudioTokenizer


# ---------------------------------------------------------------------------
# Text normalization (from Aratako/llasa-server)
# ---------------------------------------------------------------------------
REPLACE_MAP: Dict[str, str] = {
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

FULLWIDTH_ALPHA_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(
            list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)),
            list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)),
        )
    }
)
HALFWIDTH_KATAKANA_TO_FULLWIDTH = str.maketrans(
    "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ",
    "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン",
)
FULLWIDTH_DIGITS_TO_HALFWIDTH = str.maketrans(
    {chr(full): chr(half) for full, half in zip(range(0xFF10, 0xFF1A), range(0x30, 0x3A))}
)
INVALID_PATTERN = re.compile(
    r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    r"\u0041-\u005A\u0061-\u007A"
    r"\u0030-\u0039"
    r"。、!?…♪♡○「」『』（）・／ー：:_\s]"
)


def normalize_text(text: str) -> str:
    for pattern, replacement in REPLACE_MAP.items():
        text = re.sub(pattern, replacement, text)
    text = text.translate(FULLWIDTH_ALPHA_TO_HALFWIDTH)
    text = text.translate(FULLWIDTH_DIGITS_TO_HALFWIDTH)
    text = text.translate(HALFWIDTH_KATAKANA_TO_FULLWIDTH)
    text = re.sub(r"…{3,}", "……", text)
    if re.match(r"^（.+）$", text):
        text = text[1:-1]
    text = re.sub(r"（.+?）", "", text)
    text = re.sub(r"（.*$", "", text)
    if "sasayaki_" not in text:
        text = re.sub(r"_.*$", "", text)
    return text.strip()


def is_allowed_text(text: str) -> bool:
    return bool(text) and not INVALID_PATTERN.search(text)


_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9\-]+")


def sanitize_id(raw: str) -> str:
    sanitized = _ID_SAFE_RE.sub("_", raw.strip())
    sanitized = sanitized.strip("._")
    return sanitized or "utt"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SampleIn:
    base_id: str
    text: str
    audio_path: Path
    corpus: str
    style: str  # style label; "normal" for non-styled
    group_key: str  # neighbor grouping (corpus + optional style)


@dataclass
class SampleOut:
    rel_id: str  # shard/utt
    utt_base: str
    text: str
    corpus: str
    style: Optional[str]
    group_key: str
    split: str
    duration_sec: float
    token_len: int


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def load_mono_audio(path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    audio_np, sr = sf.read(path, always_2d=False)
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)
    waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (1, T)
    waveform = waveform.to(dtype=torch.float32)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    waveform = waveform * 0.99
    return waveform, sr


def compute_shard(utt_base: str) -> str:
    return hashlib.md5(utt_base.encode("utf-8")).hexdigest()[:2]


def ensure_dirs(output_dir: Path, save_audio: bool) -> Dict[str, Path]:
    text_dir = output_dir / "text"
    codes_dir = output_dir / "xcodec2_1cb"
    manifest_dir = output_dir / "manifest_final"
    neighbor_dir = output_dir / "neighbors"
    dirs = {
        "text": text_dir,
        "codes": codes_dir,
        "manifest": manifest_dir,
        "neighbors": neighbor_dir,
    }
    if save_audio:
        dirs["audio"] = output_dir / "audio"
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def write_outputs(
    rel_id: str,
    text: str,
    tokens: torch.Tensor,
    dirs: Dict[str, Path],
    split: str,
    save_audio: bool,
    waveform: Optional[torch.Tensor],
    sample_rate: Optional[int],
    overwrite: bool,
) -> int:
    shard, utt_base = rel_id.split("/", 1)

    text_parent = dirs["text"] / shard
    codes_parent = dirs["codes"] / shard
    text_parent.mkdir(exist_ok=True)
    codes_parent.mkdir(exist_ok=True)

    text_path = text_parent / f"{utt_base}.txt"
    codes_path = codes_parent / f"{utt_base}.txt"
    if not overwrite and (text_path.exists() or codes_path.exists()):
        raise FileExistsError(f"Destination exists: {text_path} / {codes_path}")

    text_path.write_text(text + "\n", encoding="utf-8")

    tokens_np = tokens.detach().cpu().numpy()
    if tokens_np.ndim == 3:
        tokens_np = tokens_np.squeeze(0)
    if tokens_np.ndim == 1:
        tokens_np = tokens_np[None, :]
    elif tokens_np.ndim == 2 and tokens_np.shape[0] > tokens_np.shape[1]:
        tokens_np = tokens_np.T
    token_len = int(tokens_np.shape[-1])
    code_lines = [" ".join(str(int(tok)) for tok in row.tolist()) for row in tokens_np]
    codes_path.write_text("\n".join(code_lines) + "\n", encoding="utf-8")

    manifest_path = dirs["manifest"] / f"{split}.txt"
    with manifest_path.open("a", encoding="utf-8") as mf:
        mf.write(f"{rel_id}\t{token_len}\n")

    if save_audio and waveform is not None and sample_rate is not None:
        audio_parent = dirs["audio"] / shard
        audio_parent.mkdir(parents=True, exist_ok=True)
        audio_path = audio_parent / f"{utt_base}.wav"
        sf.write(audio_path, waveform.squeeze(0).cpu().numpy(), sample_rate)

    return token_len


# ---------------------------------------------------------------------------
# Corpus loaders
# ---------------------------------------------------------------------------
def build_wav_map(root: Path, prefer_48k: bool = True) -> Dict[str, Path]:
    wavs = list(root.rglob("*.wav"))
    if prefer_48k:
        wavs_48 = [p for p in wavs if "48k" in p.parts]
        if wavs_48:
            wavs = wavs_48
    mapping: Dict[str, Path] = {}
    for p in wavs:
        key = p.stem
        if key in mapping:
            logging.warning("Duplicate basename %s (keeping first: %s)", key, mapping[key])
            continue
        mapping[key] = p
    return mapping


def _resolve_path(p: Path) -> Path:
    """Resolve a path; if not exists, try relative to script directory."""
    if p.exists():
        return p
    alt = SCRIPT_DIR / p.name if p.is_absolute() else SCRIPT_DIR / p
    return alt if alt.exists() else p


def load_ita_transcripts(emotion_path: Path, recitation_path: Path) -> Dict[str, str]:
    def _load_one(path: Path) -> Dict[str, str]:
        records: Dict[str, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                utt_id, rest = line.split(":", 1)
                text = rest.split(",", 1)[0].strip()
                records[utt_id.strip()] = text
        return records

    emo = _resolve_path(emotion_path)
    rec = _resolve_path(recitation_path)
    if not emo.exists():
        raise FileNotFoundError(f"ITA emotion transcript not found: {emotion_path}")
    if not rec.exists():
        raise FileNotFoundError(f"ITA recitation transcript not found: {recitation_path}")
    merged = _load_one(emo)
    merged.update(_load_one(rec))
    return merged


def load_mana_transcripts(path: Path) -> Dict[str, str]:
    path = _resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"MANA transcript not found: {path}")
    records: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            utt_id, text = line.split(":", 1)
            records[utt_id.strip()] = text.strip()
    return records


def load_voice_pairs(list_path: Path, voice_root: Path) -> List[Tuple[str, Path]]:
    voice_root = _resolve_path(voice_root)
    wav_map = {p.name: p for p in voice_root.rglob("*.wav")}
    pairs: List[Tuple[str, Path]] = []
    with list_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().replace("\ufeff", "")
            if not line or "\t" not in line or ".wav" not in line:
                continue
            text_part, file_part = line.split("\t", 1)
            text = text_part.strip(" \u3000")
            fname = file_part.strip()
            path = wav_map.get(fname)
            if path is None:
                logging.warning("[voice] audio not found for %s", fname)
                continue
            pairs.append((text, path))
    return pairs


def load_amitaro_corpus(readme_path: Path, audio_root: Path) -> List[Tuple[str, Path]]:
    readme_path = _resolve_path(readme_path)
    audio_root = _resolve_path(audio_root)
    records: List[Tuple[str, Path]] = []
    with readme_path.open("r", encoding="shift_jis") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("AMITARO_"):
                continue
            if ":" not in line:
                continue
            utt_id, text = line.split(":", 1)
            wav_path = audio_root / f"{utt_id}.wav"
            if not wav_path.exists():
                logging.warning("[amitaro corpus] missing wav %s", wav_path)
                continue
            records.append((utt_id, text.strip()))
    return records


# ---------------------------------------------------------------------------
# Neighbor builder
# ---------------------------------------------------------------------------
def build_neighbors(samples: List[SampleOut], neighbor_dir: Path, max_neighbors: int) -> None:
    groups: Dict[str, List[SampleOut]] = {}
    for s in samples:
        groups.setdefault(s.group_key, []).append(s)

    for s in tqdm(samples, desc="neighbors", dynamic_ncols=True):
        peers = [p for p in groups.get(s.group_key, []) if p.rel_id != s.rel_id]
        if not peers:
            continue
        peers.sort(key=lambda p: abs(p.duration_sec - s.duration_sec))
        if max_neighbors > 0:
            peers = peers[:max_neighbors]

        out_path = neighbor_dir / f"{s.rel_id}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for p in peers:
                dist = abs(p.duration_sec - s.duration_sec)
                f.write(f"{p.rel_id}.txt\t{dist:.2f}\t{p.duration_sec:.2f}\n")


# ---------------------------------------------------------------------------
# Sample collectors per corpus
# ---------------------------------------------------------------------------
def collect_voice_samples(voice_root: Path) -> List[SampleIn]:
    voice_root = _resolve_path(voice_root)
    list_path = _resolve_path(voice_root / "voice_list_20251119.txt")
    if not list_path.exists():
        raise FileNotFoundError(f"voice list not found: {list_path}")
    pairs = load_voice_pairs(list_path, voice_root)
    result: List[SampleIn] = []
    for text, path in pairs:
        base_id = sanitize_id(f"voice_{path.stem}")
        result.append(
            SampleIn(
                base_id=base_id,
                text=f"normal: {text}",
                audio_path=path,
                corpus="voice",
                style="normal",
                group_key="voice:normal",
            )
        )
    return result


def collect_amitaro_samples(corpus_root: Path) -> List[SampleIn]:
    corpus_root = _resolve_path(corpus_root)
    readme = corpus_root / "Readme.txt"
    audio_root = corpus_root / "48k"
    pairs = load_amitaro_corpus(readme, audio_root)
    result: List[SampleIn] = []
    for utt_id, text in pairs:
        base_id = sanitize_id(f"ami_{utt_id.lower()}")
        wav_path = audio_root / f"{utt_id}.wav"
        result.append(
            SampleIn(
                base_id=base_id,
                text=f"normal: {text}",
                audio_path=wav_path,
                corpus="amitaro_corpus",
                style="normal",
                group_key="amitaro_corpus:normal",
            )
        )
    return result


def collect_ita_samples(style_root: Path, style_tag: str, transcripts: Dict[str, str]) -> List[SampleIn]:
    style_root = _resolve_path(style_root)
    wav_map = build_wav_map(style_root)
    # Some style packages (punsuka / sasayaki) use sentence-as-filename.
    text_map: Dict[str, Path] = {}
    for p in wav_map.values():
        norm = normalize_text(p.stem)
        text_map.setdefault(norm, p)
        norm_no_num = re.sub(r"[0-9０-９]+$", "", norm)
        if norm_no_num and norm_no_num not in text_map:
            text_map[norm_no_num] = p
    has_recitation = any("recitation" in p.parts for p in wav_map.values())

    def resolve_wav(utt: str) -> Optional[Path]:
        # Normal (2.1) uses emoNormal### / recitation### filenames
        if style_tag == "normal":
            if utt.startswith("EMOTION"):
                try:
                    num = int(utt.split("_")[-1])
                except ValueError:
                    return None
                return wav_map.get(f"emoNormal{num:03d}")
            if utt.startswith("RECITATION"):
                try:
                    num = int(utt.split("_")[-1])
                except ValueError:
                    return None
                return wav_map.get(f"recitation{num:03d}")
        # Other styles use IDs as filenames
        path = wav_map.get(utt)
        if path is not None:
            return path
        return text_map.get(normalize_text(transcripts.get(utt, "")))

    result: List[SampleIn] = []
    missing = 0
    skipped_rec = 0
    for utt_id, text in transcripts.items():
        if utt_id.startswith("RECITATION") and not has_recitation:
            skipped_rec += 1
            continue
        path = resolve_wav(utt_id)
        if path is None:
            if missing < 5:
                logging.warning("[ITA %s] missing wav for %s", style_tag, utt_id)
            missing += 1
            continue
        base_id = sanitize_id(f"{style_tag}_{utt_id.lower()}")
        styled_text = f"{style_tag}: {text}"
        result.append(
            SampleIn(
                base_id=base_id,
                text=styled_text,
                audio_path=path,
                corpus="ita",
                style=style_tag,
                group_key=f"ita:{style_tag}",
            )
        )
    if missing:
        logging.warning("[ITA %s] missing %d files (showing first 5 above)", style_tag, missing)
    if skipped_rec:
        logging.info("[ITA %s] recitation entries skipped because no recitation audio found: %d", style_tag, skipped_rec)
    return result


def collect_mana_samples(style_root: Path, style_tag: str, transcripts: Dict[str, str]) -> List[SampleIn]:
    style_root = _resolve_path(style_root)
    wav_map = build_wav_map(style_root)
    text_map: Dict[str, Path] = {}
    for p in wav_map.values():
        norm = normalize_text(p.stem)
        text_map.setdefault(norm, p)
        norm_no_num = re.sub(r"[0-9０-９]+$", "", norm)
        if norm_no_num and norm_no_num not in text_map:
            text_map[norm_no_num] = p
    result: List[SampleIn] = []
    missing = 0
    for utt_id, text in transcripts.items():
        path = wav_map.get(utt_id)
        if path is None:
            path = text_map.get(normalize_text(text))
        if path is None:
            if missing < 5:
                logging.warning("[MANA %s] missing wav for %s", style_tag, utt_id)
            missing += 1
            continue
        base_id = sanitize_id(f"{style_tag}_{utt_id.lower()}")
        styled_text = f"{style_tag}: {text}"
        result.append(
            SampleIn(
                base_id=base_id,
                text=styled_text,
                audio_path=path,
                corpus="mana",
                style=style_tag,
                group_key=f"mana:{style_tag}",
            )
        )
    if missing:
        logging.warning("[MANA %s] missing %d files (showing first 5 above)", style_tag, missing)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare amitaro datasets for T5Gemma LoRA")
    parser.add_argument("--output-dir", type=Path, default=Path("processed_amitaro"))
    parser.add_argument("--codec-model", default="NandemoGHS/Anime-XCodec2-44.1kHz-v2")
    parser.add_argument("--codec-sample-rate", type=int, default=16000, help="Resample target for tokenizer input")
    parser.add_argument("--valid-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for debugging")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-audio", action="store_true", help="Also save resampled wavs to output/audio")
    parser.add_argument("--min-duration", type=float, default=0.1)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--max-neighbors", type=int, default=50)
    parser.add_argument("--ita-emotion", type=Path, default=Path("emotion_transcript_utf8.txt"))
    parser.add_argument("--ita-recitation", type=Path, default=Path("recitation_transcript_utf8.txt"))
    parser.add_argument("--mana-transcript", type=Path, default=Path("mana-corpus.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    rng = random.Random(args.seed)

    # Instantiate tokenizer and override encode_sample_rate when requested
    tokenizer = AudioTokenizer(model_name=args.codec_model)
    default_sr = getattr(tokenizer, "encode_sample_rate", tokenizer.sample_rate)
    if args.codec_sample_rate and args.codec_sample_rate != default_sr:
        logging.info("Overriding codec input rate: %s -> %s", default_sr, args.codec_sample_rate)
        tokenizer.encode_sample_rate = args.codec_sample_rate
        tokenizer.sample_rate = args.codec_sample_rate
    encode_sr = int(getattr(tokenizer, "encode_sample_rate", tokenizer.sample_rate))
    device = tokenizer.device if hasattr(tokenizer, "device") else torch.device("cpu")
    logging.info("Tokenizer device: %s, encode_sr=%d", device, encode_sr)

    # Collect raw samples
    ita_styles = {
        "normal": Path("ITAcorpus_amitaro_2.1"),
        "runrun": Path("ITAcorpus_amitaro_runrun"),
        "yofukashi": Path("ITAcorpus_amitaro_yofukashi_1.1"),
        "punsuka": Path("ITAcorpus_amitaro_punsuka_1.0"),
        "sasayaki_a": Path("ITAcorpus_amitaro_sasayaki_A1.0"),
        "sasayaki_b": Path("ITAcorpus_amitaro_sasayaki_B1.0"),
    }
    mana_styles = {
        "normal": Path("MANAcorpus_amitaro_1.1"),
        "runrun": Path("MANAcorpus_amitaro_runrun"),
        "yofukashi": Path("MANAcorpus_amitaro_yofukashi"),
        "sasayaki_a": Path("MANAcorpus_amitaro_sasayaki_A"),
        "sasayaki_b": Path("MANAcorpus_amitaro_sasayaki_B"),
    }

    ita_transcripts = load_ita_transcripts(args.ita_emotion, args.ita_recitation)
    mana_transcripts = load_mana_transcripts(args.mana_transcript)

    all_inputs: List[SampleIn] = []
    all_inputs.extend(collect_voice_samples(Path("amitarovoice_20251119_01")))
    all_inputs.extend(collect_amitaro_samples(Path("amitarocorpus_amitaro_1.0")))

    for tag, root in ita_styles.items():
        resolved = _resolve_path(root)
        if not resolved.exists():
            logging.warning("Skipping missing ITA style %s (%s)", tag, resolved)
            continue
        all_inputs.extend(collect_ita_samples(resolved, tag, ita_transcripts))

    for tag, root in mana_styles.items():
        resolved = _resolve_path(root)
        if not resolved.exists():
            logging.warning("Skipping missing MANA style %s (%s)", tag, resolved)
            continue
        all_inputs.extend(collect_mana_samples(resolved, tag, mana_transcripts))

    if args.max_samples:
        all_inputs = all_inputs[: args.max_samples]

    logging.info("Total candidate samples: %d", len(all_inputs))

    dirs = ensure_dirs(args.output_dir, save_audio=args.save_audio)
    seen_ids: set[str] = set()
    outputs: List[SampleOut] = []
    split_counts = {"train": 0, "valid": 0}
    skipped = 0

    for idx, sample in enumerate(tqdm(all_inputs, desc="encode", dynamic_ncols=True)):
        split = "valid" if args.valid_ratio > 0 and rng.random() < args.valid_ratio else "train"

        text_norm = normalize_text(sample.text)
        if not is_allowed_text(text_norm):
            skipped += 1
            logging.debug("Skipping due to invalid chars: %s", sample.base_id)
            logging.debug("  original text: %s", sample.text)
            continue

        base_id = sample.base_id
        if base_id in seen_ids:
            suffix = 1
            while f"{base_id}_{suffix}" in seen_ids:
                suffix += 1
            base_id = f"{base_id}_{suffix}"
        seen_ids.add(base_id)

        shard = compute_shard(base_id)
        rel_id = f"{shard}/{base_id}"

        try:
            waveform, sr = load_mono_audio(sample.audio_path, encode_sr)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            logging.warning("Failed to load %s: %s", sample.audio_path, exc)
            continue

        duration_sec = waveform.shape[-1] / sr
        if duration_sec < args.min_duration or duration_sec > args.max_duration:
            skipped += 1
            logging.warning("Skipping due to duration %.2f sec out of range: %s", duration_sec, rel_id)
            continue

        try:
            codes = tokenizer.encode(waveform.to(device))
            codes = codes.squeeze(0).cpu()
            if codes.numel() == 0:
                skipped += 1
                logging.debug("Skipping due to zero-length tokens: %s", sample.audio_path)
                continue
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            logging.warning("Tokenizer failed on %s: %s", sample.audio_path, exc)
            continue

        try:
            token_len = write_outputs(
                rel_id=rel_id,
                text=text_norm,
                tokens=codes,
                dirs=dirs,
                split=split,
                save_audio=args.save_audio,
                waveform=waveform if args.save_audio else None,
                sample_rate=sr if args.save_audio else None,
                overwrite=args.overwrite,
            )
        except FileExistsError as exc:
            skipped += 1
            logging.warning("Skip existing %s: %s", rel_id, exc)
            continue

        outputs.append(
            SampleOut(
                rel_id=rel_id,
                utt_base=base_id,
                text=text_norm,
                corpus=sample.corpus,
                style=sample.style,
                group_key=sample.group_key,
                split=split,
                duration_sec=duration_sec,
                token_len=token_len,
            )
        )
        split_counts[split] += 1
        if (idx + 1) % args.log_every == 0:
            logging.info(
                "Processed %d/%d (skipped %d) latest=%s len=%d",
                idx + 1,
                len(all_inputs),
                skipped,
                rel_id,
                token_len,
            )

    logging.info("Done encoding: train=%d, valid=%d, skipped=%d", split_counts["train"], split_counts["valid"], skipped)

    if outputs:
        build_neighbors(outputs, dirs["neighbors"], max_neighbors=args.max_neighbors)
        logging.info("Neighbor files written to %s", dirs["neighbors"])


if __name__ == "__main__":
    main()
