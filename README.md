---
language:
  - en
  - zh
  - de
  - it
  - pt
  - es
  - ja
  - ko
  - fr
  - ru
license: apache-2.0
tags:
  - text-to-speech
  - tts
  - rust
  - pure-rust
  - no-python
  - qwen3-tts
  - cpu-inference
  - quantized
  - q4
base_model: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
pipeline_tag: text-to-speech
library_name: qora
---

# QORA-TTS 0.6B - Pure Rust Text-to-Speech

Pure Rust TTS engine with 9 built-in speakers. No Python, no CUDA, no external ML frameworks. Single executable + model weights = portable text-to-speech that runs on any machine.

**Smart system awareness** — automatically detects your hardware (RAM, CPU threads) and adjusts generation limits so TTS runs well even on constrained systems. **9 built-in voices** — works out of the box with no reference audio needed. **10 languages** supported.

Based on [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) (Apache 2.0).

## License

This project is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). The base model [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) is released by the Qwen team under Apache 2.0.

## What It Does

QORA-TTS 0.6B converts text to natural-sounding speech. It can:

- **9 built-in voices** — male and female speakers embedded in the model, no reference audio needed
- **10 languages** — English, Chinese, German, Italian, Portuguese, Spanish, Japanese, Korean, French, Russian
- **24 kHz output** — high-quality mono WAV audio
- **Fast generation** — lighter model for quicker speech synthesis
- **Controllable generation** — adjust length, temperature, and sampling parameters

## Platform Support

| Platform | Binary | Status |
|----------|--------|--------|
| **Windows x86_64** | `qora-tts.exe` | Tested |
| **Linux x86_64** | `qora-tts` | Supported |
| **macOS aarch64** | `qora-tts` | Supported |

CPU-only — no GPU needed. Pre-built binaries on the [Releases](https://github.com/qora-protocol/QORA-TTS-12Hz-0.6B/releases) page.

## Quick Start

1. Download from the [Releases](https://github.com/qora-protocol/QORA-TTS-12Hz-0.6B/releases) page
2. Run:

```bash
# Use built-in speaker
qora-tts.exe --speaker ryan --language english --text "Hello, how are you?"

# Different speaker
qora-tts.exe --speaker serena --language chinese --text "你好世界"

# Japanese speaker
qora-tts.exe --speaker ono_anna --language japanese --text "こんにちは"

# Control length and output
qora-tts.exe --speaker aiden --language english --text "Good morning!" --max-codes 200 --output greeting.wav

# Reproducible output with seed
qora-tts.exe --speaker ryan --text "Same every time" --seed 42
```

## Files

```
qora-tts.exe          4.1 MB   Inference engine
model.qora-tts       971 MB    Q4 weights (talker + predictor + decoder)
config.json           4.8 KB   Model configuration
tokenizer.json         11 MB   Tokenizer (151,936 vocab)
vocab.json            2.7 MB   Vocabulary
merges.txt            1.6 MB   BPE merges
tokenizer_config.json 7.2 KB   Tokenizer config
```

**No safetensors needed.** Everything loads from `model.qora-tts`. The exe auto-finds all files in its own directory.

## Architecture

| Component | Details |
|-----------|---------|
| **Parameters** | 0.6B total |
| **Talker** | 28 layers, hidden=1024, 16/8 GQA heads, SwiGLU FFN 3072 |
| **Code Predictor** | 5 layers, hidden=1024, 16 code groups |
| **Speech Decoder** | 8-layer transformer + Vocos vocoder, 16 VQ codebooks |
| **Quantization** | Q4 (4-bit symmetric, group_size=32) with LUT-optimized dequantization |
| **Sample Rate** | 24 kHz mono WAV |
| **Code Rate** | 12.5 Hz (1 code = 80ms of audio) |

### How It Works

1. **Text encoding** — tokenize input text with 151K BPE vocabulary
2. **Speaker selection** — load built-in speaker embedding by name
3. **Code generation** — 28-layer transformer (Talker) generates speech codes autoregressively
4. **Code expansion** — 5-layer Code Predictor expands code0 into 16 codebooks (codes 0-15)
5. **Audio synthesis** — VQ decoder + Vocos vocoder converts codes to 24kHz waveform

## Smart System Awareness

QORA-TTS detects your system at startup and automatically adjusts generation limits:

```
QORA-TTS — Pure Rust Text-to-Speech Engine
System: 16384 MB RAM (9856 MB free), 12 threads
```

| Available RAM | Max Codes | Default | Audio Length |
|---------------|-----------|---------|-------------|
| < 4 GB | 200 | 100 | ~8s |
| 4-8 GB | 500 | 300 | ~20s |
| 8-12 GB | 1000 | 500 | ~40s |
| >= 12 GB | 2000 | 500 | ~80s |

**Hard caps apply even to explicit user values** — if you pass `--max-codes 2000` on a system with 6 GB free RAM, it gets clamped to 500 automatically. This prevents the model from running for too long on weak systems.

## CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--text <text>` | "Hello, how are you today?" | Text to synthesize |
| `--speaker <name>` | ryan | Built-in speaker name |
| `--language <name>` | english | Target language |
| `--output <path>` | output.wav | Output WAV path |
| `--max-codes <n>` | 500 | Max code timesteps (~n/12.5 seconds) |
| `--temperature <f>` | 0.8 | Sampling temperature |
| `--top-k <n>` | 50 | Top-K sampling |
| `--seed <n>` | random | Random seed for reproducibility |

## Built-in Speakers

| Speaker | Language | Description |
|---------|----------|-------------|
| **ryan** | English | Dynamic male voice |
| **aiden** | English | Sunny American male |
| **serena** | Chinese | Warm, gentle female |
| **vivian** | Chinese | Bright young female |
| **uncle_fu** | Chinese | Seasoned male |
| **dylan** | Beijing dialect | Youthful male |
| **eric** | Sichuan dialect | Lively male |
| **ono_anna** | Japanese | Playful female |
| **sohee** | Korean | Warm female |

## Supported Languages

| Language | Flag Value |
|----------|-----------|
| English | `english` |
| Chinese | `chinese` |
| German | `german` |
| Italian | `italian` |
| Portuguese | `portuguese` |
| Spanish | `spanish` |
| Japanese | `japanese` |
| Korean | `korean` |
| French | `french` |
| Russian | `russian` |

## Performance

Tested on i5-11500 (6C/12T), 16GB RAM, CPU-only:

| Phase | Time | Notes |
|-------|------|-------|
| Model Load | ~0.6s | From binary, 971 MB |
| Prefill | ~2-5s | Text + speaker embedding processing |
| Code Generation | ~1.5s/code | Autoregressive, 12.5 codes/sec of audio |
| Code Expansion | ~0.1s | 5-layer predictor, 16 codebooks |
| Audio Decode | ~0.5s/frame | VQ + Vocos vocoder |
| RAM Usage | ~970 MB | Q4 model in memory |

**Example:** "Hello, how are you?" (~3 seconds of audio) takes ~10-15 seconds total.

## Comparison with 1.7B

| | QORA-TTS 0.6B | QORA-TTS 1.7B |
|---|---------------|---------------|
| Parameters | 0.6B | 1.7B |
| Model size | 971 MB | 1559 MB |
| Voice cloning | No | Yes (ECAPA-TDNN) |
| Built-in speakers | 9 (embedded) | 25 (via voice files) |
| Code generation | ~1.5s/code | ~2.5s/code |
| Quality | Good | Higher |
| Best for | Speed + simplicity | Quality + cloning |

## Building from Source

```bash
cargo build --release
```

### Dependencies

- **Language**: Pure Rust (2021 edition)
- `half` — F16 support
- `tokenizers` — HuggingFace tokenizer
- `safetensors` — Weight loading
- `serde_json` — Config parsing
- **No ML framework** for inference — all matrix ops are hand-written Rust

### Cross-Platform Releases

Pre-built binaries via GitHub Actions for Windows x86_64, Linux x86_64, macOS aarch64.

## Model Binary Format (.qora-tts)

Custom binary format for fast loading:

```
Header:  "QTTS" magic + version + format byte
Talker:  28 transformer layers (Q4 quantized)
Predictor: 5 transformer layers + code embeddings
Decoder: VQ codebooks + 8 transformer layers + Vocos vocoder
```
