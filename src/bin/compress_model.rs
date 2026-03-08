//! Compress model weights to Q4 binary format
//! Reduces 3.8GB safetensors → ~1GB Q4 binary

use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: compress_model <input_dir> <output.qora-tts>");
        eprintln!("Example: compress_model model model_q4.qora-tts");
        std::process::exit(1);
    }

    let input_dir = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);

    eprintln!("QORA-TTS Model Compression Tool");
    eprintln!("Input: {}", input_dir.display());
    eprintln!("Output: {}", output_path.display());
    eprintln!();

    let t0 = Instant::now();

    // Load config
    let config = qora_tts::config::QoraTTSConfig::from_file(input_dir.join("config.json"))
        .expect("Failed to load config.json");

    // Load all models
    eprintln!("Loading models from safetensors...");

    let t1 = Instant::now();
    let talker = qora_tts::loader::load_talker(&input_dir, &config, true)
        .expect("Failed to load talker");
    eprintln!("  Talker loaded in {:.1?}", t1.elapsed());

    let t1 = Instant::now();
    let predictor = qora_tts::loader::load_code_predictor(&input_dir, &config, true)
        .expect("Failed to load code predictor");
    eprintln!("  Code predictor loaded in {:.1?}", t1.elapsed());

    let t1 = Instant::now();
    let decoder = qora_tts::loader::load_speech_decoder(&input_dir)
        .expect("Failed to load speech decoder");
    eprintln!("  Speech decoder loaded in {:.1?}", t1.elapsed());

    let t1 = Instant::now();
    let speaker_encoder = qora_tts::loader::load_speaker_encoder(&input_dir)
        .expect("Failed to load speaker encoder");
    eprintln!("  Speaker encoder loaded in {:.1?}", t1.elapsed());

    eprintln!();
    eprintln!("All models loaded in {:.1?}", t0.elapsed());

    // Calculate sizes
    let talker_mb = talker.memory_bytes() / (1024 * 1024);
    let predictor_mb = predictor.memory_bytes() / (1024 * 1024);
    let decoder_mb = decoder.memory_bytes() / (1024 * 1024);
    let speaker_mb = speaker_encoder.memory_bytes() / (1024 * 1024);
    let total_mb = talker_mb + predictor_mb + decoder_mb + speaker_mb;

    eprintln!("Memory usage:");
    eprintln!("  Talker: {} MB", talker_mb);
    eprintln!("  Code predictor: {} MB", predictor_mb);
    eprintln!("  Speech decoder: {} MB", decoder_mb);
    eprintln!("  Speaker encoder: {} MB", speaker_mb);
    eprintln!("  TOTAL: {} MB", total_mb);
    eprintln!();

    // Save to binary
    eprintln!("Saving Q4 binary to {}...", output_path.display());
    let t1 = Instant::now();

    qora_tts::save::save_model_with_speaker(
        &talker,
        &predictor,
        &decoder,
        &speaker_encoder,
        &output_path
    ).expect("Failed to save model");

    let file_size = std::fs::metadata(&output_path)
        .map(|m| m.len() / (1024 * 1024))
        .unwrap_or(0);

    eprintln!("Saved in {:.1?}", t1.elapsed());
    eprintln!();
    eprintln!("============================================================");
    eprintln!("Compression complete!");
    eprintln!("  Input: 3.8 GB (safetensors)");
    eprintln!("  Output: {} MB (Q4 binary)", file_size);
    eprintln!("  Reduction: {:.1}×", 3800.0 / file_size as f32);
    eprintln!("  Total time: {:.1?}", t0.elapsed());
    eprintln!("============================================================");
}
