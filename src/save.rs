//! Save/load all QORA-TTS weights to a single binary file (.qora-tts).
//!
//! Format: magic "QTTS" + version + format_id + talker + code_predictor + speech_decoder.
//! Loading from binary: ~2-3 seconds vs ~2.5 min from safetensors.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::gemv::*;
use crate::talker::{TalkerWeights, TalkerLayerWeights};
use crate::code_predictor::{CodePredictorWeights, PredictorLayerWeights};
use crate::decoder::*;
use crate::conv::*;

const MAGIC: &[u8; 4] = b"QTTS";
const VERSION: u32 = 1;

// ============================================================
// Save
// ============================================================

pub fn save_model(
    talker: &TalkerWeights,
    predictor: &CodePredictorWeights,
    decoder: &SpeechDecoderWeights,
    path: &Path,
) -> io::Result<()> {
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, File::create(path)?);

    // Header
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    let format_id: u8 = if matches!(talker.layers[0].q_proj, Weight::Q4(_)) { 1 } else { 0 };
    w.write_all(&[format_id])?;

    eprintln!("Saving talker...");
    save_talker(&mut w, talker)?;
    eprintln!("Saving code predictor...");
    save_predictor(&mut w, predictor)?;
    eprintln!("Saving speech decoder...");
    save_decoder(&mut w, decoder)?;

    w.flush()?;
    Ok(())
}

fn save_talker(w: &mut impl Write, t: &TalkerWeights) -> io::Result<()> {
    w.write_all(&(t.layers.len() as u32).to_le_bytes())?;
    w.write_all(&(t.hidden_size as u32).to_le_bytes())?;
    w.write_all(&(t.text_hidden_size as u32).to_le_bytes())?;
    w.write_all(&(t.num_heads as u32).to_le_bytes())?;
    w.write_all(&(t.num_kv_heads as u32).to_le_bytes())?;
    w.write_all(&(t.head_dim as u32).to_le_bytes())?;
    w.write_all(&(t.num_kv_groups as u32).to_le_bytes())?;
    w.write_all(&(t.mrope_section.len() as u32).to_le_bytes())?;
    for &s in &t.mrope_section {
        w.write_all(&(s as u32).to_le_bytes())?;
    }

    // Per-layer
    for layer in &t.layers {
        write_weight_io(w, &layer.q_proj)?;
        write_weight_io(w, &layer.k_proj)?;
        write_weight_io(w, &layer.v_proj)?;
        write_weight_io(w, &layer.o_proj)?;
        write_f16_vec_io(w, &layer.q_norm)?;
        write_f16_vec_io(w, &layer.k_norm)?;
        write_weight_io(w, &layer.gate_proj)?;
        write_weight_io(w, &layer.up_proj)?;
        write_weight_io(w, &layer.down_proj)?;
        write_f16_vec_io(w, &layer.input_norm_gamma)?;
        write_f16_vec_io(w, &layer.post_attn_norm_gamma)?;
    }

    // Global
    write_weight_io(w, &t.codec_embedding)?;
    write_weight_io(w, &t.text_embedding)?;
    write_weight_io(w, &t.text_proj_fc1)?;
    write_f32_vec_io(w, &t.text_proj_fc1_bias)?;
    write_weight_io(w, &t.text_proj_fc2)?;
    write_f32_vec_io(w, &t.text_proj_fc2_bias)?;
    write_weight_io(w, &t.codec_head)?;
    write_f16_vec_io(w, &t.final_norm_gamma)?;
    write_f32_vec_io(w, &t.rope_cos)?;
    write_f32_vec_io(w, &t.rope_sin)?;

    Ok(())
}

fn save_predictor(w: &mut impl Write, p: &CodePredictorWeights) -> io::Result<()> {
    w.write_all(&(p.layers.len() as u32).to_le_bytes())?;
    w.write_all(&(p.hidden_size as u32).to_le_bytes())?;
    w.write_all(&(p.num_heads as u32).to_le_bytes())?;
    w.write_all(&(p.num_kv_heads as u32).to_le_bytes())?;
    w.write_all(&(p.head_dim as u32).to_le_bytes())?;
    w.write_all(&(p.num_kv_groups as u32).to_le_bytes())?;

    // Input projection (optional — flag byte + data)
    if let Some(ref proj) = p.input_proj {
        w.write_all(&[1u8])?;  // has_proj = true
        write_weight_io(w, proj)?;
        write_f32_vec_io(w, p.input_proj_bias.as_ref().unwrap())?;
    } else {
        w.write_all(&[0u8])?;  // has_proj = false
    }

    w.write_all(&(p.codec_embeddings.len() as u32).to_le_bytes())?;
    for emb in &p.codec_embeddings {
        write_weight_io(w, emb)?;
    }

    for layer in &p.layers {
        write_weight_io(w, &layer.q_proj)?;
        write_weight_io(w, &layer.k_proj)?;
        write_weight_io(w, &layer.v_proj)?;
        write_weight_io(w, &layer.o_proj)?;
        write_f16_vec_io(w, &layer.q_norm)?;
        write_f16_vec_io(w, &layer.k_norm)?;
        write_weight_io(w, &layer.gate_proj)?;
        write_weight_io(w, &layer.up_proj)?;
        write_weight_io(w, &layer.down_proj)?;
        write_f16_vec_io(w, &layer.input_norm_gamma)?;
        write_f16_vec_io(w, &layer.post_attn_norm_gamma)?;
    }

    w.write_all(&(p.lm_heads.len() as u32).to_le_bytes())?;
    for h in &p.lm_heads {
        write_weight_io(w, h)?;
    }

    write_f16_vec_io(w, &p.final_norm_gamma)?;
    write_f32_vec_io(w, &p.rope_cos)?;
    write_f32_vec_io(w, &p.rope_sin)?;

    Ok(())
}

fn save_conv1d(w: &mut impl Write, c: &Conv1dWeight) -> io::Result<()> {
    w.write_all(&(c.in_channels as u32).to_le_bytes())?;
    w.write_all(&(c.out_channels as u32).to_le_bytes())?;
    w.write_all(&(c.kernel_size as u32).to_le_bytes())?;
    w.write_all(&(c.stride as u32).to_le_bytes())?;
    w.write_all(&(c.padding as u32).to_le_bytes())?;
    w.write_all(&(c.dilation as u32).to_le_bytes())?;
    write_f32_vec_io(w, &c.weight)?;
    write_f32_vec_io(w, &c.bias)?;
    Ok(())
}

fn save_conv_t1d(w: &mut impl Write, c: &ConvTranspose1dWeight) -> io::Result<()> {
    w.write_all(&(c.in_channels as u32).to_le_bytes())?;
    w.write_all(&(c.out_channels as u32).to_le_bytes())?;
    w.write_all(&(c.kernel_size as u32).to_le_bytes())?;
    w.write_all(&(c.stride as u32).to_le_bytes())?;
    w.write_all(&(c.padding as u32).to_le_bytes())?;
    write_f32_vec_io(w, &c.weight)?;
    write_f32_vec_io(w, &c.bias)?;
    Ok(())
}

fn save_dw_conv(w: &mut impl Write, c: &DepthwiseConv1dWeight) -> io::Result<()> {
    w.write_all(&(c.channels as u32).to_le_bytes())?;
    w.write_all(&(c.kernel_size as u32).to_le_bytes())?;
    w.write_all(&(c.padding as u32).to_le_bytes())?;
    write_f32_vec_io(w, &c.weight)?;
    write_f32_vec_io(w, &c.bias)?;
    Ok(())
}

fn save_decoder(w: &mut impl Write, d: &SpeechDecoderWeights) -> io::Result<()> {
    // Config
    w.write_all(&(d.hidden_size as u32).to_le_bytes())?;
    w.write_all(&(d.num_heads as u32).to_le_bytes())?;
    w.write_all(&(d.head_dim as u32).to_le_bytes())?;
    w.write_all(&(d.intermediate_size as u32).to_le_bytes())?;
    w.write_all(&(d.sliding_window as u32).to_le_bytes())?;
    w.write_all(&(d.codebook_dim as u32).to_le_bytes())?;

    // Codebooks
    write_f32_vec_io(w, &d.first_codebook)?;
    write_f32_vec_io(w, &d.first_output_proj)?;
    w.write_all(&(d.rest_codebooks.len() as u32).to_le_bytes())?;
    for cb in &d.rest_codebooks {
        write_f32_vec_io(w, cb)?;
    }
    write_f32_vec_io(w, &d.rest_output_proj)?;

    // Pre-conv
    save_conv1d(w, &d.pre_conv)?;

    // Transformer
    write_f32_vec_io(w, &d.tf_input_w)?;
    write_f32_vec_io(w, &d.tf_input_b)?;
    w.write_all(&(d.tf_layers.len() as u32).to_le_bytes())?;
    for layer in &d.tf_layers {
        write_f32_vec_io(w, &layer.q_proj)?;
        write_f32_vec_io(w, &layer.k_proj)?;
        write_f32_vec_io(w, &layer.v_proj)?;
        write_f32_vec_io(w, &layer.o_proj)?;
        write_f32_vec_io(w, &layer.gate_proj)?;
        write_f32_vec_io(w, &layer.up_proj)?;
        write_f32_vec_io(w, &layer.down_proj)?;
        write_f32_vec_io(w, &layer.input_norm)?;
        write_f32_vec_io(w, &layer.post_attn_norm)?;
        write_f32_vec_io(w, &layer.attn_scale)?;
        write_f32_vec_io(w, &layer.mlp_scale)?;
    }
    write_f32_vec_io(w, &d.tf_norm)?;
    write_f32_vec_io(w, &d.tf_output_w)?;
    write_f32_vec_io(w, &d.tf_output_b)?;

    // Upsample
    w.write_all(&(d.upsample.len() as u32).to_le_bytes())?;
    for u in &d.upsample {
        w.write_all(&(u.channels as u32).to_le_bytes())?;
        save_conv_t1d(w, &u.conv_t)?;
        save_dw_conv(w, &u.dw_conv)?;
        write_f32_vec_io(w, &u.norm_w)?;
        write_f32_vec_io(w, &u.norm_b)?;
        write_f32_vec_io(w, &u.pw1_w)?;
        write_f32_vec_io(w, &u.pw1_b)?;
        write_f32_vec_io(w, &u.pw2_w)?;
        write_f32_vec_io(w, &u.pw2_b)?;
        write_f32_vec_io(w, &u.gamma)?;
    }

    // Vocos
    save_conv1d(w, &d.vocos_init)?;
    w.write_all(&(d.vocos_blocks.len() as u32).to_le_bytes())?;
    for vb in &d.vocos_blocks {
        write_f32_vec_io(w, &vb.pre_snake_alpha)?;
        write_f32_vec_io(w, &vb.pre_snake_beta)?;
        save_conv_t1d(w, &vb.upsample)?;
        w.write_all(&(vb.residuals.len() as u32).to_le_bytes())?;
        for r in &vb.residuals {
            write_f32_vec_io(w, &r.act1_alpha)?;
            write_f32_vec_io(w, &r.act1_beta)?;
            save_conv1d(w, &r.conv1)?;
            write_f32_vec_io(w, &r.act2_alpha)?;
            write_f32_vec_io(w, &r.act2_beta)?;
            save_conv1d(w, &r.conv2)?;
        }
    }

    // Output
    write_f32_vec_io(w, &d.final_snake_alpha)?;
    write_f32_vec_io(w, &d.final_snake_beta)?;
    save_conv1d(w, &d.output_conv)?;

    // RoPE
    write_f32_vec_io(w, &d.rope_cos)?;
    write_f32_vec_io(w, &d.rope_sin)?;

    Ok(())
}

// ============================================================
// Load
// ============================================================

pub fn load_model(path: &Path) -> io::Result<(TalkerWeights, CodePredictorWeights, SpeechDecoderWeights, Option<SpeakerEncoderWeights>)> {
    let mut r = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic (expected QTTS)"));
    }
    let version = read_u32_io(&mut r)?;
    if version != VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("Version {version}, expected {VERSION}")));
    }
    let format_id = read_u8_io(&mut r)?;

    eprintln!("Loading talker...");
    let talker = load_talker(&mut r, format_id)?;
    eprintln!("Loading code predictor...");
    let predictor = load_predictor(&mut r, format_id)?;
    eprintln!("Loading speech decoder...");
    let decoder = load_decoder_weights(&mut r)?;

    // Try to load speaker encoder (optional, for backward compatibility)
    eprintln!("Loading speaker encoder...");
    let speaker_encoder = match crate::loader::load_speaker_encoder_binary(&mut r) {
        Ok(se) => {
            eprintln!("  Speaker encoder loaded from binary");
            Some(se)
        }
        Err(e) => {
            eprintln!("  Speaker encoder load error: {}", e);
            eprintln!("  Will load from safetensors if needed");
            None
        }
    };

    Ok((talker, predictor, decoder, speaker_encoder))
}

fn load_talker(r: &mut impl Read, fmt: u8) -> io::Result<TalkerWeights> {
    let num_layers = read_u32_io(r)? as usize;
    let hidden_size = read_u32_io(r)? as usize;
    let text_hidden_size = read_u32_io(r)? as usize;
    let num_heads = read_u32_io(r)? as usize;
    let num_kv_heads = read_u32_io(r)? as usize;
    let head_dim = read_u32_io(r)? as usize;
    let num_kv_groups = read_u32_io(r)? as usize;
    let sec_len = read_u32_io(r)? as usize;
    let mut mrope_section = Vec::with_capacity(sec_len);
    for _ in 0..sec_len {
        mrope_section.push(read_u32_io(r)? as usize);
    }

    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        if i % 6 == 0 { eprintln!("  Layer {i}/{num_layers}..."); }
        layers.push(TalkerLayerWeights {
            q_proj: read_weight_io(r, fmt)?,
            k_proj: read_weight_io(r, fmt)?,
            v_proj: read_weight_io(r, fmt)?,
            o_proj: read_weight_io(r, fmt)?,
            q_norm: read_f16_vec_io(r)?,
            k_norm: read_f16_vec_io(r)?,
            gate_proj: read_weight_io(r, fmt)?,
            up_proj: read_weight_io(r, fmt)?,
            down_proj: read_weight_io(r, fmt)?,
            input_norm_gamma: read_f16_vec_io(r)?,
            post_attn_norm_gamma: read_f16_vec_io(r)?,
        });
    }

    let codec_embedding = read_weight_io(r, fmt)?;
    let text_embedding = read_weight_io(r, fmt)?;
    let text_proj_fc1 = read_weight_io(r, fmt)?;
    let text_proj_fc1_bias = read_f32_vec_io(r)?;
    let text_proj_fc2 = read_weight_io(r, fmt)?;
    let text_proj_fc2_bias = read_f32_vec_io(r)?;
    let codec_head = read_weight_io(r, fmt)?;
    let final_norm_gamma = read_f16_vec_io(r)?;
    let rope_cos = read_f32_vec_io(r)?;
    let rope_sin = read_f32_vec_io(r)?;

    Ok(TalkerWeights {
        layers, codec_embedding, text_embedding,
        text_proj_fc1, text_proj_fc1_bias,
        text_proj_fc2, text_proj_fc2_bias,
        codec_head, final_norm_gamma,
        rope_cos, rope_sin,
        hidden_size, text_hidden_size, num_heads, num_kv_heads, head_dim, num_kv_groups, mrope_section,
    })
}

fn load_predictor(r: &mut impl Read, fmt: u8) -> io::Result<CodePredictorWeights> {
    let num_layers = read_u32_io(r)? as usize;
    let hidden_size = read_u32_io(r)? as usize;
    let num_heads = read_u32_io(r)? as usize;
    let num_kv_heads = read_u32_io(r)? as usize;
    let head_dim = read_u32_io(r)? as usize;
    let num_kv_groups = read_u32_io(r)? as usize;

    // Input projection (optional)
    let mut has_proj_buf = [0u8; 1];
    r.read_exact(&mut has_proj_buf)?;
    let (input_proj, input_proj_bias) = if has_proj_buf[0] == 1 {
        (Some(read_weight_io(r, fmt)?), Some(read_f32_vec_io(r)?))
    } else {
        (None, None)
    };

    let num_emb = read_u32_io(r)? as usize;
    let mut codec_embeddings = Vec::with_capacity(num_emb);
    for _ in 0..num_emb {
        codec_embeddings.push(read_weight_io(r, fmt)?);
    }

    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        layers.push(PredictorLayerWeights {
            q_proj: read_weight_io(r, fmt)?,
            k_proj: read_weight_io(r, fmt)?,
            v_proj: read_weight_io(r, fmt)?,
            o_proj: read_weight_io(r, fmt)?,
            q_norm: read_f16_vec_io(r)?,
            k_norm: read_f16_vec_io(r)?,
            gate_proj: read_weight_io(r, fmt)?,
            up_proj: read_weight_io(r, fmt)?,
            down_proj: read_weight_io(r, fmt)?,
            input_norm_gamma: read_f16_vec_io(r)?,
            post_attn_norm_gamma: read_f16_vec_io(r)?,
        });
    }

    let num_heads_lm = read_u32_io(r)? as usize;
    let mut lm_heads = Vec::with_capacity(num_heads_lm);
    for _ in 0..num_heads_lm {
        lm_heads.push(read_weight_io(r, fmt)?);
    }

    let final_norm_gamma = read_f16_vec_io(r)?;
    let rope_cos = read_f32_vec_io(r)?;
    let rope_sin = read_f32_vec_io(r)?;

    Ok(CodePredictorWeights {
        input_proj, input_proj_bias, codec_embeddings,
        layers, lm_heads, final_norm_gamma,
        rope_cos, rope_sin,
        hidden_size, num_heads, num_kv_heads, head_dim, num_kv_groups,
    })
}

fn load_conv1d(r: &mut impl Read) -> io::Result<Conv1dWeight> {
    Ok(Conv1dWeight {
        in_channels: read_u32_io(r)? as usize,
        out_channels: read_u32_io(r)? as usize,
        kernel_size: read_u32_io(r)? as usize,
        stride: read_u32_io(r)? as usize,
        padding: read_u32_io(r)? as usize,
        dilation: read_u32_io(r)? as usize,
        weight: read_f32_vec_io(r)?,
        bias: read_f32_vec_io(r)?,
    })
}

fn load_conv_t1d(r: &mut impl Read) -> io::Result<ConvTranspose1dWeight> {
    Ok(ConvTranspose1dWeight {
        in_channels: read_u32_io(r)? as usize,
        out_channels: read_u32_io(r)? as usize,
        kernel_size: read_u32_io(r)? as usize,
        stride: read_u32_io(r)? as usize,
        padding: read_u32_io(r)? as usize,
        weight: read_f32_vec_io(r)?,
        bias: read_f32_vec_io(r)?,
    })
}

fn load_dw_conv(r: &mut impl Read) -> io::Result<DepthwiseConv1dWeight> {
    Ok(DepthwiseConv1dWeight {
        channels: read_u32_io(r)? as usize,
        kernel_size: read_u32_io(r)? as usize,
        padding: read_u32_io(r)? as usize,
        weight: read_f32_vec_io(r)?,
        bias: read_f32_vec_io(r)?,
    })
}

fn load_decoder_weights(r: &mut impl Read) -> io::Result<SpeechDecoderWeights> {
    let hidden_size = read_u32_io(r)? as usize;
    let num_heads = read_u32_io(r)? as usize;
    let head_dim = read_u32_io(r)? as usize;
    let intermediate_size = read_u32_io(r)? as usize;
    let sliding_window = read_u32_io(r)? as usize;
    let codebook_dim = read_u32_io(r)? as usize;

    let first_codebook = read_f32_vec_io(r)?;
    let first_output_proj = read_f32_vec_io(r)?;
    let num_rest = read_u32_io(r)? as usize;
    let mut rest_codebooks = Vec::with_capacity(num_rest);
    for _ in 0..num_rest { rest_codebooks.push(read_f32_vec_io(r)?); }
    let rest_output_proj = read_f32_vec_io(r)?;

    let pre_conv = load_conv1d(r)?;

    let tf_input_w = read_f32_vec_io(r)?;
    let tf_input_b = read_f32_vec_io(r)?;
    let num_tf = read_u32_io(r)? as usize;
    let mut tf_layers = Vec::with_capacity(num_tf);
    for _ in 0..num_tf {
        tf_layers.push(DecoderTfLayer {
            q_proj: read_f32_vec_io(r)?,
            k_proj: read_f32_vec_io(r)?,
            v_proj: read_f32_vec_io(r)?,
            o_proj: read_f32_vec_io(r)?,
            gate_proj: read_f32_vec_io(r)?,
            up_proj: read_f32_vec_io(r)?,
            down_proj: read_f32_vec_io(r)?,
            input_norm: read_f32_vec_io(r)?,
            post_attn_norm: read_f32_vec_io(r)?,
            attn_scale: read_f32_vec_io(r)?,
            mlp_scale: read_f32_vec_io(r)?,
        });
    }
    let tf_norm = read_f32_vec_io(r)?;
    let tf_output_w = read_f32_vec_io(r)?;
    let tf_output_b = read_f32_vec_io(r)?;

    let num_up = read_u32_io(r)? as usize;
    let mut upsample = Vec::with_capacity(num_up);
    for _ in 0..num_up {
        let channels = read_u32_io(r)? as usize;
        upsample.push(UpsampleStage {
            conv_t: load_conv_t1d(r)?,
            dw_conv: load_dw_conv(r)?,
            norm_w: read_f32_vec_io(r)?,
            norm_b: read_f32_vec_io(r)?,
            pw1_w: read_f32_vec_io(r)?,
            pw1_b: read_f32_vec_io(r)?,
            pw2_w: read_f32_vec_io(r)?,
            pw2_b: read_f32_vec_io(r)?,
            gamma: read_f32_vec_io(r)?,
            channels,
        });
    }

    let vocos_init = load_conv1d(r)?;
    let num_vb = read_u32_io(r)? as usize;
    let mut vocos_blocks = Vec::with_capacity(num_vb);
    for _ in 0..num_vb {
        let pre_snake_alpha = read_f32_vec_io(r)?;
        let pre_snake_beta = read_f32_vec_io(r)?;
        let upsample_conv = load_conv_t1d(r)?;
        let num_res = read_u32_io(r)? as usize;
        let mut residuals = Vec::with_capacity(num_res);
        for _ in 0..num_res {
            residuals.push(VocosResidual {
                act1_alpha: read_f32_vec_io(r)?,
                act1_beta: read_f32_vec_io(r)?,
                conv1: load_conv1d(r)?,
                act2_alpha: read_f32_vec_io(r)?,
                act2_beta: read_f32_vec_io(r)?,
                conv2: load_conv1d(r)?,
            });
        }
        vocos_blocks.push(VocosBlock {
            pre_snake_alpha, pre_snake_beta,
            upsample: upsample_conv, residuals,
        });
    }

    let final_snake_alpha = read_f32_vec_io(r)?;
    let final_snake_beta = read_f32_vec_io(r)?;
    let output_conv = load_conv1d(r)?;
    let rope_cos = read_f32_vec_io(r)?;
    let rope_sin = read_f32_vec_io(r)?;

    Ok(SpeechDecoderWeights {
        first_codebook, first_output_proj,
        rest_codebooks, rest_output_proj,
        pre_conv,
        tf_input_w, tf_input_b,
        tf_layers, tf_norm,
        tf_output_w, tf_output_b,
        upsample,
        vocos_init, vocos_blocks,
        final_snake_alpha, final_snake_beta, output_conv,
        rope_cos, rope_sin,
        hidden_size, num_heads, head_dim,
        intermediate_size, sliding_window, codebook_dim,
    })
}

use crate::speaker_encoder::SpeakerEncoderWeights;

/// Save model with speaker encoder to binary
pub fn save_model_with_speaker(
    talker: &TalkerWeights,
    predictor: &CodePredictorWeights,
    decoder: &SpeechDecoderWeights,
    speaker_encoder: &SpeakerEncoderWeights,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, File::create(path)?);

    // Header
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    let format_id: u8 = if matches!(talker.layers[0].q_proj, Weight::Q4(_)) { 1 } else { 0 };
    w.write_all(&[format_id])?;

    eprintln!("Saving talker...");
    save_talker(&mut w, talker)?;
    eprintln!("Saving code predictor...");
    save_predictor(&mut w, predictor)?;
    eprintln!("Saving speech decoder...");
    save_decoder(&mut w, decoder)?;
    eprintln!("Saving speaker encoder...");
    save_speaker_encoder(&mut w, speaker_encoder)?;

    w.flush()?;
    Ok(())
}

fn save_speaker_encoder(w: &mut impl Write, se: &SpeakerEncoderWeights) -> io::Result<()> {
    // Save initial conv
    write_conv1d_io(w, &se.initial_conv)?;

    // Save blocks count
    w.write_all(&(se.blocks.len() as u32).to_le_bytes())?;

    // Save each Res2Net block
    for block in &se.blocks {
        // Res2Net branches (7)
        w.write_all(&(block.res2net_blocks.len() as u32).to_le_bytes())?;
        for branch in &block.res2net_blocks {
            write_conv1d_io(w, branch)?;
        }
        // TDNN layers
        write_conv1d_io(w, &block.tdnn1)?;
        write_conv1d_io(w, &block.tdnn2)?;
        // SE attention
        write_conv1d_io(w, &block.se_conv1)?;
        write_conv1d_io(w, &block.se_conv2)?;
        // Channels
        w.write_all(&(block.channels as u32).to_le_bytes())?;
    }

    // Save ASP
    write_conv1d_io(w, &se.asp.conv)?;
    write_conv1d_io(w, &se.asp.tdnn)?;

    // Save MFA
    write_conv1d_io(w, &se.mfa)?;

    // Save FC
    write_f32_vec_io(w, &se.fc_weight)?;
    write_f32_vec_io(w, &se.fc_bias)?;

    Ok(())
}

fn write_conv1d_io(w: &mut impl Write, conv: &Conv1dWeight) -> io::Result<()> {
    w.write_all(&(conv.in_channels as u32).to_le_bytes())?;
    w.write_all(&(conv.out_channels as u32).to_le_bytes())?;
    w.write_all(&(conv.kernel_size as u32).to_le_bytes())?;
    w.write_all(&(conv.stride as u32).to_le_bytes())?;
    w.write_all(&(conv.padding as u32).to_le_bytes())?;
    w.write_all(&(conv.dilation as u32).to_le_bytes())?;
    write_f32_vec_io(w, &conv.weight)?;
    // Check if bias is non-zero
    let has_bias = conv.bias.iter().any(|&x| x != 0.0);
    if has_bias {
        w.write_all(&[1u8])?; // has bias
        write_f32_vec_io(w, &conv.bias)?;
    } else {
        w.write_all(&[0u8])?; // no bias
    }
    Ok(())
}
