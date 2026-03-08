//! Direct safetensors loading — bf16 → f32 → transpose → f16/Q4.
//! Binary format loading — pre-quantized Q4/F16 binary for fast startup.
//!
//! Weight naming (from model.safetensors):
//!   talker.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
//!   talker.model.layers.{i}.self_attn.{q,k}_norm.weight
//!   talker.model.layers.{i}.mlp.{gate,up,down}_proj.weight
//!   talker.model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight
//!   talker.model.text_embedding.weight
//!   talker.model.codec_embedding.weight
//!   talker.model.norm.weight
//!   talker.text_projection.linear_fc{1,2}.{weight,bias}
//!   talker.codec_head.weight
//!   talker.code_predictor.small_to_mtp_projection.{weight,bias}
//!   talker.code_predictor.model.layers.{i}.*
//!   talker.code_predictor.model.codec_embedding.{0..14}.weight
//!   talker.code_predictor.model.norm.weight
//!   talker.code_predictor.lm_head.{0..14}.weight

use std::path::Path;
use std::collections::HashMap;
use half::f16;
use safetensors::SafeTensors;

use crate::gemv::{self, Weight, build_weight, f32_to_f16};
use crate::talker::{TalkerWeights, TalkerLayerWeights};
use crate::code_predictor::{CodePredictorWeights, PredictorLayerWeights};
use crate::conv::Conv1dWeight;
use crate::speaker_encoder::{SpeakerEncoderWeights, Res2NetBlock, ASPWeights};
use crate::rope;

// ============================================================
// Binary format loading (Q4/F16 pre-quantized)
// ============================================================

struct BinaryTensor {
    _name: String,
    shape: Vec<usize>,
    data: TensorData,
}

enum TensorData {
    F16(Vec<f16>),
    Q4 { scales: Vec<f16>, packed: Vec<u8> },
}

fn load_binary_tensors(path: &Path) -> Result<HashMap<String, BinaryTensor>, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let mut pos = 0;

    // Read header: "QORA-TTS-Q4\x00\x00\x00\x00\x00" (16 bytes)
    if data.len() < 16 {
        return Err("Binary file too small".into());
    }
    let header = &data[0..16];
    if !header.starts_with(b"QORA-TTS-Q4") {
        return Err("Invalid binary header".into());
    }
    pos += 16;

    // Read version and num_tensors
    let version = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
    pos += 4;
    let num_tensors = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
    pos += 4;

    eprintln!("Loading binary format version {version}, {num_tensors} tensors...");

    let mut tensors = HashMap::new();

    for _ in 0..num_tensors {
        // Read name
        let name_len = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
        pos += 4;
        let name = String::from_utf8(data[pos..pos+name_len].to_vec())?;
        pos += name_len;

        // Read shape
        let rank = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
        pos += 4;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            let dim = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
            pos += 4;
            shape.push(dim);
        }

        // Read format flag (0=F16, 1=Q4)
        let format = data[pos];
        pos += 1;

        let tensor_data = if format == 1 {
            // Q4
            let num_scales = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
            pos += 4;
            let mut scales = Vec::with_capacity(num_scales);
            for _ in 0..num_scales {
                let bits = u16::from_le_bytes([data[pos], data[pos+1]]);
                scales.push(f16::from_bits(bits));
                pos += 2;
            }

            let num_packed = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
            pos += 4;
            let packed = data[pos..pos+num_packed].to_vec();
            pos += num_packed;

            TensorData::Q4 { scales, packed }
        } else {
            // F16
            let num_values = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
            pos += 4;
            let mut values = Vec::with_capacity(num_values);
            for _ in 0..num_values {
                let bits = u16::from_le_bytes([data[pos], data[pos+1]]);
                values.push(f16::from_bits(bits));
                pos += 2;
            }

            TensorData::F16(values)
        };

        tensors.insert(name.clone(), BinaryTensor { _name: name, shape, data: tensor_data });
    }

    Ok(tensors)
}

/// Convert binary tensor to f32 Vec (for biases, norms)
fn binary_to_f32(tensors: &HashMap<String, BinaryTensor>, key: &str) -> Vec<f32> {
    let tensor = tensors.get(key).unwrap_or_else(|| panic!("Missing tensor: {key}"));
    match &tensor.data {
        TensorData::F16(values) => values.iter().map(|&v| v.to_f32()).collect(),
        TensorData::Q4 { .. } => panic!("Tensor {key} is Q4, expected F16"),
    }
}

/// Convert binary tensor to f16 Vec (for norms)
fn binary_to_f16(tensors: &HashMap<String, BinaryTensor>, key: &str) -> Vec<f16> {
    let tensor = tensors.get(key).unwrap_or_else(|| panic!("Missing tensor: {key}"));
    match &tensor.data {
        TensorData::F16(values) => values.clone(),
        TensorData::Q4 { .. } => panic!("Tensor {key} is Q4, expected F16"),
    }
}

/// Convert binary tensor to Weight (linear or embedding, already transposed in binary)
fn binary_to_weight(tensors: &HashMap<String, BinaryTensor>, key: &str) -> Weight {
    let tensor = tensors.get(key).unwrap_or_else(|| panic!("Missing tensor: {key}"));
    let shape = &tensor.shape;
    let k = shape[0];  // rows (input dimension)
    let n = shape[1];  // cols (output dimension)

    match &tensor.data {
        TensorData::F16(values) => {
            Weight::F16(gemv::F16Weight {
                data: values.clone(),
                k,
                n,
            })
        }
        TensorData::Q4 { scales, packed } => {
            Weight::Q4(gemv::Q4Weight {
                packed: packed.clone(),
                scales: scales.clone(),
                k,
                n,
            })
        }
    }
}

// ============================================================
// BF16 conversion helpers
// ============================================================

/// Convert raw bytes (bf16 format) to f32 Vec.
fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let num_elements = bytes.len() / 2;
    let mut out = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        // bf16 → f32: shift left by 16
        let f32_bits = (bits as u32) << 16;
        out.push(f32::from_bits(f32_bits));
    }
    out
}

/// Read a tensor from safetensors, convert bf16 → f32.
fn read_tensor(st: &SafeTensors, key: &str) -> Vec<f32> {
    let tensor = st.tensor(key)
        .unwrap_or_else(|_| panic!("Missing tensor: {key}"));
    bf16_bytes_to_f32(tensor.data())
}

/// Read a tensor as f16 Vec.
fn read_tensor_f16(st: &SafeTensors, key: &str) -> Vec<f16> {
    let data = read_tensor(st, key);
    f32_to_f16(&data)
}

/// Read a tensor and build a Weight (transposed: HF stores [out, in], we need [in, out]).
fn read_linear(st: &SafeTensors, key: &str, use_q4: bool) -> Weight {
    let tensor = st.tensor(key)
        .unwrap_or_else(|_| panic!("Missing tensor: {key}"));
    let shape = tensor.shape();
    let out_dim = shape[0];
    let in_dim = shape[1];
    let data = bf16_bytes_to_f32(tensor.data());

    // Transpose [out, in] → [in, out]
    let mut transposed = vec![0.0f32; in_dim * out_dim];
    for r in 0..out_dim {
        for c in 0..in_dim {
            transposed[c * out_dim + r] = data[r * in_dim + c];
        }
    }

    build_weight(&transposed, in_dim, out_dim, use_q4)
}

/// Read an embedding weight (no transpose needed: [vocab, dim] is already [rows, cols]).
fn read_embedding(st: &SafeTensors, key: &str, use_q4: bool) -> Weight {
    let tensor = st.tensor(key)
        .unwrap_or_else(|_| panic!("Missing tensor: {key}"));
    let shape = tensor.shape();
    let vocab = shape[0];
    let dim = shape[1];
    let data = bf16_bytes_to_f32(tensor.data());
    build_weight(&data, vocab, dim, use_q4)
}

/// Check if a tensor exists in safetensors.
fn has_tensor(st: &SafeTensors, key: &str) -> bool {
    st.tensor(key).is_ok()
}

// ============================================================
// Load talker weights
// ============================================================

pub fn load_talker(
    model_path: &Path,
    config: &crate::config::QoraTTSConfig,
    use_q4: bool,
) -> Result<TalkerWeights, Box<dyn std::error::Error>> {
    let file_data = std::fs::read(model_path.join("model.safetensors"))?;
    let st = SafeTensors::deserialize(&file_data)?;

    let tc = &config.talker_config;
    let num_layers = tc.num_hidden_layers;

    eprintln!("Loading talker weights ({num_layers} layers)...");

    // Per-layer weights
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        if i % 6 == 0 {
            eprintln!("  Layer {i}/{num_layers}...");
        }
        let prefix = format!("talker.model.layers.{i}");
        layers.push(TalkerLayerWeights {
            q_proj: read_linear(&st, &format!("{prefix}.self_attn.q_proj.weight"), use_q4),
            k_proj: read_linear(&st, &format!("{prefix}.self_attn.k_proj.weight"), use_q4),
            v_proj: read_linear(&st, &format!("{prefix}.self_attn.v_proj.weight"), use_q4),
            o_proj: read_linear(&st, &format!("{prefix}.self_attn.o_proj.weight"), use_q4),
            q_norm: read_tensor_f16(&st, &format!("{prefix}.self_attn.q_norm.weight")),
            k_norm: read_tensor_f16(&st, &format!("{prefix}.self_attn.k_norm.weight")),
            gate_proj: read_linear(&st, &format!("{prefix}.mlp.gate_proj.weight"), use_q4),
            up_proj: read_linear(&st, &format!("{prefix}.mlp.up_proj.weight"), use_q4),
            down_proj: read_linear(&st, &format!("{prefix}.mlp.down_proj.weight"), use_q4),
            input_norm_gamma: read_tensor_f16(&st, &format!("{prefix}.input_layernorm.weight")),
            post_attn_norm_gamma: read_tensor_f16(&st, &format!("{prefix}.post_attention_layernorm.weight")),
        });
    }

    // Global weights
    let codec_embedding = read_embedding(&st, "talker.model.codec_embedding.weight", use_q4);
    let text_embedding = read_embedding(&st, "talker.model.text_embedding.weight", use_q4);
    let text_proj_fc1 = read_linear(&st, "talker.text_projection.linear_fc1.weight", use_q4);
    let text_proj_fc1_bias = read_tensor(&st, "talker.text_projection.linear_fc1.bias");
    let text_proj_fc2 = read_linear(&st, "talker.text_projection.linear_fc2.weight", use_q4);
    let text_proj_fc2_bias = read_tensor(&st, "talker.text_projection.linear_fc2.bias");
    let codec_head = read_linear(&st, "talker.codec_head.weight", use_q4);
    let final_norm_gamma = read_tensor_f16(&st, "talker.model.norm.weight");

    eprintln!("  Building RoPE tables...");

    // Build RoPE tables
    let head_dim = tc.head_dim;
    let max_pos = tc.max_position_embeddings;
    let theta = tc.rope_theta;
    let mrope_section = tc.mrope_section().to_vec();
    let (rope_cos, rope_sin) = rope::build_mrope_tables(head_dim, max_pos, theta, &mrope_section);

    eprintln!("  Talker loaded: {} layers, hidden={}", num_layers, tc.hidden_size);

    Ok(TalkerWeights {
        layers,
        codec_embedding,
        text_embedding,
        text_proj_fc1,
        text_proj_fc1_bias,
        text_proj_fc2,
        text_proj_fc2_bias,
        codec_head,
        final_norm_gamma,
        rope_cos,
        rope_sin,
        hidden_size: tc.hidden_size,
        text_hidden_size: tc.text_hidden_size,
        num_heads: tc.num_attention_heads,
        num_kv_heads: tc.num_key_value_heads,
        head_dim,
        num_kv_groups: tc.num_kv_groups(),
        mrope_section,
    })
}

// ============================================================
// Load code predictor weights
// ============================================================

pub fn load_code_predictor(
    model_path: &Path,
    config: &crate::config::QoraTTSConfig,
    use_q4: bool,
) -> Result<CodePredictorWeights, Box<dyn std::error::Error>> {
    let file_data = std::fs::read(model_path.join("model.safetensors"))?;
    let st = SafeTensors::deserialize(&file_data)?;

    let cp = config.talker_config.code_predictor_config.as_ref()
        .expect("Missing code_predictor_config");
    let num_layers = cp.num_hidden_layers;

    eprintln!("Loading code predictor ({num_layers} layers)...");

    // Input projection (optional — absent when talker_hidden == cp_hidden)
    let input_proj = if has_tensor(&st, "talker.code_predictor.small_to_mtp_projection.weight") {
        Some(read_linear(&st, "talker.code_predictor.small_to_mtp_projection.weight", use_q4))
    } else {
        eprintln!("  No input_proj found (talker_hidden == cp_hidden), skipping projection");
        None
    };
    let input_proj_bias = if has_tensor(&st, "talker.code_predictor.small_to_mtp_projection.bias") {
        Some(read_tensor(&st, "talker.code_predictor.small_to_mtp_projection.bias"))
    } else {
        None
    };

    // Codec embeddings (15)
    let mut codec_embeddings = Vec::with_capacity(15);
    for g in 0..15 {
        codec_embeddings.push(read_embedding(
            &st,
            &format!("talker.code_predictor.model.codec_embedding.{g}.weight"),
            use_q4,
        ));
    }

    // Per-layer weights
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let prefix = format!("talker.code_predictor.model.layers.{i}");
        layers.push(PredictorLayerWeights {
            q_proj: read_linear(&st, &format!("{prefix}.self_attn.q_proj.weight"), use_q4),
            k_proj: read_linear(&st, &format!("{prefix}.self_attn.k_proj.weight"), use_q4),
            v_proj: read_linear(&st, &format!("{prefix}.self_attn.v_proj.weight"), use_q4),
            o_proj: read_linear(&st, &format!("{prefix}.self_attn.o_proj.weight"), use_q4),
            q_norm: read_tensor_f16(&st, &format!("{prefix}.self_attn.q_norm.weight")),
            k_norm: read_tensor_f16(&st, &format!("{prefix}.self_attn.k_norm.weight")),
            gate_proj: read_linear(&st, &format!("{prefix}.mlp.gate_proj.weight"), use_q4),
            up_proj: read_linear(&st, &format!("{prefix}.mlp.up_proj.weight"), use_q4),
            down_proj: read_linear(&st, &format!("{prefix}.mlp.down_proj.weight"), use_q4),
            input_norm_gamma: read_tensor_f16(&st, &format!("{prefix}.input_layernorm.weight")),
            post_attn_norm_gamma: read_tensor_f16(&st, &format!("{prefix}.post_attention_layernorm.weight")),
        });
    }

    // lm_heads (15)
    let mut lm_heads = Vec::with_capacity(15);
    for g in 0..15 {
        lm_heads.push(read_linear(
            &st,
            &format!("talker.code_predictor.lm_head.{g}.weight"),
            use_q4,
        ));
    }

    let final_norm_gamma = read_tensor_f16(&st, "talker.code_predictor.model.norm.weight");

    // Build RoPE tables for code predictor (standard, not multimodal)
    let head_dim = cp.head_dim;
    let half_dim = head_dim / 2;
    let max_pos = cp.max_position_embeddings.max(65536);
    let theta = cp.rope_theta;

    let mut rope_cos = vec![0.0f32; max_pos * half_dim];
    let mut rope_sin = vec![0.0f32; max_pos * half_dim];
    for i in 0..half_dim {
        let freq = 1.0 / theta.powf((2 * i) as f64 / head_dim as f64);
        for pos in 0..max_pos {
            let angle = pos as f64 * freq;
            rope_cos[pos * half_dim + i] = angle.cos() as f32;
            rope_sin[pos * half_dim + i] = angle.sin() as f32;
        }
    }

    eprintln!("  Code predictor loaded: {} layers, hidden={}", num_layers, cp.hidden_size);

    Ok(CodePredictorWeights {
        input_proj,
        input_proj_bias,
        codec_embeddings,
        layers,
        lm_heads,
        final_norm_gamma,
        rope_cos,
        rope_sin,
        hidden_size: cp.hidden_size,
        num_heads: cp.num_attention_heads,
        num_kv_heads: cp.num_key_value_heads,
        head_dim,
        num_kv_groups: cp.num_kv_groups(),
    })
}

// ============================================================
// Load speech decoder weights (from speech_tokenizer/)
// ============================================================

/// Read f32 tensor from safetensors (handling both f32 and bf16).
fn read_tensor_f32(st: &SafeTensors, key: &str) -> Vec<f32> {
    let tensor = st.tensor(key)
        .unwrap_or_else(|_| panic!("Missing tensor: {key}"));
    match tensor.dtype() {
        safetensors::Dtype::F32 => {
            let bytes = tensor.data();
            let num = bytes.len() / 4;
            let mut out = Vec::with_capacity(num);
            for i in 0..num {
                let bits = u32::from_le_bytes([
                    bytes[i*4], bytes[i*4+1], bytes[i*4+2], bytes[i*4+3],
                ]);
                out.push(f32::from_bits(bits));
            }
            out
        }
        safetensors::Dtype::BF16 => bf16_bytes_to_f32(tensor.data()),
        safetensors::Dtype::F16 => {
            let bytes = tensor.data();
            let num = bytes.len() / 2;
            let mut out = Vec::with_capacity(num);
            for i in 0..num {
                let bits = u16::from_le_bytes([bytes[i*2], bytes[i*2+1]]);
                out.push(f16::from_bits(bits).to_f32());
            }
            out
        }
        dt => panic!("Unsupported dtype {dt:?} for tensor {key}"),
    }
}

/// Transpose [out, in] → [in, out] for f32_gemv.
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Read linear weight, transpose for gemv: [out, in] → [in, out].
fn read_linear_f32(st: &SafeTensors, key: &str) -> Vec<f32> {
    let tensor = st.tensor(key)
        .unwrap_or_else(|_| panic!("Missing tensor: {key}"));
    let shape = tensor.shape();
    let out_dim = shape[0];
    let in_dim = shape[1];
    let data = read_tensor_f32(st, key);
    transpose_2d(&data, out_dim, in_dim)
}

/// Compute actual codebook from EMA: embedding = embedding_sum / cluster_usage.
fn ema_codebook(st: &SafeTensors, sum_key: &str, usage_key: &str) -> Vec<f32> {
    let emb_sum = read_tensor_f32(st, sum_key);
    let usage = read_tensor_f32(st, usage_key);
    let vocab = usage.len();
    let dim = emb_sum.len() / vocab;
    let mut codebook = vec![0.0f32; vocab * dim];
    for i in 0..vocab {
        let u = usage[i].max(1e-8); // avoid division by zero
        for d in 0..dim {
            codebook[i * dim + d] = emb_sum[i * dim + d] / u;
        }
    }
    codebook
}

pub fn load_speech_decoder(
    model_path: &Path,
) -> Result<crate::decoder::SpeechDecoderWeights, Box<dyn std::error::Error>> {
    use crate::conv::*;
    use crate::decoder::*;

    let st_path = model_path.join("speech_tokenizer").join("model.safetensors");
    if !st_path.exists() {
        return Err(format!("Speech decoder not found: {}", st_path.display()).into());
    }

    let file_data = std::fs::read(&st_path)?;
    let st = SafeTensors::deserialize(&file_data)?;

    eprintln!("Loading speech decoder ({} tensors)...", st.len());

    // ---------- Codebooks (VQ) ----------
    eprintln!("  Loading codebooks...");
    let first_codebook = ema_codebook(
        &st,
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage",
    );
    // output_proj: Conv1d [512, 256, 1] → treat as [512, 256] matrix, transpose for gemv
    let first_out_proj_raw = read_tensor_f32(&st, "decoder.quantizer.rvq_first.output_proj.weight");
    // Shape [512, 256, 1] → remove k=1 dim → [512, 256] → transpose → [256, 512]
    let first_output_proj = transpose_2d(&first_out_proj_raw, 512, 256);

    let mut rest_codebooks = Vec::with_capacity(15);
    for g in 0..15 {
        rest_codebooks.push(ema_codebook(
            &st,
            &format!("decoder.quantizer.rvq_rest.vq.layers.{g}._codebook.embedding_sum"),
            &format!("decoder.quantizer.rvq_rest.vq.layers.{g}._codebook.cluster_usage"),
        ));
    }
    let rest_out_proj_raw = read_tensor_f32(&st, "decoder.quantizer.rvq_rest.output_proj.weight");
    let rest_output_proj = transpose_2d(&rest_out_proj_raw, 512, 256);

    // ---------- Pre-conv ----------
    eprintln!("  Loading pre-conv...");
    let pre_conv = Conv1dWeight {
        weight: read_tensor_f32(&st, "decoder.pre_conv.conv.weight"),
        bias: read_tensor_f32(&st, "decoder.pre_conv.conv.bias"),
        in_channels: 512, out_channels: 1024, kernel_size: 3,
        stride: 1, padding: 0, dilation: 1,
    };

    // ---------- Pre-transformer ----------
    eprintln!("  Loading transformer (8 layers)...");
    let tf_input_w = read_linear_f32(&st, "decoder.pre_transformer.input_proj.weight");
    let tf_input_b = read_tensor_f32(&st, "decoder.pre_transformer.input_proj.bias");
    let tf_output_w = read_linear_f32(&st, "decoder.pre_transformer.output_proj.weight");
    let tf_output_b = read_tensor_f32(&st, "decoder.pre_transformer.output_proj.bias");
    let tf_norm = read_tensor_f32(&st, "decoder.pre_transformer.norm.weight");

    let mut tf_layers = Vec::with_capacity(8);
    for i in 0..8 {
        let p = format!("decoder.pre_transformer.layers.{i}");
        tf_layers.push(DecoderTfLayer {
            q_proj: read_linear_f32(&st, &format!("{p}.self_attn.q_proj.weight")),
            k_proj: read_linear_f32(&st, &format!("{p}.self_attn.k_proj.weight")),
            v_proj: read_linear_f32(&st, &format!("{p}.self_attn.v_proj.weight")),
            o_proj: read_linear_f32(&st, &format!("{p}.self_attn.o_proj.weight")),
            gate_proj: read_linear_f32(&st, &format!("{p}.mlp.gate_proj.weight")),
            up_proj: read_linear_f32(&st, &format!("{p}.mlp.up_proj.weight")),
            down_proj: read_linear_f32(&st, &format!("{p}.mlp.down_proj.weight")),
            input_norm: read_tensor_f32(&st, &format!("{p}.input_layernorm.weight")),
            post_attn_norm: read_tensor_f32(&st, &format!("{p}.post_attention_layernorm.weight")),
            attn_scale: read_tensor_f32(&st, &format!("{p}.self_attn_layer_scale.scale")),
            mlp_scale: read_tensor_f32(&st, &format!("{p}.mlp_layer_scale.scale")),
        });
    }

    // ---------- Upsample (2 stages) ----------
    eprintln!("  Loading upsample...");
    let mut upsample = Vec::with_capacity(2);
    for i in 0..2 {
        let ct = format!("decoder.upsample.{i}.0");
        let cn = format!("decoder.upsample.{i}.1");
        upsample.push(UpsampleStage {
            conv_t: ConvTranspose1dWeight {
                weight: read_tensor_f32(&st, &format!("{ct}.conv.weight")),
                bias: read_tensor_f32(&st, &format!("{ct}.conv.bias")),
                in_channels: 1024, out_channels: 1024, kernel_size: 2,
                stride: 2, padding: 0,
            },
            dw_conv: DepthwiseConv1dWeight {
                weight: read_tensor_f32(&st, &format!("{cn}.dwconv.conv.weight")),
                bias: read_tensor_f32(&st, &format!("{cn}.dwconv.conv.bias")),
                channels: 1024, kernel_size: 7, padding: 3, // same-padding
            },
            norm_w: read_tensor_f32(&st, &format!("{cn}.norm.weight")),
            norm_b: read_tensor_f32(&st, &format!("{cn}.norm.bias")),
            pw1_w: transpose_2d(&read_tensor_f32(&st, &format!("{cn}.pwconv1.weight")), 4096, 1024),
            pw1_b: read_tensor_f32(&st, &format!("{cn}.pwconv1.bias")),
            pw2_w: transpose_2d(&read_tensor_f32(&st, &format!("{cn}.pwconv2.weight")), 1024, 4096),
            pw2_b: read_tensor_f32(&st, &format!("{cn}.pwconv2.bias")),
            gamma: read_tensor_f32(&st, &format!("{cn}.gamma")),
            channels: 1024,
        });
    }

    // ---------- Vocos ----------
    eprintln!("  Loading vocos...");

    // Initial conv: CausalConv1d(1024→1536, k=7)
    let vocos_init = Conv1dWeight {
        weight: read_tensor_f32(&st, "decoder.decoder.0.conv.weight"),
        bias: read_tensor_f32(&st, "decoder.decoder.0.conv.bias"),
        in_channels: 1024, out_channels: 1536, kernel_size: 7,
        stride: 1, padding: 0, dilation: 1,
    };

    // Vocos blocks: decoder.decoder.{1,2,3,4}
    let vocos_dims = [(1536, 768, 16, 8), (768, 384, 10, 5), (384, 192, 8, 4), (192, 96, 6, 3)];
    let mut vocos_blocks = Vec::with_capacity(4);
    for (bi, &(in_ch, out_ch, k_up, stride_up)) in vocos_dims.iter().enumerate() {
        let b = format!("decoder.decoder.{}", bi + 1);
        let dilations = [1usize, 3, 9]; // V2 decoder uses dilations (1, 3, 9)
        let mut residuals = Vec::with_capacity(3);
        for ri in 0..3 {
            let r = format!("{b}.block.{}", ri + 2);
            residuals.push(VocosResidual {
                act1_alpha: read_tensor_f32(&st, &format!("{r}.act1.alpha")),
                act1_beta: read_tensor_f32(&st, &format!("{r}.act1.beta")),
                conv1: Conv1dWeight {
                    weight: read_tensor_f32(&st, &format!("{r}.conv1.conv.weight")),
                    bias: read_tensor_f32(&st, &format!("{r}.conv1.conv.bias")),
                    in_channels: out_ch, out_channels: out_ch, kernel_size: 7,
                    stride: 1, padding: 0, dilation: dilations[ri],
                },
                act2_alpha: read_tensor_f32(&st, &format!("{r}.act2.alpha")),
                act2_beta: read_tensor_f32(&st, &format!("{r}.act2.beta")),
                conv2: Conv1dWeight {
                    weight: read_tensor_f32(&st, &format!("{r}.conv2.conv.weight")),
                    bias: read_tensor_f32(&st, &format!("{r}.conv2.conv.bias")),
                    in_channels: out_ch, out_channels: out_ch, kernel_size: 1,
                    stride: 1, padding: 0, dilation: 1,
                },
            });
        }
        vocos_blocks.push(VocosBlock {
            pre_snake_alpha: read_tensor_f32(&st, &format!("{b}.block.0.alpha")),
            pre_snake_beta: read_tensor_f32(&st, &format!("{b}.block.0.beta")),
            upsample: ConvTranspose1dWeight {
                weight: read_tensor_f32(&st, &format!("{b}.block.1.conv.weight")),
                bias: read_tensor_f32(&st, &format!("{b}.block.1.conv.bias")),
                in_channels: in_ch, out_channels: out_ch, kernel_size: k_up,
                stride: stride_up, padding: 0,
            },
            residuals,
        });
    }

    // Output: SnakeBeta(96) + Conv1d(96→1, k=7)
    let final_snake_alpha = read_tensor_f32(&st, "decoder.decoder.5.alpha");
    let final_snake_beta = read_tensor_f32(&st, "decoder.decoder.5.beta");
    let output_conv = Conv1dWeight {
        weight: read_tensor_f32(&st, "decoder.decoder.6.conv.weight"),
        bias: read_tensor_f32(&st, "decoder.decoder.6.conv.bias"),
        in_channels: 96, out_channels: 1, kernel_size: 7,
        stride: 1, padding: 0, dilation: 1,
    };

    // Build RoPE tables (standard, not multimodal)
    let head_dim: usize = 64;
    let half_dim = head_dim / 2;
    let max_pos: usize = 8000;
    let theta: f64 = 10000.0;
    let mut rope_cos = vec![0.0f32; max_pos * half_dim];
    let mut rope_sin = vec![0.0f32; max_pos * half_dim];
    for i in 0..half_dim {
        let freq = 1.0 / theta.powf((2 * i) as f64 / head_dim as f64);
        for pos in 0..max_pos {
            let angle = pos as f64 * freq;
            rope_cos[pos * half_dim + i] = angle.cos() as f32;
            rope_sin[pos * half_dim + i] = angle.sin() as f32;
        }
    }

    let total_mb = {
        let mut t = 0usize;
        t += first_codebook.len() * 4 + first_output_proj.len() * 4;
        for cb in &rest_codebooks { t += cb.len() * 4; }
        t += rest_output_proj.len() * 4;
        t += pre_conv.weight.len() * 4;
        t += tf_input_w.len() * 4 + tf_output_w.len() * 4;
        for l in &tf_layers {
            t += (l.q_proj.len() + l.k_proj.len() + l.v_proj.len() + l.o_proj.len()) * 4;
            t += (l.gate_proj.len() + l.up_proj.len() + l.down_proj.len()) * 4;
        }
        for u in &upsample { t += u.conv_t.weight.len() * 4 + u.pw1_w.len() * 4 + u.pw2_w.len() * 4; }
        t += vocos_init.weight.len() * 4;
        for vb in &vocos_blocks { t += vb.upsample.weight.len() * 4; }
        t += output_conv.weight.len() * 4;
        t / (1024 * 1024)
    };
    eprintln!("  Speech decoder loaded (~{total_mb} MB)");

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
        hidden_size: 512, num_heads: 16, head_dim: 64,
        intermediate_size: 1024, sliding_window: 72,
        codebook_dim: 256,
    })
}

// ============================================================
// Load from binary format (fast)
// ============================================================

pub fn load_talker_binary(
    model_path: &Path,
    config: &crate::config::QoraTTSConfig,
) -> Result<TalkerWeights, Box<dyn std::error::Error>> {
    let binary_path = model_path.join("model.qora-tts");
    let tensors = load_binary_tensors(&binary_path)?;

    let tc = &config.talker_config;
    let num_layers = tc.num_hidden_layers;

    eprintln!("Loading talker from binary ({num_layers} layers)...");

    // Per-layer weights
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        if i % 6 == 0 {
            eprintln!("  Layer {i}/{num_layers}...");
        }
        let prefix = format!("talker.model.layers.{i}");
        layers.push(TalkerLayerWeights {
            q_proj: binary_to_weight(&tensors, &format!("{prefix}.self_attn.q_proj.weight")),
            k_proj: binary_to_weight(&tensors, &format!("{prefix}.self_attn.k_proj.weight")),
            v_proj: binary_to_weight(&tensors, &format!("{prefix}.self_attn.v_proj.weight")),
            o_proj: binary_to_weight(&tensors, &format!("{prefix}.self_attn.o_proj.weight")),
            q_norm: binary_to_f16(&tensors, &format!("{prefix}.self_attn.q_norm.weight")),
            k_norm: binary_to_f16(&tensors, &format!("{prefix}.self_attn.k_norm.weight")),
            gate_proj: binary_to_weight(&tensors, &format!("{prefix}.mlp.gate_proj.weight")),
            up_proj: binary_to_weight(&tensors, &format!("{prefix}.mlp.up_proj.weight")),
            down_proj: binary_to_weight(&tensors, &format!("{prefix}.mlp.down_proj.weight")),
            input_norm_gamma: binary_to_f16(&tensors, &format!("{prefix}.input_layernorm.weight")),
            post_attn_norm_gamma: binary_to_f16(&tensors, &format!("{prefix}.post_attention_layernorm.weight")),
        });
    }

    // Global weights
    let codec_embedding = binary_to_weight(&tensors, "talker.model.codec_embedding.weight");
    let text_embedding = binary_to_weight(&tensors, "talker.model.text_embedding.weight");
    let text_proj_fc1 = binary_to_weight(&tensors, "talker.text_projection.linear_fc1.weight");
    let text_proj_fc1_bias = binary_to_f32(&tensors, "talker.text_projection.linear_fc1.bias");
    let text_proj_fc2 = binary_to_weight(&tensors, "talker.text_projection.linear_fc2.weight");
    let text_proj_fc2_bias = binary_to_f32(&tensors, "talker.text_projection.linear_fc2.bias");
    let codec_head = binary_to_weight(&tensors, "talker.codec_head.weight");
    let final_norm_gamma = binary_to_f16(&tensors, "talker.model.norm.weight");

    eprintln!("  Building RoPE tables...");

    // Build RoPE tables
    let head_dim = tc.head_dim;
    let max_pos = tc.max_position_embeddings;
    let theta = tc.rope_theta;
    let mrope_section = tc.mrope_section().to_vec();
    let (rope_cos, rope_sin) = rope::build_mrope_tables(head_dim, max_pos, theta, &mrope_section);

    eprintln!("  Talker loaded: {} layers, hidden={}", num_layers, tc.hidden_size);

    Ok(TalkerWeights {
        layers,
        codec_embedding,
        text_embedding,
        text_proj_fc1,
        text_proj_fc1_bias,
        text_proj_fc2,
        text_proj_fc2_bias,
        codec_head,
        final_norm_gamma,
        rope_cos,
        rope_sin,
        hidden_size: tc.hidden_size,
        text_hidden_size: tc.text_hidden_size,
        num_heads: tc.num_attention_heads,
        num_kv_heads: tc.num_key_value_heads,
        head_dim,
        num_kv_groups: tc.num_kv_groups(),
        mrope_section,
    })
}

pub fn load_code_predictor_binary(
    model_path: &Path,
    config: &crate::config::QoraTTSConfig,
) -> Result<CodePredictorWeights, Box<dyn std::error::Error>> {
    let binary_path = model_path.join("model.qora-tts");
    let tensors = load_binary_tensors(&binary_path)?;

    let cp = config.talker_config.code_predictor_config.as_ref()
        .expect("Missing code_predictor_config");
    let num_layers = cp.num_hidden_layers;

    eprintln!("Loading code predictor from binary ({num_layers} layers)...");

    // Input projection (optional — absent when talker_hidden == cp_hidden)
    let input_proj = tensors.get("talker.code_predictor.small_to_mtp_projection.weight")
        .map(|_| binary_to_weight(&tensors, "talker.code_predictor.small_to_mtp_projection.weight"));
    let input_proj_bias = tensors.get("talker.code_predictor.small_to_mtp_projection.bias")
        .map(|_| binary_to_f32(&tensors, "talker.code_predictor.small_to_mtp_projection.bias"));

    // Codec embeddings (15)
    let mut codec_embeddings = Vec::with_capacity(15);
    for g in 0..15 {
        codec_embeddings.push(binary_to_weight(
            &tensors,
            &format!("talker.code_predictor.model.codec_embedding.{g}.weight"),
        ));
    }

    // Per-layer weights
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let prefix = format!("talker.code_predictor.model.layers.{i}");
        layers.push(PredictorLayerWeights {
            q_proj: binary_to_weight(&tensors, &format!("{prefix}.self_attn.q_proj.weight")),
            k_proj: binary_to_weight(&tensors, &format!("{prefix}.self_attn.k_proj.weight")),
            v_proj: binary_to_weight(&tensors, &format!("{prefix}.self_attn.v_proj.weight")),
            o_proj: binary_to_weight(&tensors, &format!("{prefix}.self_attn.o_proj.weight")),
            q_norm: binary_to_f16(&tensors, &format!("{prefix}.self_attn.q_norm.weight")),
            k_norm: binary_to_f16(&tensors, &format!("{prefix}.self_attn.k_norm.weight")),
            gate_proj: binary_to_weight(&tensors, &format!("{prefix}.mlp.gate_proj.weight")),
            up_proj: binary_to_weight(&tensors, &format!("{prefix}.mlp.up_proj.weight")),
            down_proj: binary_to_weight(&tensors, &format!("{prefix}.mlp.down_proj.weight")),
            input_norm_gamma: binary_to_f16(&tensors, &format!("{prefix}.input_layernorm.weight")),
            post_attn_norm_gamma: binary_to_f16(&tensors, &format!("{prefix}.post_attention_layernorm.weight")),
        });
    }

    // lm_heads (15)
    let mut lm_heads = Vec::with_capacity(15);
    for g in 0..15 {
        lm_heads.push(binary_to_weight(
            &tensors,
            &format!("talker.code_predictor.lm_head.{g}.weight"),
        ));
    }

    let final_norm_gamma = binary_to_f16(&tensors, "talker.code_predictor.model.norm.weight");

    // Build RoPE tables for code predictor (standard, not multimodal)
    let head_dim = cp.head_dim;
    let half_dim = head_dim / 2;
    let max_pos = cp.max_position_embeddings.max(65536);
    let theta = cp.rope_theta;

    let mut rope_cos = vec![0.0f32; max_pos * half_dim];
    let mut rope_sin = vec![0.0f32; max_pos * half_dim];
    for i in 0..half_dim {
        let freq = 1.0 / theta.powf((2 * i) as f64 / head_dim as f64);
        for pos in 0..max_pos {
            let angle = pos as f64 * freq;
            rope_cos[pos * half_dim + i] = angle.cos() as f32;
            rope_sin[pos * half_dim + i] = angle.sin() as f32;
        }
    }

    eprintln!("  Code predictor loaded: {} layers, hidden={}", num_layers, cp.hidden_size);

    Ok(CodePredictorWeights {
        input_proj,
        input_proj_bias,
        codec_embeddings,
        layers,
        lm_heads,
        final_norm_gamma,
        rope_cos,
        rope_sin,
        hidden_size: cp.hidden_size,
        num_heads: cp.num_attention_heads,
        num_kv_heads: cp.num_key_value_heads,
        head_dim,
        num_kv_groups: cp.num_kv_groups(),
    })
}

// ============================================================
// Speaker Encoder Loading
// ============================================================


pub fn load_speaker_encoder(
    model_dir: &Path,
) -> Result<SpeakerEncoderWeights, Box<dyn std::error::Error>> {
    let weights_path = model_dir.join("model.safetensors");
    let buffer = std::fs::read(&weights_path)?;
    let st = SafeTensors::deserialize(&buffer)?;

    eprintln!("Loading speaker encoder (ECAPA-TDNN)...");

    // Initial conv: [512, 128, 5]
    let initial_conv = load_conv1d_weight(&st, "speaker_encoder.blocks.0.conv")?;

    // 3 SE-Res2Net blocks with dilations [2, 3, 4]
    let dilations = [2, 3, 4];
    let mut blocks = Vec::new();
    for (idx, &dil) in (1..=3).zip(dilations.iter()) {
        blocks.push(load_res2net_block(&st, idx, dil)?);
    }

    // ASP
    let asp = ASPWeights {
        conv: load_conv1d_weight(&st, "speaker_encoder.asp.conv")?,
        tdnn: load_conv1d_weight(&st, "speaker_encoder.asp.tdnn.conv")?,
    };

    // MFA
    let mfa = load_conv1d_weight(&st, "speaker_encoder.mfa.conv")?;

    // FC
    let fc_weight = read_tensor_f32(&st, "speaker_encoder.fc.weight");
    let fc_bias = read_tensor_f32(&st, "speaker_encoder.fc.bias");

    eprintln!("  Speaker encoder loaded: 3 Res2Net blocks, output dim=2048");

    Ok(SpeakerEncoderWeights {
        initial_conv,
        blocks,
        asp,
        mfa,
        fc_weight,
        fc_bias,
    })
}

fn load_res2net_block(
    st: &SafeTensors,
    block_idx: usize,
    dilation: usize,
) -> Result<Res2NetBlock, Box<dyn std::error::Error>> {
    let prefix = format!("speaker_encoder.blocks.{block_idx}");

    // 7 Res2Net branches (scale=8, 7 convolutions + 1 passthrough)
    let mut res2net_blocks = Vec::new();
    for i in 0..7 {
        let mut conv = load_conv1d_weight(st, &format!("{prefix}.res2net_block.blocks.{i}.conv"))?;
        conv.dilation = dilation;  // Set block-specific dilation
        res2net_blocks.push(conv);
    }

    // TDNN layers
    let tdnn1 = load_conv1d_weight(st, &format!("{prefix}.tdnn1.conv"))?;
    let tdnn2 = load_conv1d_weight(st, &format!("{prefix}.tdnn2.conv"))?;

    // SE blocks
    let se_conv1 = load_conv1d_weight(st, &format!("{prefix}.se_block.conv1"))?;
    let se_conv2 = load_conv1d_weight(st, &format!("{prefix}.se_block.conv2"))?;

    Ok(Res2NetBlock {
        res2net_blocks,
        tdnn1,
        tdnn2,
        se_conv1,
        se_conv2,
        channels: 512,
    })
}

fn load_conv1d_weight(
    st: &SafeTensors,
    prefix: &str,
) -> Result<crate::conv::Conv1dWeight, Box<dyn std::error::Error>> {
    let weight = read_tensor_f32(st, &format!("{prefix}.weight"));
    let bias = read_tensor_f32(st, &format!("{prefix}.bias"));

    // Infer shape from weight: [out_ch, in_ch, kernel]
    let weight_view = st.tensor(&format!("{prefix}.weight"))?;
    let shape = weight_view.shape();

    if shape.len() != 3 {
        return Err(format!("Expected 3D weight, got {:?}", shape).into());
    }

    let out_channels = shape[0];
    let in_channels = shape[1];
    let kernel_size = shape[2];

    Ok(crate::conv::Conv1dWeight {
        weight,
        bias,
        in_channels,
        out_channels,
        kernel_size,
        stride: 1,  // Default
        padding: 0,  // Will be handled by conv1d_padded
        dilation: 1,  // Default
    })
}

fn read_u32_loader(r: &mut impl std::io::Read) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}



/// Load speaker encoder from binary file
pub fn load_speaker_encoder_binary(r: &mut impl std::io::Read) -> std::io::Result<SpeakerEncoderWeights> {
    // Load initial conv
    let initial_conv = read_conv1d_io(r)?;

    // Load blocks count
    let num_blocks = read_u32_loader(r)? as usize;
    let mut blocks = Vec::with_capacity(num_blocks);

    // Load each Res2Net block
    for _ in 0..num_blocks {
        // Res2Net branches
        let num_branches = read_u32_loader(r)? as usize;
        let mut res2net_blocks = Vec::with_capacity(num_branches);
        for _ in 0..num_branches {
            res2net_blocks.push(read_conv1d_io(r)?);
        }

        // TDNN layers
        let tdnn1 = read_conv1d_io(r)?;
        let tdnn2 = read_conv1d_io(r)?;

        // SE attention
        let se_conv1 = read_conv1d_io(r)?;
        let se_conv2 = read_conv1d_io(r)?;

        // Channels
        let channels = read_u32_loader(r)? as usize;

        blocks.push(crate::speaker_encoder::Res2NetBlock {
            res2net_blocks,
            tdnn1,
            tdnn2,
            se_conv1,
            se_conv2,
            channels,
        });
    }

    // Load ASP
    let asp_conv = read_conv1d_io(r)?;
    let asp_tdnn = read_conv1d_io(r)?;
    let asp = crate::speaker_encoder::ASPWeights {
        conv: asp_conv,
        tdnn: asp_tdnn,
    };

    // Load MFA
    let mfa = read_conv1d_io(r)?;

    // Load FC (uses u64 length prefix from write_f32_vec_io)
    let fc_weight = crate::gemv::read_f32_vec_io(r)?;
    let fc_bias = crate::gemv::read_f32_vec_io(r)?;

    Ok(SpeakerEncoderWeights {
        initial_conv,
        blocks,
        asp,
        mfa,
        fc_weight,
        fc_bias,
    })
}

fn read_conv1d_io(r: &mut impl std::io::Read) -> std::io::Result<Conv1dWeight> {
    let in_channels = read_u32_loader(r)? as usize;
    let out_channels = read_u32_loader(r)? as usize;
    let kernel_size = read_u32_loader(r)? as usize;
    let stride = read_u32_loader(r)? as usize;
    let padding = read_u32_loader(r)? as usize;
    let dilation = read_u32_loader(r)? as usize;
    // Note: write_conv1d_io uses write_f32_vec_io from gemv.rs which writes u64 length prefix
    let weight = crate::gemv::read_f32_vec_io(r)?;

    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    let has_bias = buf[0] == 1;

    let bias = if has_bias {
        crate::gemv::read_f32_vec_io(r)?
    } else {
        vec![0.0; out_channels]
    };

    Ok(Conv1dWeight {
        weight,
        bias,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
    })
}
