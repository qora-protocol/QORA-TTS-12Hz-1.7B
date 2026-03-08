//! Speaker Encoder (ECAPA-TDNN) for voice cloning
//!
//! Architecture: Res2Net + SE attention + ASP pooling
//! Input: [128, T] mel-spectrogram
//! Output: [enc_dim] speaker embedding
//!
//! ECAPA-TDNN speaker encoder for QORA-TTS voice cloning
//! Flow: InitialConv → 3 SE-Res2Net blocks → MFA(cat block outputs) → ASP → FC

use crate::conv::*;


// ============================================================
// Weight structures
// ============================================================

pub struct SpeakerEncoderWeights {
    pub initial_conv: Conv1dWeight,          // [512, 128, 5]
    pub blocks: Vec<Res2NetBlock>,           // 3 SE-Res2Net blocks
    pub asp: ASPWeights,                     // Attentive Statistics Pooling
    pub mfa: Conv1dWeight,                   // [1536, 1536, 1] multi-layer feature aggregation
    pub fc_weight: Vec<f32>,                 // [enc_dim, 3072]
    pub fc_bias: Vec<f32>,                   // [enc_dim]
}

pub struct Res2NetBlock {
    pub res2net_blocks: Vec<Conv1dWeight>,   // scale-1 branches (7 convolutions for scale=8)
    pub tdnn1: Conv1dWeight,
    pub tdnn2: Conv1dWeight,
    pub se_conv1: Conv1dWeight,              // SE squeeze
    pub se_conv2: Conv1dWeight,              // SE excite
    pub channels: usize,                     // 512
}

pub struct ASPWeights {
    pub conv: Conv1dWeight,                  // Attention: [attention_ch → channels]
    pub tdnn: Conv1dWeight,                  // TDNN: [channels*3 → attention_ch]
}

impl SpeakerEncoderWeights {
    pub fn memory_bytes(&self) -> usize {
        let mut total = self.initial_conv.weight.len() * 4;
        for block in &self.blocks {
            for conv in &block.res2net_blocks {
                total += conv.weight.len() * 4;
            }
            total += block.tdnn1.weight.len() * 4;
            total += block.tdnn2.weight.len() * 4;
            total += block.se_conv1.weight.len() * 4;
            total += block.se_conv2.weight.len() * 4;
        }
        total += self.asp.conv.weight.len() * 4;
        total += self.asp.tdnn.weight.len() * 4;
        total += self.mfa.weight.len() * 4;
        total += self.fc_weight.len() * 4;
        total += self.fc_bias.len() * 4;
        total
    }
}

// ============================================================
// Forward pass
// ============================================================

/// Extract speaker embedding from mel-spectrogram
/// Input: [128, T] mel-spec in channel-first format
/// Output: [enc_dim] embedding
pub fn extract_speaker_embedding(
    mel_spec: &[f32],
    weights: &SpeakerEncoderWeights,
    n_mels: usize,  // 128
) -> Vec<f32> {
    let t_len = mel_spec.len() / n_mels;
    eprintln!("  Mel-spec: [{n_mels}, {t_len}]");

    // 1. Initial conv (block 0): [128, T] → [512, T]
    let mut x = conv1d_dilated(mel_spec, &weights.initial_conv, n_mels);
    // Apply ReLU after initial conv
    for v in &mut x {
        *v = v.max(0.0);
    }
    let channels = weights.initial_conv.out_channels;
    let t1 = x.len() / channels;
    eprintln!("  Initial conv: [{channels}, {t1}]");

    // 2. SE-Res2Net blocks (blocks 1-3), collecting outputs for MFA
    let mut block_outputs: Vec<Vec<f32>> = Vec::new();
    for (i, block) in weights.blocks.iter().enumerate() {
        x = se_res2net_forward(&x, block);
        let t = x.len() / channels;
        eprintln!("  SE-Res2Net block {i}: [{channels}, {t}]");
        block_outputs.push(x.clone());
    }

    // 3. Multi-layer Feature Aggregation: concat all block outputs along channels
    // Each block output is [512, T], concatenated → [1536, T]
    let t_out = x.len() / channels;
    let total_ch = channels * block_outputs.len();  // 1536
    let mut mfa_input = vec![0.0f32; total_ch * t_out];
    for (bi, block_out) in block_outputs.iter().enumerate() {
        let ch_offset = bi * channels;
        for c in 0..channels {
            for t in 0..t_out {
                mfa_input[(ch_offset + c) * t_out + t] = block_out[c * t_out + t];
            }
        }
    }
    eprintln!("  MFA input: [{total_ch}, {t_out}]");

    // Apply MFA conv (1x1): [1536, T] → [1536, T] with ReLU (it's a TimeDelayNetBlock)
    let mfa_conv = conv1d_dilated(&mfa_input, &weights.mfa, total_ch);
    let mfa_ch = weights.mfa.out_channels;
    let mfa_out: Vec<f32> = mfa_conv.iter().map(|&v| v.max(0.0)).collect();
    let mfa_t = mfa_out.len() / mfa_ch;
    eprintln!("  MFA output: [{mfa_ch}, {mfa_t}]");

    // 4. ASP: [1536, T] → [3072]
    let asp_out = asp_forward(&mfa_out, &weights.asp, mfa_ch);
    eprintln!("  ASP output: [{}]", asp_out.len());

    // 5. FC projection: [3072] → [enc_dim]
    let embedding = fc_projection(&asp_out, &weights.fc_weight, &weights.fc_bias);

    // Debug: print embedding stats
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mean: f32 = embedding.iter().sum::<f32>() / embedding.len() as f32;
    let min = embedding.iter().copied().fold(f32::INFINITY, f32::min);
    let max = embedding.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    eprintln!("  Embedding: [{}], norm={:.4}, mean={:.6}, range=[{:.4}, {:.4}]", embedding.len(), norm, mean, min, max);

    embedding
}

/// SE-Res2Net block forward pass
/// Architecture: tdnn1 → Res2Net(cascading) → tdnn2 → SE → residual add
fn se_res2net_forward(input: &[f32], block: &Res2NetBlock) -> Vec<f32> {
    let channels = block.channels;  // 512
    let length = input.len() / channels;

    // TDNN1: [512, T] → [512, T]
    let tdnn1_out = conv1d_dilated(input, &block.tdnn1, channels);
    // ReLU
    let tdnn1_act: Vec<f32> = tdnn1_out.iter().map(|&v| v.max(0.0)).collect();
    let tdnn1_len = tdnn1_act.len() / channels;

    // Res2Net with cascading (scale = num_convs + 1)
    let scale = block.res2net_blocks.len() + 1;  // 7 convs + 1 passthrough = 8
    let branch_ch = channels / scale;

    let mut branches: Vec<Vec<f32>> = Vec::new();

    for i in 0..scale {
        // Extract chunk i from tdnn1 output: channels[i*branch_ch .. (i+1)*branch_ch]
        let start_ch = i * branch_ch;
        let mut chunk = vec![0.0f32; branch_ch * tdnn1_len];
        for c in 0..branch_ch {
            for t in 0..tdnn1_len {
                chunk[c * tdnn1_len + t] = tdnn1_act[(start_ch + c) * tdnn1_len + t];
            }
        }

        if i == 0 {
            // Branch 0: passthrough
            branches.push(chunk);
        } else if i == 1 {
            // Branch 1: conv(chunk)
            let out = conv1d_dilated(&chunk, &block.res2net_blocks[i - 1], branch_ch);
            // ReLU
            let activated: Vec<f32> = out.iter().map(|&v| v.max(0.0)).collect();
            branches.push(activated);
        } else {
            // Branch i (i>=2): conv(chunk + prev_branch_output) — cascading!
            let prev = &branches[i - 1];
            let prev_len = prev.len() / branch_ch;
            let mut combined = chunk.clone();
            for c in 0..branch_ch {
                for t in 0..tdnn1_len.min(prev_len) {
                    combined[c * tdnn1_len + t] += prev[c * prev_len + t];
                }
            }
            let out = conv1d_dilated(&combined, &block.res2net_blocks[i - 1], branch_ch);
            // ReLU
            let activated: Vec<f32> = out.iter().map(|&v| v.max(0.0)).collect();
            branches.push(activated);
        }
    }

    // Concatenate branches back to [channels, T]
    let out_t = branches[0].len() / branch_ch;
    let mut res2net_out = vec![0.0f32; channels * out_t];
    for (i, branch) in branches.iter().enumerate() {
        let start_ch = i * branch_ch;
        let br_len = branch.len() / branch_ch;
        for c in 0..branch_ch {
            for t in 0..out_t.min(br_len) {
                res2net_out[(start_ch + c) * out_t + t] = branch[c * br_len + t];
            }
        }
    }

    // TDNN2: [512, T] → [512, T]
    let tdnn2_out = conv1d_dilated(&res2net_out, &block.tdnn2, channels);
    // ReLU
    let tdnn2_act: Vec<f32> = tdnn2_out.iter().map(|&v| v.max(0.0)).collect();
    let tdnn2_len = tdnn2_act.len() / channels;

    // SE (Squeeze-Excitation) attention
    let se_out = se_attention(&tdnn2_act, &block.se_conv1, &block.se_conv2, channels);

    // Residual connection: se_out + input
    let mut output = vec![0.0f32; channels * tdnn2_len];
    for c in 0..channels {
        for t in 0..tdnn2_len.min(length) {
            output[c * tdnn2_len + t] = se_out[c * tdnn2_len + t] + input[c * length + t];
        }
    }

    output
}

/// Squeeze-Excitation attention
fn se_attention(
    input: &[f32],
    conv1: &Conv1dWeight,
    conv2: &Conv1dWeight,
    channels: usize,
) -> Vec<f32> {
    let length = input.len() / channels;

    // Global average pooling: [channels, T] → [channels]
    let mut pooled = vec![0.0f32; channels];
    for c in 0..channels {
        let mut sum = 0.0f32;
        for t in 0..length {
            sum += input[c * length + t];
        }
        pooled[c] = sum / length as f32;
    }

    // Squeeze: conv1 [channels → se_channels]
    let squeezed = conv1d_1x1_vec(&pooled, conv1);

    // ReLU
    let activated: Vec<f32> = squeezed.iter().map(|&x| x.max(0.0)).collect();

    // Excite: conv2 [se_channels → channels]
    let excited = conv1d_1x1_vec(&activated, conv2);

    // Sigmoid
    let weights: Vec<f32> = excited.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();

    // Apply channel-wise attention
    let mut output = vec![0.0f32; input.len()];
    for c in 0..channels {
        let w = weights[c];
        for t in 0..length {
            output[c * length + t] = input[c * length + t] * w;
        }
    }

    output
}

/// Attentive Statistics Pooling
/// Input: [channels, T] → Output: [channels * 2] (mean + std)
fn asp_forward(input: &[f32], weights: &ASPWeights, channels: usize) -> Vec<f32> {
    let length = input.len() / channels;

    // Compute global mean and std for attention input
    let mut global_mean = vec![0.0f32; channels];
    let mut global_std = vec![0.0f32; channels];
    for c in 0..channels {
        let mut sum = 0.0f32;
        let mut sq_sum = 0.0f32;
        for t in 0..length {
            let v = input[c * length + t];
            sum += v;
            sq_sum += v * v;
        }
        global_mean[c] = sum / length as f32;
        let variance = sq_sum / length as f32 - global_mean[c] * global_mean[c];
        global_std[c] = variance.max(0.0).sqrt();
    }

    // Concatenate [input, mean_expanded, std_expanded] → [channels*3, T]
    let cat_ch = channels * 3;
    let mut cat_input = vec![0.0f32; cat_ch * length];
    for c in 0..channels {
        for t in 0..length {
            cat_input[c * length + t] = input[c * length + t];
            cat_input[(channels + c) * length + t] = global_mean[c];
            cat_input[(2 * channels + c) * length + t] = global_std[c];
        }
    }

    // TDNN: [channels*3, T] → [attention_ch, T] (TimeDelayNetBlock = Conv + ReLU)
    let attn_raw = conv1d_dilated(&cat_input, &weights.tdnn, cat_ch);
    let attn_ch = weights.tdnn.out_channels;
    let _attn_len = attn_raw.len() / attn_ch;

    // ReLU (from TimeDelayNetBlock) then Tanh
    let attn_tanh: Vec<f32> = attn_raw.iter().map(|&x| x.max(0.0).tanh()).collect();

    // Conv: [attention_ch, T] → [channels, T]
    let attn_logits = conv1d_dilated(&attn_tanh, &weights.conv, attn_ch);
    let attn_out_ch = weights.conv.out_channels;
    let attn_out_len = attn_logits.len() / attn_out_ch;

    // Softmax over time for each channel
    let mut attention = vec![0.0f32; attn_logits.len()];
    for c in 0..attn_out_ch {
        let mut max_val = f32::NEG_INFINITY;
        for t in 0..attn_out_len {
            max_val = max_val.max(attn_logits[c * attn_out_len + t]);
        }
        let mut sum = 0.0f32;
        for t in 0..attn_out_len {
            let v = (attn_logits[c * attn_out_len + t] - max_val).exp();
            attention[c * attn_out_len + t] = v;
            sum += v;
        }
        if sum > 0.0 {
            for t in 0..attn_out_len {
                attention[c * attn_out_len + t] /= sum;
            }
        }
    }

    // Weighted mean and std
    let use_len = length.min(attn_out_len);
    let mut mean = vec![0.0f32; attn_out_ch];
    let mut std = vec![0.0f32; attn_out_ch];

    for c in 0..attn_out_ch {
        let mut weighted_sum = 0.0f32;
        let mut weighted_sq_sum = 0.0f32;

        for t in 0..use_len {
            let val = input[c * length + t];
            let w = attention[c * attn_out_len + t];
            weighted_sum += w * val;
            weighted_sq_sum += w * val * val;
        }

        mean[c] = weighted_sum;
        std[c] = (weighted_sq_sum - weighted_sum * weighted_sum).max(0.0).sqrt();
    }

    // Output: [mean, std] → [channels * 2]
    let mut output = mean;
    output.extend_from_slice(&std);
    output
}

/// Conv1d with dilation support, same-padding, and reflect padding mode
/// Matches Python nn.Conv1d(padding="same", padding_mode="reflect")
fn conv1d_dilated(input: &[f32], conv: &Conv1dWeight, in_channels: usize) -> Vec<f32> {
    let in_len = input.len() / in_channels;
    let kernel_size = conv.kernel_size;
    let stride = conv.stride.max(1);
    let dilation = conv.dilation.max(1);
    let out_channels = conv.out_channels;

    // Effective kernel size with dilation
    let eff_kernel = dilation * (kernel_size - 1) + 1;

    // Calculate padding for "same" mode
    let pad_total = ((in_len - 1) * stride + eff_kernel).saturating_sub(in_len);
    let pad_left = pad_total / 2;
    let pad_right = pad_total - pad_left;

    // Reflect-pad input (matching Python padding_mode="reflect")
    let padded_len = in_len + pad_left + pad_right;
    let mut padded = vec![0.0f32; in_channels * padded_len];

    for c in 0..in_channels {
        let c_in = c * in_len;
        let c_pad = c * padded_len;

        // Left reflect padding
        for t in 0..pad_left {
            let src = pad_left - t;  // reflect: 1, 2, 3, ...
            let src = if src < in_len { src } else { src % in_len.max(1) };
            padded[c_pad + t] = input[c_in + src];
        }
        // Center: copy original
        for t in 0..in_len {
            padded[c_pad + pad_left + t] = input[c_in + t];
        }
        // Right reflect padding
        for t in 0..pad_right {
            let src = in_len.saturating_sub(2 + t);  // reflect: len-2, len-3, ...
            padded[c_pad + pad_left + in_len + t] = input[c_in + src];
        }
    }

    // Apply convolution with dilation
    let out_len = (padded_len - eff_kernel) / stride + 1;
    let mut output = vec![0.0f32; out_channels * out_len];

    for oc in 0..out_channels {
        for t_out in 0..out_len {
            let t_in = t_out * stride;
            let mut sum = conv.bias[oc];

            for ic in 0..in_channels {
                for k in 0..kernel_size {
                    let idx_in = ic * padded_len + t_in + k * dilation;
                    let idx_w = (oc * in_channels + ic) * kernel_size + k;
                    sum += padded[idx_in] * conv.weight[idx_w];
                }
            }

            output[oc * out_len + t_out] = sum;
        }
    }

    output
}

/// 1x1 convolution on vector (linear projection)
fn conv1d_1x1_vec(input: &[f32], conv: &Conv1dWeight) -> Vec<f32> {
    let in_ch = conv.in_channels;
    let out_ch = conv.out_channels;

    let mut output = conv.bias.clone();

    for oc in 0..out_ch {
        for ic in 0..in_ch {
            let w_idx = oc * in_ch + ic;
            if w_idx < conv.weight.len() && ic < input.len() {
                output[oc] += input[ic] * conv.weight[w_idx];
            }
        }
    }

    output
}

/// Final FC projection
fn fc_projection(input: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let in_dim = input.len();
    let out_dim = bias.len();

    let mut output = bias.to_vec();

    for out_idx in 0..out_dim {
        for in_idx in 0..in_dim {
            let w_idx = out_idx * in_dim + in_idx;
            if w_idx < weight.len() {
                output[out_idx] += input[in_idx] * weight[w_idx];
            }
        }
    }

    output
}
