//! Convolution primitives for the speech decoder.
//!
//! All operations work on f32 buffers in channel-first format: [channels, length].
//! Heavy operations are multi-threaded for CPU parallelism.

use std::thread;

fn num_threads() -> usize {
    thread::available_parallelism().map(|n| n.get()).unwrap_or(6)
}

/// Wrapper to send raw pointers across threads (we guarantee non-overlapping writes).
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    #[inline]
    unsafe fn add(self, count: usize) -> *mut f32 {
        self.0.add(count)
    }
}

// ============================================================
// Conv1d
// ============================================================

pub struct Conv1dWeight {
    pub weight: Vec<f32>,  // [out_channels, in_channels, kernel_size]
    pub bias: Vec<f32>,    // [out_channels]
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

/// Standard 1D convolution — multi-threaded by output channels.
/// Input: [in_channels, in_length], Output: [out_channels, out_length]
pub fn conv1d(input: &[f32], w: &Conv1dWeight) -> Vec<f32> {
    let in_len = input.len() / w.in_channels;
    let out_len = (in_len + 2 * w.padding - w.dilation * (w.kernel_size - 1) - 1) / w.stride + 1;
    let total = w.out_channels * out_len;
    let mut output = vec![0.0f32; total];

    let ops = w.out_channels * w.in_channels * w.kernel_size * out_len;
    if ops < 500_000 {
        // Small: single-threaded
        conv1d_range(input, w, 0, w.out_channels, in_len, out_len, &mut output);
        return output;
    }

    let n_threads = num_threads().min(w.out_channels);
    let chunk = (w.out_channels + n_threads - 1) / n_threads;
    let out_ptr = SendPtr(output.as_mut_ptr());

    thread::scope(|s| {
        for tid in 0..n_threads {
            let oc_start = tid * chunk;
            let oc_end = (oc_start + chunk).min(w.out_channels);
            if oc_start >= oc_end { break; }
            let ptr = out_ptr;
            s.spawn(move || {
                let out_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        ptr.add(oc_start * out_len),
                        (oc_end - oc_start) * out_len,
                    )
                };
                conv1d_range(input, w, oc_start, oc_end, in_len, out_len, out_slice);
            });
        }
    });
    output
}

#[inline]
fn conv1d_range(
    input: &[f32], w: &Conv1dWeight,
    oc_start: usize, oc_end: usize,
    in_len: usize, out_len: usize,
    output: &mut [f32],
) {
    let ick = w.in_channels * w.kernel_size;
    for oc in oc_start..oc_end {
        let local_oc = oc - oc_start;
        let w_base = oc * ick;
        for o in 0..out_len {
            let mut sum = w.bias[oc];
            for ic in 0..w.in_channels {
                let in_row = ic * in_len;
                let w_row = w_base + ic * w.kernel_size;
                for k in 0..w.kernel_size {
                    let in_pos = o * w.stride + k * w.dilation;
                    if in_pos >= w.padding && in_pos < in_len + w.padding {
                        let idx = in_pos - w.padding;
                        sum += input[in_row + idx] * w.weight[w_row + k];
                    }
                }
            }
            output[local_oc * out_len + o] = sum;
        }
    }
}

// ============================================================
// CausalConv1d
// ============================================================

/// Causal Conv1d — left-padded, multi-threaded by output channels.
/// Padding = (kernel_size - 1) * dilation.
pub fn causal_conv1d(input: &[f32], w: &Conv1dWeight) -> Vec<f32> {
    let in_len = input.len() / w.in_channels;
    let out_len = in_len; // causal: same length

    let mut output = vec![0.0f32; w.out_channels * out_len];

    let ops = w.out_channels * w.in_channels * w.kernel_size * out_len;
    if ops < 500_000 {
        causal_conv1d_range(input, w, 0, w.out_channels, in_len, out_len, &mut output);
        return output;
    }

    let n_threads = num_threads().min(w.out_channels);
    let chunk = (w.out_channels + n_threads - 1) / n_threads;
    let out_ptr = SendPtr(output.as_mut_ptr());

    thread::scope(|s| {
        for tid in 0..n_threads {
            let oc_start = tid * chunk;
            let oc_end = (oc_start + chunk).min(w.out_channels);
            if oc_start >= oc_end { break; }
            let ptr = out_ptr;
            s.spawn(move || {
                let out_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        ptr.add(oc_start * out_len),
                        (oc_end - oc_start) * out_len,
                    )
                };
                causal_conv1d_range(input, w, oc_start, oc_end, in_len, out_len, out_slice);
            });
        }
    });
    output
}

#[inline]
fn causal_conv1d_range(
    input: &[f32], w: &Conv1dWeight,
    oc_start: usize, oc_end: usize,
    in_len: usize, out_len: usize,
    output: &mut [f32],
) {
    let causal_pad = (w.kernel_size - 1) * w.dilation;
    let ick = w.in_channels * w.kernel_size;
    for oc in oc_start..oc_end {
        let local_oc = oc - oc_start;
        let w_base = oc * ick;
        for o in 0..out_len {
            let mut sum = w.bias[oc];
            for ic in 0..w.in_channels {
                let in_row = ic * in_len;
                let w_row = w_base + ic * w.kernel_size;
                for k in 0..w.kernel_size {
                    let in_pos = o + k * w.dilation;
                    if in_pos >= causal_pad {
                        let idx = in_pos - causal_pad;
                        if idx < in_len {
                            sum += input[in_row + idx] * w.weight[w_row + k];
                        }
                    }
                }
            }
            output[local_oc * out_len + o] = sum;
        }
    }
}

// ============================================================
// ConvTranspose1d
// ============================================================

pub struct ConvTranspose1dWeight {
    pub weight: Vec<f32>,  // [in_channels, out_channels, kernel_size]
    pub bias: Vec<f32>,    // [out_channels]
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

/// Transposed 1D convolution — multi-threaded by output channels.
/// Input: [in_channels, in_length], Output: [out_channels, out_length]
pub fn conv_transpose1d(input: &[f32], w: &ConvTranspose1dWeight) -> Vec<f32> {
    let in_len = input.len() / w.in_channels;
    let out_len = (in_len - 1) * w.stride - 2 * w.padding + w.kernel_size;
    let mut output = vec![0.0f32; w.out_channels * out_len];

    let ops = w.in_channels * w.out_channels * w.kernel_size * in_len;
    if ops < 500_000 {
        conv_transpose1d_range(input, w, 0, w.out_channels, in_len, out_len, &mut output);
        return output;
    }

    let n_threads = num_threads().min(w.out_channels);
    let chunk = (w.out_channels + n_threads - 1) / n_threads;
    let out_ptr = SendPtr(output.as_mut_ptr());

    thread::scope(|s| {
        for tid in 0..n_threads {
            let oc_start = tid * chunk;
            let oc_end = (oc_start + chunk).min(w.out_channels);
            if oc_start >= oc_end { break; }
            let ptr = out_ptr;
            s.spawn(move || {
                let out_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        ptr.add(oc_start * out_len),
                        (oc_end - oc_start) * out_len,
                    )
                };
                conv_transpose1d_range(input, w, oc_start, oc_end, in_len, out_len, out_slice);
            });
        }
    });
    output
}

/// Per output channel: gather contributions from all input channels.
#[inline]
fn conv_transpose1d_range(
    input: &[f32], w: &ConvTranspose1dWeight,
    oc_start: usize, oc_end: usize,
    in_len: usize, out_len: usize,
    output: &mut [f32],
) {
    for oc in oc_start..oc_end {
        let local_oc = oc - oc_start;
        let out_row = local_oc * out_len;
        // Initialize with bias
        for o in 0..out_len {
            output[out_row + o] = w.bias[oc];
        }
        // Gather from all input channels
        for ic in 0..w.in_channels {
            let in_row = ic * in_len;
            let w_base = ic * w.out_channels * w.kernel_size + oc * w.kernel_size;
            for i in 0..in_len {
                let val = input[in_row + i];
                if val == 0.0 { continue; }
                for k in 0..w.kernel_size {
                    let o_pos_raw = i as isize * w.stride as isize + k as isize - w.padding as isize;
                    if o_pos_raw >= 0 && (o_pos_raw as usize) < out_len {
                        output[out_row + o_pos_raw as usize] += val * w.weight[w_base + k];
                    }
                }
            }
        }
    }
}

// ============================================================
// Depthwise Conv1d (for ConvNeXt)
// ============================================================

pub struct DepthwiseConv1dWeight {
    pub weight: Vec<f32>,  // [channels, 1, kernel_size]
    pub bias: Vec<f32>,    // [channels]
    pub channels: usize,
    pub kernel_size: usize,
    pub padding: usize,
}

/// Depthwise 1D convolution — multi-threaded by channels.
/// Uses causal (left-only) padding: pad = kernel_size - 1.
pub fn depthwise_conv1d(input: &[f32], w: &DepthwiseConv1dWeight) -> Vec<f32> {
    let in_len = input.len() / w.channels;
    let causal_pad = w.kernel_size - 1;
    let out_len = in_len;
    let mut output = vec![0.0f32; w.channels * out_len];

    let ops = w.channels * w.kernel_size * out_len;
    if ops < 500_000 {
        dw_conv1d_range(input, w, 0, w.channels, in_len, out_len, causal_pad, &mut output);
        return output;
    }

    let n_threads = num_threads().min(w.channels);
    let chunk = (w.channels + n_threads - 1) / n_threads;
    let out_ptr = SendPtr(output.as_mut_ptr());

    thread::scope(|s| {
        for tid in 0..n_threads {
            let c_start = tid * chunk;
            let c_end = (c_start + chunk).min(w.channels);
            if c_start >= c_end { break; }
            let ptr = out_ptr;
            s.spawn(move || {
                let out_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        ptr.add(c_start * out_len),
                        (c_end - c_start) * out_len,
                    )
                };
                dw_conv1d_range(input, w, c_start, c_end, in_len, out_len, causal_pad, out_slice);
            });
        }
    });
    output
}

#[inline]
fn dw_conv1d_range(
    input: &[f32], w: &DepthwiseConv1dWeight,
    c_start: usize, c_end: usize,
    in_len: usize, out_len: usize, causal_pad: usize,
    output: &mut [f32],
) {
    for c in c_start..c_end {
        let local_c = c - c_start;
        for o in 0..out_len {
            let mut sum = w.bias[c];
            for k in 0..w.kernel_size {
                let in_pos = o + k;
                if in_pos >= causal_pad {
                    let idx = in_pos - causal_pad;
                    if idx < in_len {
                        sum += input[c * in_len + idx] * w.weight[c * w.kernel_size + k];
                    }
                }
            }
            output[local_c * out_len + o] = sum;
        }
    }
}

// ============================================================
// SnakeBeta activation
// ============================================================

/// SnakeBeta: x + (1/exp(beta)) * sin^2(exp(alpha) * x)
/// Multi-threaded by channels.
pub fn snake_beta(input: &[f32], alpha: &[f32], beta: &[f32], channels: usize) -> Vec<f32> {
    let length = input.len() / channels;
    let mut output = vec![0.0f32; input.len()];

    if channels * length < 500_000 {
        snake_beta_range(input, alpha, beta, 0, channels, length, &mut output);
        return output;
    }

    let n_threads = num_threads().min(channels);
    let chunk = (channels + n_threads - 1) / n_threads;
    let out_ptr = SendPtr(output.as_mut_ptr());

    thread::scope(|s| {
        for tid in 0..n_threads {
            let c_start = tid * chunk;
            let c_end = (c_start + chunk).min(channels);
            if c_start >= c_end { break; }
            let ptr = out_ptr;
            s.spawn(move || {
                let out_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        ptr.add(c_start * length),
                        (c_end - c_start) * length,
                    )
                };
                snake_beta_range(input, alpha, beta, c_start, c_end, length, out_slice);
            });
        }
    });
    output
}

#[inline]
fn snake_beta_range(
    input: &[f32], alpha: &[f32], beta: &[f32],
    c_start: usize, c_end: usize, length: usize,
    output: &mut [f32],
) {
    for c in c_start..c_end {
        let local_c = c - c_start;
        let a = alpha[c].exp();
        let inv_b = (-beta[c]).exp();
        for t in 0..length {
            let x = input[c * length + t];
            let sin_val = (a * x).sin();
            output[local_c * length + t] = x + inv_b * sin_val * sin_val;
        }
    }
}

// ============================================================
// GroupNorm (used in ConvNeXt)
// ============================================================

/// Group normalization.
/// Input/output: [channels, length], with channels split into num_groups.
pub fn group_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    length: usize,
    num_groups: usize,
    eps: f32,
) -> Vec<f32> {
    let channels_per_group = channels / num_groups;
    let mut output = vec![0.0f32; channels * length];

    for g in 0..num_groups {
        let c_start = g * channels_per_group;
        let c_end = c_start + channels_per_group;
        let group_size = channels_per_group * length;

        let mut sum = 0.0f32;
        for c in c_start..c_end {
            for t in 0..length {
                sum += input[c * length + t];
            }
        }
        let mean = sum / group_size as f32;

        let mut var_sum = 0.0f32;
        for c in c_start..c_end {
            for t in 0..length {
                let diff = input[c * length + t] - mean;
                var_sum += diff * diff;
            }
        }
        let inv_std = 1.0 / (var_sum / group_size as f32 + eps).sqrt();

        for c in c_start..c_end {
            for t in 0..length {
                output[c * length + t] = (input[c * length + t] - mean) * inv_std * gamma[c] + beta[c];
            }
        }
    }
    output
}

// ============================================================
// GELU activation (used in ConvNeXt)
// ============================================================

/// GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
#[inline]
pub fn gelu(x: f32) -> f32 {
    x * 0.5 * (1.0 + (x * 0.7071067811865476).tanh())
}
