//! Multimodal RoPE for QORA-TTS.
//!
//! Key difference from standard RoPE:
//! - **Interleaved** pairing: (dim[2i], dim[2i+1]) instead of (dim[i], dim[i+half_dim])
//! - **3 sections** with independent position IDs: mrope_section=[24, 20, 20]
//!   Section 0: dims 0..48 (24 pairs), Section 1: dims 48..88 (20 pairs), Section 2: dims 88..128 (20 pairs)

/// Precompute RoPE cos/sin tables for multimodal RoPE.
/// Returns (cos_table, sin_table) each of shape [max_pos, half_dim].
/// half_dim = head_dim / 2 = sum(mrope_section) = 64 for head_dim=128.
pub fn build_mrope_tables(
    head_dim: usize,
    max_pos: usize,
    theta: f64,
    mrope_section: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    debug_assert_eq!(mrope_section.iter().sum::<usize>(), half_dim);

    let mut cos_table = vec![0.0f32; max_pos * half_dim];
    let mut sin_table = vec![0.0f32; max_pos * half_dim];

    // Each section uses interleaved dim indices
    // For section s covering dims [start..start+section_size]:
    //   freq_i = 1 / (theta ^ (2*i / head_dim)) for i = start..start+section_size
    let mut dim_offset = 0;
    for &section_size in mrope_section {
        for i in 0..section_size {
            let dim_idx = dim_offset + i;
            let freq = 1.0 / theta.powf((2 * dim_idx) as f64 / head_dim as f64);
            for pos in 0..max_pos {
                let angle = pos as f64 * freq;
                cos_table[pos * half_dim + dim_idx] = angle.cos() as f32;
                sin_table[pos * half_dim + dim_idx] = angle.sin() as f32;
            }
        }
        dim_offset += section_size;
    }

    (cos_table, sin_table)
}

/// Apply multimodal RoPE (interleaved) in-place on a flat buffer [num_heads * head_dim].
///
/// For interleaved RoPE, pairs are (dim[2i], dim[2i+1]):
///   x_new[2i]   = x[2i] * cos - x[2i+1] * sin
///   x_new[2i+1] = x[2i+1] * cos + x[2i] * sin
///
/// Each section uses its own position_id.
#[inline]
pub fn apply_mrope_interleaved(
    data: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    mrope_section: &[usize],
    position_ids: &[usize; 3],
    cos_table: &[f32],
    sin_table: &[f32],
) {
    let half_dim = head_dim / 2;
    debug_assert!(mrope_section.len() <= 3);

    // Build the cos/sin values for this token's position
    // Each section uses its own position_id
    let mut cos_vals = vec![0.0f32; half_dim];
    let mut sin_vals = vec![0.0f32; half_dim];

    let mut dim_offset = 0;
    for (s, &section_size) in mrope_section.iter().enumerate() {
        let pos = position_ids[s];
        let table_offset = pos * half_dim;
        for i in 0..section_size {
            cos_vals[dim_offset + i] = cos_table[table_offset + dim_offset + i];
            sin_vals[dim_offset + i] = sin_table[table_offset + dim_offset + i];
        }
        dim_offset += section_size;
    }

    // Apply SPLIT RoPE to each head (matching working code)
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half_dim {
            // SPLIT pattern: first half and second half
            let x1 = data[base + i];
            let x2 = data[base + half_dim + i];
            let c = cos_vals[i];
            let s = sin_vals[i];
            // Rotate: [x1*cos - x2*sin, x2*cos + x1*sin]
            data[base + i] = x1 * c - x2 * s;
            data[base + half_dim + i] = x2 * c + x1 * s;
        }
    }
}

/// Apply interleaved RoPE for a batch of tokens during prefill.
/// data: [seq_len, num_heads * head_dim]
/// For TTS, all 3 position sections typically use the same position (sequential).
pub fn apply_mrope_interleaved_batch(
    data: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    mrope_section: &[usize],
    start_pos: usize,
    cos_table: &[f32],
    sin_table: &[f32],
) {
    let stride = num_heads * head_dim;
    for t in 0..seq_len {
        let pos = start_pos + t;
        let position_ids = [pos, pos, pos];
        apply_mrope_interleaved(
            &mut data[t * stride..(t + 1) * stride],
            num_heads,
            head_dim,
            mrope_section,
            &position_ids,
            cos_table,
            sin_table,
        );
    }
}

/// Apply standard interleaved RoPE (for code predictor).
/// data: [num_heads * head_dim]
/// No multi-section, just standard RoPE.
pub fn apply_rope_interleaved(
    data: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    position: usize,
    cos_table: &[f32],
    sin_table: &[f32],
) {
    let half_dim = head_dim / 2;
    for h in 0..num_heads {
        let offset = h * head_dim;
        for d in 0..half_dim {
            let idx = position * half_dim + d;
            let cos = cos_table[idx];
            let sin = sin_table[idx];

            // SPLIT pattern: x1 = first half, x2 = second half
            let x1 = data[offset + d];
            let x2 = data[offset + half_dim + d];

            // Rotate: [x1*cos - x2*sin, x2*cos + x1*sin]
            data[offset + d] = x1 * cos - x2 * sin;
            data[offset + half_dim + d] = x2 * cos + x1 * sin;
        }
    }
}
