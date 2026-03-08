//! PCM 16-bit WAV writer/reader (mono).

use std::io::{self, Write};
use std::path::Path;

/// Write f32 audio samples [-1.0, 1.0] as a 16-bit PCM WAV file.
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);

    let num_samples = samples.len() as u32;
    let bytes_per_sample: u16 = 2;
    let num_channels: u16 = 1;
    let data_size = num_samples * bytes_per_sample as u32;
    let file_size = 36 + data_size;

    // RIFF header
    f.write_all(b"RIFF")?;
    f.write_all(&file_size.to_le_bytes())?;
    f.write_all(b"WAVE")?;

    // fmt chunk
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;         // chunk size
    f.write_all(&1u16.to_le_bytes())?;           // PCM format
    f.write_all(&num_channels.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    let byte_rate = sample_rate * num_channels as u32 * bytes_per_sample as u32;
    f.write_all(&byte_rate.to_le_bytes())?;
    let block_align = num_channels * bytes_per_sample;
    f.write_all(&block_align.to_le_bytes())?;
    f.write_all(&(bytes_per_sample * 8).to_le_bytes())?; // bits per sample

    // data chunk
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_val = (clamped * 32767.0) as i16;
        f.write_all(&i16_val.to_le_bytes())?;
    }

    f.flush()?;
    Ok(())
}

/// Read WAV file and return (samples, sample_rate)
/// Converts to mono f32 samples [-1.0, 1.0]
pub fn read_wav(path: &Path) -> io::Result<(Vec<f32>, u32)> {
    let all_data = std::fs::read(path)?;

    if all_data.len() < 44 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "WAV file too small"));
    }

    // Check RIFF header
    if &all_data[0..4] != b"RIFF" || &all_data[8..12] != b"WAVE" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a valid WAV file"));
    }

    // Find fmt chunk
    let mut pos = 12;
    let mut num_channels = 0usize;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut fmt_found = false;

    while pos + 8 <= all_data.len() {
        let chunk_id = &all_data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            all_data[pos + 4],
            all_data[pos + 5],
            all_data[pos + 6],
            all_data[pos + 7],
        ]) as usize;

        if chunk_id == b"fmt " {
            if pos + 8 + chunk_size > all_data.len() {
                break;
            }
            num_channels = u16::from_le_bytes([all_data[pos + 10], all_data[pos + 11]]) as usize;
            sample_rate = u32::from_le_bytes([
                all_data[pos + 12],
                all_data[pos + 13],
                all_data[pos + 14],
                all_data[pos + 15],
            ]);
            bits_per_sample = u16::from_le_bytes([all_data[pos + 22], all_data[pos + 23]]);
            fmt_found = true;
        } else if chunk_id == b"data" && fmt_found {
            // Found data chunk
            if bits_per_sample != 16 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Only 16-bit WAV supported"));
            }

            let data_size = chunk_size;
            let data_start = pos + 8;

            if data_start + data_size > all_data.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Data chunk size exceeds file"));
            }

            let data = &all_data[data_start..data_start + data_size];
            let num_samples = data_size / 2 / num_channels;

            // Convert to mono f32
            let mut samples = vec![0.0f32; num_samples];

            for i in 0..num_samples {
                let mut sum = 0.0f32;
                for ch in 0..num_channels {
                    let offset = (i * num_channels + ch) * 2;
                    if offset + 1 < data.len() {
                        let i16_val = i16::from_le_bytes([data[offset], data[offset + 1]]);
                        sum += i16_val as f32 / 32768.0;
                    }
                }
                samples[i] = sum / num_channels as f32;
            }

            return Ok((samples, sample_rate));
        }

        pos += 8 + chunk_size;
        // Chunks are word-aligned
        if chunk_size % 2 == 1 {
            pos += 1;
        }
    }

    Err(io::Error::new(io::ErrorKind::InvalidData, "Missing data chunk"))
}
