//! Parse QORA-TTS config.json.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct QoraTTSConfig {
    #[serde(default)]
    pub architectures: Vec<String>,

    pub talker_config: TalkerConfig,

    // Top-level special tokens
    #[serde(default = "default_tts_bos")]
    pub tts_bos_token_id: u32,
    #[serde(default = "default_tts_eos")]
    pub tts_eos_token_id: u32,
    #[serde(default = "default_tts_pad")]
    pub tts_pad_token_id: u32,

    #[serde(default = "default_im_start")]
    pub im_start_token_id: u32,
    #[serde(default = "default_im_end")]
    pub im_end_token_id: u32,
    #[serde(default = "default_assistant")]
    pub assistant_token_id: u32,
}

fn default_tts_bos() -> u32 { 151672 }
fn default_tts_eos() -> u32 { 151673 }
fn default_tts_pad() -> u32 { 151671 }
fn default_im_start() -> u32 { 151644 }
fn default_im_end() -> u32 { 151645 }
fn default_assistant() -> u32 { 77091 }

#[derive(Debug, Deserialize)]
pub struct TalkerConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default = "default_num_code_groups")]
    pub num_code_groups: usize,
    #[serde(default = "default_text_hidden_size")]
    pub text_hidden_size: usize,
    #[serde(default = "default_text_vocab_size")]
    pub text_vocab_size: usize,
    #[serde(default = "default_talker_vocab")]
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    // Code predictor nested config
    #[serde(default)]
    pub code_predictor_config: Option<CodePredictorConfig>,

    // Codec special tokens
    #[serde(default = "default_codec_bos")]
    pub codec_bos_id: u32,
    #[serde(default = "default_codec_eos")]
    pub codec_eos_token_id: u32,
    #[serde(default = "default_codec_pad")]
    pub codec_pad_id: u32,
    #[serde(default = "default_codec_think")]
    pub codec_think_id: u32,
    #[serde(default = "default_codec_nothink")]
    pub codec_nothink_id: u32,

    // Speaker/language maps (nested in talker_config)
    #[serde(default)]
    pub spk_id: HashMap<String, u32>,
    #[serde(default)]
    pub codec_language_id: HashMap<String, u32>,
}

#[derive(Debug, Deserialize)]
pub struct CodePredictorConfig {
    #[serde(default = "default_cp_hidden")]
    pub hidden_size: usize,
    #[serde(default = "default_cp_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_cp_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_cp_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_cp_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "default_cp_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_cp_vocab")]
    pub vocab_size: usize,
    #[serde(default = "default_num_code_groups")]
    pub num_code_groups: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default = "default_cp_max_pos")]
    pub max_position_embeddings: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RopeScaling {
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default = "default_true")]
    pub interleaved: bool,
    #[serde(default)]
    pub rope_type: Option<String>,
}

// Talker defaults
fn default_hidden_size() -> usize { 2048 }
fn default_num_hidden_layers() -> usize { 28 }
fn default_num_attention_heads() -> usize { 16 }
fn default_num_kv_heads() -> usize { 8 }
fn default_intermediate_size() -> usize { 6144 }
fn default_rope_theta() -> f64 { 1000000.0 }
fn default_num_code_groups() -> usize { 16 }
fn default_text_hidden_size() -> usize { 2048 }
fn default_text_vocab_size() -> usize { 151936 }
fn default_talker_vocab() -> usize { 3072 }
fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_head_dim() -> usize { 128 }
fn default_max_position_embeddings() -> usize { 32768 }
fn default_true() -> bool { true }

// Code predictor defaults
fn default_cp_hidden() -> usize { 1024 }
fn default_cp_layers() -> usize { 5 }
fn default_cp_heads() -> usize { 16 }
fn default_cp_kv_heads() -> usize { 8 }
fn default_cp_intermediate() -> usize { 3072 }
fn default_cp_head_dim() -> usize { 128 }
fn default_cp_vocab() -> usize { 2048 }
fn default_cp_max_pos() -> usize { 65536 }

// Codec special token defaults
fn default_codec_bos() -> u32 { 2149 }
fn default_codec_eos() -> u32 { 2150 }
fn default_codec_pad() -> u32 { 2148 }
fn default_codec_think() -> u32 { 2154 }
fn default_codec_nothink() -> u32 { 2155 }

impl TalkerConfig {
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn mrope_section(&self) -> &[usize] {
        if let Some(ref rs) = self.rope_scaling {
            &rs.mrope_section
        } else {
            &[24, 20, 20]
        }
    }

    pub fn is_interleaved(&self) -> bool {
        self.rope_scaling.as_ref().map(|rs| rs.interleaved).unwrap_or(true)
    }
}

impl CodePredictorConfig {
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

impl QoraTTSConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&data)?;
        Ok(config)
    }
}
