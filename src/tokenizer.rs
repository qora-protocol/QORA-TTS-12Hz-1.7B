//! Text tokenizer for QORA-TTS.

use std::path::Path;

pub struct TTSTokenizer {
    inner: tokenizers::Tokenizer,
}

impl TTSTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("Failed to load tokenizer: {e}"))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let encoding = self.inner.encode(text, false).expect("Failed to encode");
        encoding.get_ids().to_vec()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, true).expect("Failed to decode")
    }

    /// Build the TTS prompt for the talker model.
    /// Format:
    /// [IM_START, ASSISTANT, NEWLINE] (text) +
    /// [think, think_bos, lang, think_eos, speaker, pad, bos] (codec) +
    /// text tokens (text)
    pub fn format_tts_prompt(
        &self,
        text: &str,
        speaker_id: u32,
        language_id: u32,
    ) -> Vec<u32> {
        // Role prefix (text tokens): <|im_start|>assistant\n
        let mut full = Vec::new();
        full.push(151644); // IM_START
        full.push(77091);  // ASSISTANT
        full.push(198);    // NEWLINE

        // Codec prefix (7 tokens)
        full.push(2154); // CODEC_THINK
        full.push(2156); // CODEC_THINK_BOS
        full.push(language_id); // language
        full.push(2157); // CODEC_THINK_EOS
        full.push(speaker_id); // speaker
        full.push(2148); // CODEC_PAD
        full.push(2149); // CODEC_BOS

        // Text content (without role prefix, just the text itself)
        let text_tokens = self.encode(text);
        full.extend(text_tokens);

        full
    }

    /// Build the text-only portion of the TTS prompt (language_id + text tokens).
    /// Used when voice codes replace the speaker_id token.
    pub fn format_tts_prompt_text_only(
        &self,
        text: &str,
        language_id: u32,
    ) -> Vec<u32> {
        let prompt_text = format!(
            "<|im_start|>assistant\n{text}<|im_end|>"
        );
        let tokens = self.encode(&prompt_text);

        let mut full = Vec::with_capacity(tokens.len() + 1);
        full.push(language_id);
        full.extend(tokens);

        full
    }
}
