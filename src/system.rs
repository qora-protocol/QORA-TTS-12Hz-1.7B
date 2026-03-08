//! System awareness: detect RAM, CPU threads, and smart parameter adjustment for TTS.

/// System information detected at startup.
pub struct SystemInfo {
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub cpu_threads: usize,
}

/// Smart parameter limits for TTS based on system capabilities.
pub struct SmartLimits {
    pub max_codes: usize,
    pub default_max_codes: usize,
    pub warning: Option<&'static str>,
}

impl SystemInfo {
    /// Detect system info. Uses wmic on Windows, /proc/meminfo on Linux.
    pub fn detect() -> Self {
        let cpu_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let (total, available) = detect_ram();

        SystemInfo {
            total_ram_mb: total,
            available_ram_mb: available,
            cpu_threads,
        }
    }

    /// Compute smart limits for TTS based on system capabilities.
    /// Note: TTS memory is fixed once the model loads (no growing KV cache).
    /// These limits cap generation TIME, not memory — slower systems get shorter limits
    /// to avoid excessively long runs (e.g. 2000 codes at 2.5s/code = 83 minutes).
    pub fn smart_limits(&self) -> SmartLimits {
        let threads = self.cpu_threads;
        let avail = self.available_ram_mb;

        // Check if system can even load the model (~1GB for 0.6B, ~1.6GB for 1.7B)
        if avail < 1500 {
            return SmartLimits {
                max_codes: 100,
                default_max_codes: 50,
                warning: Some("Very low RAM — model may not load. Limiting to 100 codes (~4s audio)"),
            };
        }

        // Time-based limits: fewer cores = slower generation = tighter cap
        if threads <= 4 {
            SmartLimits {
                max_codes: 300,
                default_max_codes: 150,
                warning: Some("Few CPU threads — limiting to 300 codes (~12s audio) to keep generation time reasonable"),
            }
        } else if threads <= 8 {
            SmartLimits {
                max_codes: 1000,
                default_max_codes: 500,
                warning: None,
            }
        } else {
            SmartLimits {
                max_codes: 2000,
                default_max_codes: 500,
                warning: None,
            }
        }
    }
}

/// Detect total and available RAM in MB.
fn detect_ram() -> (u64, u64) {
    #[cfg(target_os = "windows")]
    {
        if let Some((total, avail)) = detect_ram_windows() {
            return (total, avail);
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Some((total, avail)) = detect_ram_linux() {
            return (total, avail);
        }
    }

    // Fallback: assume 8 GB total, 4 GB available
    (8192, 4096)
}

#[cfg(target_os = "windows")]
fn detect_ram_windows() -> Option<(u64, u64)> {
    let output = std::process::Command::new("wmic")
        .args(["OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory", "/format:csv"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 3 {
            let free_kb: u64 = parts[1].trim().parse().ok().unwrap_or(0);
            let total_kb: u64 = parts[2].trim().parse().ok().unwrap_or(0);
            if total_kb > 0 {
                return Some((total_kb / 1024, free_kb / 1024));
            }
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn detect_ram_linux() -> Option<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb = 0u64;
    let mut avail_kb = 0u64;
    for line in text.lines() {
        if line.starts_with("MemTotal:") {
            total_kb = parse_meminfo_kb(line);
        } else if line.starts_with("MemAvailable:") {
            avail_kb = parse_meminfo_kb(line);
        }
    }
    if total_kb > 0 {
        Some((total_kb / 1024, avail_kb / 1024))
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kb(line: &str) -> u64 {
    line.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}
