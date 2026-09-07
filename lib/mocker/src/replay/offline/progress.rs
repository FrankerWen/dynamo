// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use indicatif::{ProgressBar, ProgressStyle};

const INNER_PROGRESS_ENV: &str = "DYN_MOCKER_REPLAY_INNER_PROGRESS";

pub(super) struct ReplayProgress {
    bar: Option<ProgressBar>,
}

impl ReplayProgress {
    pub(super) fn new(total_requests: usize, label: &'static str) -> Self {
        Self::new_with_enabled(total_requests, label, inner_progress_enabled())
    }

    fn new_with_enabled(total_requests: usize, label: &'static str, enabled: bool) -> Self {
        if !enabled {
            return Self { bar: None };
        }
        let bar = ProgressBar::new(total_requests as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta}) {msg}",
            )
            .expect("progress bar template must be valid")
            .progress_chars("#>-"),
        );
        bar.set_message(label);
        Self { bar: Some(bar) }
    }

    pub(super) fn inc_completed(&self) {
        if let Some(bar) = &self.bar {
            bar.inc(1);
        }
    }

    pub(super) fn finish(&self) {
        if let Some(bar) = &self.bar {
            bar.finish_and_clear();
        }
    }
}

impl Drop for ReplayProgress {
    fn drop(&mut self) {
        if let Some(bar) = &self.bar {
            if !bar.is_finished() {
                bar.finish_and_clear();
            }
        }
    }
}

fn inner_progress_enabled() -> bool {
    std::env::var(INNER_PROGRESS_ENV)
        .ok()
        .is_some_and(|value| progress_value_enabled(&value))
}

fn progress_value_enabled(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inner_progress_is_opt_in() {
        for value in ["1", "TRUE", " yes ", "On"] {
            assert!(progress_value_enabled(value), "value={value:?}");
        }
        for value in ["", "0", "false", "off", "unexpected"] {
            assert!(!progress_value_enabled(value), "value={value:?}");
        }
    }

    #[test]
    fn disabled_progress_does_not_construct_a_tty_writer() {
        let progress = ReplayProgress::new_with_enabled(10, "test", false);
        assert!(progress.bar.is_none());
        progress.inc_completed();
        progress.finish();
    }
}
