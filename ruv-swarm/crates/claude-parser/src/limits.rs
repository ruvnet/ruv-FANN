//! Budgets for buffered stream input and retained events.

use crate::{ParserError, Result};
use tokio::io::{AsyncBufRead, AsyncBufReadExt};

/// Limits apply across all streams parsed by one parser instance.
#[derive(Debug, Clone, Copy)]
pub struct ParserLimits {
    /// Maximum physical line length, including its newline if present.
    pub max_line_bytes: usize,
    /// Maximum input bytes, including invalid events and blank lines.
    pub max_input_bytes: usize,
    /// Maximum successfully parsed events retained for export.
    pub max_events: usize,
}

impl Default for ParserLimits {
    fn default() -> Self {
        Self {
            max_line_bytes: 1024 * 1024,
            max_input_bytes: 64 * 1024 * 1024,
            max_events: 100_000,
        }
    }
}

/// Read without allowing a delimiter-free line to grow the buffer unboundedly.
pub(crate) async fn read_bounded_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    line: &mut Vec<u8>,
    limits: ParserLimits,
    input_bytes: &mut usize,
) -> Result<usize> {
    line.clear();
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Ok(line.len());
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let amount = newline.map_or(available.len(), |index| index + 1);
        let length = line
            .len()
            .checked_add(amount)
            .ok_or(ParserError::LimitExceeded {
                resource: "line bytes",
                limit: limits.max_line_bytes,
            })?;
        if length > limits.max_line_bytes {
            return Err(ParserError::LimitExceeded {
                resource: "line bytes",
                limit: limits.max_line_bytes,
            });
        }
        if amount > limits.max_input_bytes.saturating_sub(*input_bytes) {
            return Err(ParserError::LimitExceeded {
                resource: "input bytes",
                limit: limits.max_input_bytes,
            });
        }
        line.extend_from_slice(&available[..amount]);
        reader.consume(amount);
        *input_bytes += amount;
        if newline.is_some() {
            return Ok(line.len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ClaudeStreamParser;

    #[tokio::test]
    async fn accepts_exact_record_limit_and_final_line_without_newline() {
        let record = br#"{"type":"message_stop"}"#;
        let mut parser = ClaudeStreamParser::with_limits(ParserLimits {
            max_line_bytes: record.len(),
            max_input_bytes: record.len(),
            max_events: 1,
        })
        .unwrap();
        parser.parse_stream(&record[..]).await.unwrap();
        assert_eq!(parser.export_training_data().events.len(), 1);
    }

    #[tokio::test]
    async fn rejects_oversized_unterminated_lines_before_storing_events() {
        let mut parser = ClaudeStreamParser::with_limits(ParserLimits {
            max_line_bytes: 16,
            max_input_bytes: 1024,
            max_events: 10,
        })
        .unwrap();
        assert!(matches!(
            parser.parse_stream(&[b'x'; 17][..]).await,
            Err(ParserError::LimitExceeded {
                resource: "line bytes",
                ..
            })
        ));
        assert!(parser.export_training_data().events.is_empty());
    }

    #[tokio::test]
    async fn counts_blank_and_invalid_bytes_across_streams() {
        let mut parser = ClaudeStreamParser::with_limits(ParserLimits {
            max_line_bytes: 16,
            max_input_bytes: 6,
            max_events: 10,
        })
        .unwrap();
        parser.parse_stream(&b"\nnope\n"[..]).await.unwrap();
        assert!(matches!(
            parser.parse_stream(&b"\n"[..]).await,
            Err(ParserError::LimitExceeded {
                resource: "input bytes",
                ..
            })
        ));
    }

    #[tokio::test]
    async fn enforces_event_budget_across_streams() {
        let record = br#"{"type":"message_stop"}"#;
        let mut parser = ClaudeStreamParser::with_limits(ParserLimits {
            max_line_bytes: 128,
            max_input_bytes: 1024,
            max_events: 1,
        })
        .unwrap();
        parser.parse_stream(&record[..]).await.unwrap();
        assert!(matches!(
            parser.parse_stream(&record[..]).await,
            Err(ParserError::LimitExceeded {
                resource: "events",
                limit: 1
            })
        ));
        assert_eq!(parser.export_training_data().events.len(), 1);
    }
}
