use anyhow::Result;
use chrono::{DateTime, Local};
use clap::ValueEnum;
use colored::Colorize;
use comfy_table::{presets::UTF8_FULL, Attribute, Cell, Color, ContentArrangement, Table};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use std::fmt::Display;
use std::io::{self, Write};
use std::time::Duration;

/// Output format options
#[derive(Debug, Clone, Copy, PartialEq, ValueEnum)]
pub enum OutputFormat {
    /// Automatically detect format based on terminal
    Auto,
    /// Human-readable output with colors and tables
    Pretty,
    /// JSON output for scripting
    Json,
    /// Plain text output
    Plain,
    /// CSV output
    Csv,
}

/// Output handler for consistent formatting
pub struct OutputHandler {
    format: OutputFormat,
    is_terminal: bool,
}

impl OutputHandler {
    pub fn new(format: OutputFormat) -> Self {
        let is_terminal = atty::is(atty::Stream::Stdout);

        Self {
            format: match format {
                OutputFormat::Auto => {
                    if is_terminal {
                        OutputFormat::Pretty
                    } else {
                        OutputFormat::Json
                    }
                }
                other => other,
            },
            is_terminal,
        }
    }

    /// Print a success message
    pub fn success(&self, message: &str) {
        match self.format {
            OutputFormat::Pretty => {
                println!("{} {}", "✓".green().bold(), message);
            }
            OutputFormat::Json => {
                let output = serde_json::json!({
                    "status": "success",
                    "message": message,
                });
                println!("{}", serde_json::to_string(&output).unwrap());
            }
            _ => println!("SUCCESS: {}", message),
        }
    }

    /// Print an error message
    pub fn error(&self, message: &str) {
        match self.format {
            OutputFormat::Pretty => {
                eprintln!("{} {}", "✗".red().bold(), message.red());
            }
            OutputFormat::Json => {
                let output = serde_json::json!({
                    "status": "error",
                    "message": message,
                });
                eprintln!("{}", serde_json::to_string(&output).unwrap());
            }
            _ => eprintln!("ERROR: {}", message),
        }
    }

    /// Print a warning message
    pub fn warning(&self, message: &str) {
        match self.format {
            OutputFormat::Pretty => {
                println!("{} {}", "⚠".yellow().bold(), message.yellow());
            }
            OutputFormat::Json => {
                let output = serde_json::json!({
                    "status": "warning",
                    "message": message,
                });
                println!("{}", serde_json::to_string(&output).unwrap());
            }
            _ => println!("WARNING: {}", message),
        }
    }

    /// Print an info message
    pub fn info(&self, message: &str) {
        match self.format {
            OutputFormat::Pretty => {
                println!("{} {}", "ℹ".blue().bold(), message);
            }
            OutputFormat::Json => {
                let output = serde_json::json!({
                    "status": "info",
                    "message": message,
                });
                println!("{}", serde_json::to_string(&output).unwrap());
            }
            _ => println!("INFO: {}", message),
        }
    }

    /// Print any serializable data
    pub fn print_data<T: Serialize>(&self, data: &T) -> Result<()> {
        match self.format {
            OutputFormat::Json => {
                let json = serde_json::to_string_pretty(data)?;
                println!("{}", json);
            }
            OutputFormat::Csv => {
                // For CSV, we need to handle this case-by-case in the calling code
                // as CSV structure depends on the data type
                self.warning("CSV output should be handled by specific commands");
            }
            _ => {
                // For pretty/plain, we also need case-by-case handling
                let json = serde_json::to_string_pretty(data)?;
                println!("{}", json);
            }
        }
        Ok(())
    }

    /// Create a progress bar
    pub fn progress_bar(&self, len: u64, message: &str) -> Option<ProgressBar> {
        if !self.is_terminal || self.format == OutputFormat::Json {
            return None;
        }

        let pb = ProgressBar::new(len);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message(message.to_string());
        Some(pb)
    }

    /// Create a spinner for indefinite progress
    pub fn spinner(&self, message: &str) -> Option<ProgressBar> {
        if !self.is_terminal || self.format == OutputFormat::Json {
            return None;
        }

        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        pb.set_message(message.to_string());
        pb.enable_steady_tick(Duration::from_millis(100));
        Some(pb)
    }

    /// Print a table
    pub fn print_table(&self, headers: Vec<&str>, rows: Vec<Vec<String>>) {
        match self.format {
            OutputFormat::Pretty => {
                let mut table = Table::new();
                table
                    .load_preset(UTF8_FULL)
                    .set_content_arrangement(ContentArrangement::Dynamic)
                    .set_header(
                        headers
                            .iter()
                            .map(|h| Cell::new(h).add_attribute(Attribute::Bold)),
                    );

                for row in rows {
                    table.add_row(row);
                }

                println!("{table}");
            }
            OutputFormat::Csv => {
                // Print CSV header
                println!("{}", headers.join(","));

                // Print CSV rows
                for row in rows {
                    let escaped: Vec<String> = row
                        .iter()
                        .map(|field| {
                            if field.contains(',') || field.contains('"') || field.contains('\n') {
                                format!("\"{}\"", field.replace('"', "\"\""))
                            } else {
                                field.clone()
                            }
                        })
                        .collect();
                    println!("{}", escaped.join(","));
                }
            }
            _ => {
                // Plain text format
                let max_widths: Vec<usize> = headers
                    .iter()
                    .enumerate()
                    .map(|(i, h)| {
                        let header_len = h.len();
                        let max_row_len = rows
                            .iter()
                            .map(|row| row.get(i).map(|s| s.len()).unwrap_or(0))
                            .max()
                            .unwrap_or(0);
                        header_len.max(max_row_len)
                    })
                    .collect();

                // Print headers
                let header_line: Vec<String> = headers
                    .iter()
                    .zip(&max_widths)
                    .map(|(h, w)| format!("{:width$}", h, width = w))
                    .collect();
                println!("{}", header_line.join(" | "));

                // Print separator
                let separator: Vec<String> = max_widths.iter().map(|w| "-".repeat(*w)).collect();
                println!("{}", separator.join("-+-"));

                // Print rows
                for row in rows {
                    let row_line: Vec<String> = row
                        .iter()
                        .zip(&max_widths)
                        .map(|(cell, w)| format!("{:width$}", cell, width = w))
                        .collect();
                    println!("{}", row_line.join(" | "));
                }
            }
        }
    }

    /// Print a status item with color coding
    pub fn print_status(&self, label: &str, value: &str, status: StatusLevel) {
        match self.format {
            OutputFormat::Pretty => {
                let colored_value = match status {
                    StatusLevel::Good => value.green(),
                    StatusLevel::Warning => value.yellow(),
                    StatusLevel::Error => value.red(),
                    StatusLevel::Info => value.blue(),
                    StatusLevel::Neutral => value.normal(),
                };
                println!("{}: {}", label.bold(), colored_value);
            }
            OutputFormat::Json => {
                let output = serde_json::json!({
                    "label": label,
                    "value": value,
                    "status": status.to_string(),
                });
                println!("{}", serde_json::to_string(&output).unwrap());
            }
            _ => {
                println!("{}: {}", label, value);
            }
        }
    }

    /// Print a timestamp
    pub fn print_timestamp(&self, label: &str, time: DateTime<Local>) {
        let formatted = time.format("%Y-%m-%d %H:%M:%S %Z").to_string();
        self.print_status(label, &formatted, StatusLevel::Info);
    }

    /// Ask for user confirmation
    pub fn confirm(&self, message: &str, default: bool) -> Result<bool> {
        if !self.is_terminal {
            return Ok(default);
        }

        match self.format {
            OutputFormat::Pretty => {
                print!("{} {} ", "?".yellow().bold(), message);
                if default {
                    print!("[Y/n] ");
                } else {
                    print!("[y/N] ");
                }
                io::stdout().flush()?;

                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                let input = input.trim().to_lowercase();

                Ok(match input.as_str() {
                    "" => default,
                    "y" | "yes" => true,
                    "n" | "no" => false,
                    _ => default,
                })
            }
            _ => Ok(default),
        }
    }

    /// Create a section header
    pub fn section(&self, title: &str) {
        match self.format {
            OutputFormat::Pretty => {
                println!("\n{}", title.bold().underline());
            }
            OutputFormat::Json => {
                // JSON doesn't need section headers
            }
            _ => {
                println!("\n=== {} ===", title);
            }
        }
    }

    /// Print a list of items
    pub fn list<T: Display>(&self, items: &[T], ordered: bool) {
        match self.format {
            OutputFormat::Pretty => {
                for (i, item) in items.iter().enumerate() {
                    if ordered {
                        println!("  {}. {}", i + 1, item);
                    } else {
                        println!("  {} {}", "•".blue(), item);
                    }
                }
            }
            OutputFormat::Json => {
                let items: Vec<String> = items.iter().map(|i| i.to_string()).collect();
                self.print_data(&items).unwrap();
            }
            _ => {
                for (i, item) in items.iter().enumerate() {
                    if ordered {
                        println!("{}. {}", i + 1, item);
                    } else {
                        println!("- {}", item);
                    }
                }
            }
        }
    }

    /// Print key-value pairs
    pub fn key_value(&self, pairs: &[(String, String)]) {
        match self.format {
            OutputFormat::Pretty => {
                for (key, value) in pairs {
                    println!("  {}: {}", key.bold(), value);
                }
            }
            OutputFormat::Json => {
                let map: std::collections::HashMap<_, _> =
                    pairs.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                self.print_data(&map).unwrap();
            }
            _ => {
                for (key, value) in pairs {
                    println!("{}: {}", key, value);
                }
            }
        }
    }
}

/// Status level for color coding
#[derive(Debug, Clone, Copy)]
pub enum StatusLevel {
    Good,
    Warning,
    Error,
    Info,
    Neutral,
}

impl Display for StatusLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StatusLevel::Good => write!(f, "good"),
            StatusLevel::Warning => write!(f, "warning"),
            StatusLevel::Error => write!(f, "error"),
            StatusLevel::Info => write!(f, "info"),
            StatusLevel::Neutral => write!(f, "neutral"),
        }
    }
}
