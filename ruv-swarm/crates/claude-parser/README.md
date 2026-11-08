# claude-parser

Parse line-delimited Claude-style JSON events and collect tool, timing, token and error summaries. Parsed events can be exported to JSON or JSONL for later analysis.

## Parse a saved stream

```rust
use claude_parser::ClaudeStreamParser;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut parser = ClaudeStreamParser::new();
    let input = tokio::fs::File::open("events.jsonl").await?;
    let metrics = parser.parse_stream(input).await?;
    println!("Reported tokens: {}", metrics.token_usage.total_tokens);
    parser.export_training_data().to_json_file("parsed-events.json").await?;
    Ok(())
}
```

The accepted schema includes message, content-block, tool-use, thinking, function-result, error and usage events. An optional `data: ` prefix is accepted. Unknown or malformed JSON events are logged without printing the raw event and skipped; invalid UTF-8 and exceeded budgets return errors. This schema is not a guarantee of compatibility with every Claude CLI release or SSE framing convention.

## Input budgets

The defaults are 1 MiB per physical line (including its newline), 64 MiB of total input and 100,000 retained valid events. Limits count across repeated `parse_stream` calls on the same parser; blank and malformed lines consume the input budget. Use `ClaudeStreamParser::with_limits(ParserLimits { ... })` to set explicit positive budgets. Create a new parser for a new independent session.

Token counts come from incoming events. Timing summaries describe parser observation time; they do not establish model quality or token savings. Exports may contain the full original event content, so choose output locations accordingly.

## Check it

```sh
cargo test --locked -p claude-parser --lib
```

Regression coverage includes unterminated oversized lines, exact limits, cumulative input budgets and event retention limits.

The package requires Rust 1.88 and participates in the workspace all-target/all-feature acceptance gate.

The versions in this checkout are **unpublished release candidates**. See the [workspace guide](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm) for all 14 packages and the [review record](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-rust-swarm.md) for checks, dependency findings and remaining limits.

![Animated map of the ruv-FANN packages](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/docs/assets/package-map.svg)
