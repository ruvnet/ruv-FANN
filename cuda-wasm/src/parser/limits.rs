//! Conservative limits for the recursive, subset CUDA parser.

use crate::{parse_error, Result};

/// Reject oversized or deeply nested source before recursive parsing.
///
/// A source unit is limited to 1 MiB, delimiter nesting to 32, and each
/// uninterrupted expression/declaration to 256 non-whitespace characters.
/// Comments and quoted strings are skipped. The expression limit also bounds
/// unary operator recursion and left-associated AST destruction.
pub(super) fn validate_source(source: &str) -> Result<()> {
    if source.len() > 1024 * 1024 {
        return Err(parse_error!("CUDA source exceeds 1 MiB parser limit"));
    }
    let mut chars = source.chars().peekable();
    let mut depth = 0usize;
    let mut expression_len = 0usize;
    while let Some(c) = chars.next() {
        match c {
            '/' if chars.peek() == Some(&'/') => {
                chars.next();
                for c in chars.by_ref() {
                    if c == '\n' {
                        break;
                    }
                }
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                let mut closed = false;
                while let Some(c) = chars.next() {
                    if c == '*' && chars.peek() == Some(&'/') {
                        chars.next();
                        closed = true;
                        break;
                    }
                }
                if !closed {
                    return Err(parse_error!("Unterminated CUDA comment"));
                }
            }
            '\'' | '"' => {
                let quote = c;
                let mut closed = false;
                while let Some(c) = chars.next() {
                    if c == '\\' {
                        chars.next();
                    } else if c == quote {
                        closed = true;
                        break;
                    }
                }
                if !closed {
                    return Err(parse_error!("Unterminated CUDA string"));
                }
            }
            '(' | '[' | '{' => {
                depth += 1;
                if depth > 32 {
                    return Err(parse_error!("CUDA nesting exceeds parser limit of 32"));
                }
                if c == '{' {
                    expression_len = 0;
                }
            }
            ')' | ']' | '}' => {
                depth = depth.saturating_sub(1);
                if c == '}' {
                    expression_len = 0;
                }
            }
            ';' => expression_len = 0,
            _ if c.is_whitespace() => {}
            _ => {
                expression_len += 1;
                if expression_len > 256 {
                    return Err(parse_error!(
                        "CUDA expression exceeds parser limit of 256 characters"
                    ));
                }
            }
        }
    }
    Ok(())
}
