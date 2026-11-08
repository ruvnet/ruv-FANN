//! CUDA source code parser using nom combinators
//!
//! Parses a subset of CUDA C++ sufficient for common GPU kernels.

use nom::{
    IResult,
    branch::alt,
    bytes::complete::{tag, take_while, take_while1, take_until},
    character::complete::{char, multispace0, multispace1, digit1, alpha1, one_of},
    combinator::{opt, map, value, recognize},
    multi::{separated_list0, many1},
    sequence::{pair, tuple, delimited, preceded},
};

use crate::{Result, parse_error};
use super::ast::*;

// ═══════════════════════════════════════════════════════════════
//  Utility combinators
// ═══════════════════════════════════════════════════════════════

/// Whitespace + comment skipper
fn ws(input: &str) -> IResult<&str, ()> {
    let mut rest = input;
    loop {
        let (r, _) = multispace0(rest)?;
        rest = r;
        if rest.starts_with("//") {
            let end = rest.find('\n').unwrap_or(rest.len());
            rest = &rest[end..];
        } else if rest.starts_with("/*") {
            if let Some(end) = rest.find("*/") {
                rest = &rest[end + 2..];
            } else {
                return Err(nom::Err::Error(nom::error::Error::new(rest, nom::error::ErrorKind::Tag)));
            }
        } else {
            break;
        }
    }
    Ok((rest, ()))
}

/// Parse with surrounding whitespace/comments
fn ws_around<'a, F, O>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O>
where
    F: FnMut(&'a str) -> IResult<&'a str, O>,
{
    delimited(ws, inner, ws)
}

/// Parse an identifier: [a-zA-Z_][a-zA-Z0-9_]*
fn identifier(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        take_while(|c: char| c.is_alphanumeric() || c == '_'),
    ))(input)
}

/// Parse an identifier with surrounding whitespace
fn ws_ident(input: &str) -> IResult<&str, &str> {
    let (input, _) = ws(input)?;
    identifier(input)
}

/// Parse a specific tag with surrounding whitespace
fn ws_tag<'a>(t: &'a str) -> impl FnMut(&'a str) -> IResult<&'a str, &'a str> {
    delimited(ws, tag(t), ws)
}

/// Helper to call `tag` with explicit type annotation, avoiding turbofish issues
fn t<'a>(s: &'a str) -> impl FnMut(&'a str) -> IResult<&'a str, &'a str> {
    tag(s)
}

/// Check that the character after a keyword is not alphanumeric/underscore
fn keyword<'a>(kw: &'a str) -> impl FnMut(&'a str) -> IResult<&'a str, &'a str> {
    move |input: &'a str| {
        let (rest, matched) = tag(kw)(input)?;
        // Make sure it's not just a prefix of a longer identifier
        if let Some(c) = rest.chars().next() {
            if c.is_alphanumeric() || c == '_' {
                return Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag)));
            }
        }
        Ok((rest, matched))
    }
}

// ═══════════════════════════════════════════════════════════════
//  Literal parsing
// ═══════════════════════════════════════════════════════════════

fn parse_float_literal(input: &str) -> IResult<&str, Expression> {
    // Try: digits.digits[e/E[+-]digits][f/F]  OR  .digits[e/E[+-]digits][f/F]  OR  digits e/E [+-] digits [f/F]  OR  digits f/F
    let (rest, text) = recognize(alt((
        // 1.0, 1.0f, 1.0e5, 1.0e5f, 1., 1.f
        recognize(tuple((
            digit1,
            char('.'),
            opt(digit1),
            opt(recognize(tuple((one_of("eE"), opt(one_of("+-")), digit1)))),
            opt(one_of("fF")),
        ))),
        // .5, .5f, .5e3
        recognize(tuple((
            char('.'),
            digit1,
            opt(recognize(tuple((one_of("eE"), opt(one_of("+-")), digit1)))),
            opt(one_of("fF")),
        ))),
        // 1e5, 1e5f
        recognize(tuple((
            digit1,
            one_of("eE"),
            opt(one_of("+-")),
            digit1,
            opt(one_of("fF")),
        ))),
        // 1f  (integer with f suffix = float)
        recognize(tuple((digit1, one_of("fF")))),
    )))(input)?;

    let clean = text.trim_end_matches(|c| c == 'f' || c == 'F');
    let val: f64 = clean.parse().unwrap_or(0.0);
    Ok((rest, Expression::Literal(Literal::Float(val))))
}

fn parse_hex_literal(input: &str) -> IResult<&str, Expression> {
    let (rest, text) = recognize(tuple((
        tag("0"),
        one_of("xX"),
        take_while1(|c: char| c.is_ascii_hexdigit()),
        opt(one_of("uUlL")),
    )))(input)?;
    let clean = text.trim_start_matches("0x").trim_start_matches("0X");
    let clean = clean.trim_end_matches(|c| c == 'u' || c == 'U' || c == 'l' || c == 'L');
    let val = u64::from_str_radix(clean, 16).unwrap_or(0);
    if text.contains('u') || text.contains('U') {
        Ok((rest, Expression::Literal(Literal::UInt(val))))
    } else {
        Ok((rest, Expression::Literal(Literal::Int(val as i64))))
    }
}

fn parse_int_literal(input: &str) -> IResult<&str, Expression> {
    let (rest, text) = recognize(pair(
        digit1,
        opt(recognize(many1(one_of("uUlL")))),
    ))(input)?;
    let clean = text.trim_end_matches(|c| c == 'u' || c == 'U' || c == 'l' || c == 'L');
    if text.contains('u') || text.contains('U') {
        let val: u64 = clean.parse().unwrap_or(0);
        Ok((rest, Expression::Literal(Literal::UInt(val))))
    } else {
        let val: i64 = clean.parse().unwrap_or(0);
        Ok((rest, Expression::Literal(Literal::Int(val))))
    }
}

fn parse_literal(input: &str) -> IResult<&str, Expression> {
    alt((
        parse_hex_literal,
        parse_float_literal,
        parse_int_literal,
    ))(input)
}

// ═══════════════════════════════════════════════════════════════
//  Type parsing
// ═══════════════════════════════════════════════════════════════

fn parse_base_type(input: &str) -> IResult<&str, Type> {
    let (input, _) = ws(input)?;
    alt((
        value(Type::Void, keyword("void")),
        value(Type::Bool, keyword("bool")),
        // Vector types before scalar to avoid partial match
        value(Type::Vector(VectorType { element: Box::new(Type::Float(FloatType::F32)), size: 2 }), keyword("float2")),
        value(Type::Vector(VectorType { element: Box::new(Type::Float(FloatType::F32)), size: 3 }), keyword("float3")),
        value(Type::Vector(VectorType { element: Box::new(Type::Float(FloatType::F32)), size: 4 }), keyword("float4")),
        value(Type::Vector(VectorType { element: Box::new(Type::Int(IntType::I32)), size: 2 }), keyword("int2")),
        value(Type::Vector(VectorType { element: Box::new(Type::Int(IntType::I32)), size: 3 }), keyword("int3")),
        value(Type::Vector(VectorType { element: Box::new(Type::Int(IntType::I32)), size: 4 }), keyword("int4")),
        value(Type::Vector(VectorType { element: Box::new(Type::Float(FloatType::F64)), size: 2 }), keyword("double2")),
        value(Type::Vector(VectorType { element: Box::new(Type::Float(FloatType::F64)), size: 3 }), keyword("double3")),
        value(Type::Vector(VectorType { element: Box::new(Type::Float(FloatType::F64)), size: 4 }), keyword("double4")),
        value(Type::Named("dim3".to_string()), keyword("dim3")),
        value(Type::Float(FloatType::F64), keyword("double")),
        value(Type::Float(FloatType::F32), keyword("float")),
        // "unsigned int", "unsigned" alone
        map(preceded(keyword("unsigned"), opt(preceded(multispace1, keyword("int")))), |_| Type::Int(IntType::U32)),
        // "long long"
        map(pair(keyword("long"), opt(preceded(multispace1, keyword("long")))), |(_, ll)| {
            if ll.is_some() { Type::Int(IntType::I64) } else { Type::Int(IntType::I64) }
        }),
        value(Type::Int(IntType::I16), keyword("short")),
        value(Type::Int(IntType::I8), keyword("char")),
        value(Type::Int(IntType::I32), keyword("int")),
        // size_t mapped to U64
        value(Type::Int(IntType::U64), keyword("size_t")),
        // Named/user-defined type (fallback)
        map(identifier, |name: &str| Type::Named(name.to_string())),
    ))(input)
}

/// Parse a full type including const, pointer, and array suffixes
fn parse_type(input: &str) -> IResult<&str, (Type, Vec<ParamQualifier>)> {
    let (input, _) = ws(input)?;

    // Collect leading qualifiers: const, volatile, __restrict__
    let mut qualifiers = Vec::new();
    let mut rest = input;
    loop {
        let (r, _) = ws(rest)?;
        if let Ok((r2, _)) = keyword("const")(r) {
            qualifiers.push(ParamQualifier::Const);
            rest = r2;
        } else if let Ok((r2, _)) = keyword("volatile")(r) {
            qualifiers.push(ParamQualifier::Volatile);
            rest = r2;
        } else if let Ok((r2, _)) = t("__restrict__")(r) {
            qualifiers.push(ParamQualifier::Restrict);
            rest = r2;
        } else {
            break;
        }
    }

    // Parse the base type
    let (rest, mut ty) = parse_base_type(rest)?;

    // Trailing qualifiers/pointers
    let mut rest = rest;
    loop {
        let (r, _) = ws(rest)?;
        if let Ok((r2, _)) = char::<&str, nom::error::Error<&str>>('*')(r) {
            ty = Type::Pointer(Box::new(ty));
            rest = r2;
            // After *, may have const/__restrict__
            let (r3, _) = ws(rest)?;
            if let Ok((r4, _)) = keyword("const")(r3) {
                qualifiers.push(ParamQualifier::Const);
                rest = r4;
            } else if let Ok((r4, _)) = t("__restrict__")(r3) {
                qualifiers.push(ParamQualifier::Restrict);
                rest = r4;
            } else if let Ok((r4, _)) = keyword("restrict")(r3) {
                qualifiers.push(ParamQualifier::Restrict);
                rest = r4;
            } else {
                rest = r3;
            }
        } else {
            rest = r;
            break;
        }
    }

    Ok((rest, (ty, qualifiers)))
}

// ═══════════════════════════════════════════════════════════════
//  Expression parsing (precedence climbing)
// ═══════════════════════════════════════════════════════════════

/// Primary expression: literals, variables, parenthesised, casts, CUDA builtins
fn parse_primary(input: &str) -> IResult<&str, Expression> {
    let (input, _) = ws(input)?;
    alt((
        parse_cuda_builtin,
        parse_sizeof_expr,
        parse_cast_or_paren,
        parse_literal,
        parse_ident_or_call,
    ))(input)
}

/// Parse sizeof(type) or sizeof(expr)
fn parse_sizeof_expr(input: &str) -> IResult<&str, Expression> {
    let (input, _) = keyword("sizeof")(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char('(')(input)?;
    let (input, _) = ws(input)?;
    // Try to parse a type first, fall back to expression
    if let Ok((rest, (ty, _))) = parse_type(input) {
        let (rest, _) = ws(rest)?;
        let (rest, _) = char(')')(rest)?;
        // Return as a call for simplicity
        Ok((rest, Expression::Call {
            name: "sizeof".to_string(),
            args: vec![Expression::Var(format!("{:?}", ty))],
        }))
    } else {
        let (input, expr) = parse_expr(input)?;
        let (input, _) = ws(input)?;
        let (input, _) = char(')')(input)?;
        Ok((input, Expression::Call {
            name: "sizeof".to_string(),
            args: vec![expr],
        }))
    }
}

/// threadIdx.x, blockIdx.y, blockDim.z, gridDim.x
fn parse_cuda_builtin(input: &str) -> IResult<&str, Expression> {
    let (input, builtin) = alt((
        tag("threadIdx"),
        tag("blockIdx"),
        tag("blockDim"),
        tag("gridDim"),
    ))(input)?;
    // Ensure not part of a longer ident
    if let Some(c) = input.chars().next() {
        if c.is_alphanumeric() || c == '_' {
            return Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag)));
        }
    }
    let (input, _) = ws(input)?;
    let (input, _) = char('.')(input)?;
    let (input, _) = ws(input)?;
    let (input, dim_str) = alt((tag("x"), tag("y"), tag("z")))(input)?;
    let dim = match dim_str {
        "x" => Dimension::X,
        "y" => Dimension::Y,
        "z" => Dimension::Z,
        _ => unreachable!(),
    };
    let expr = match builtin {
        "threadIdx" => Expression::ThreadIdx(dim),
        "blockIdx" => Expression::BlockIdx(dim),
        "blockDim" => Expression::BlockDim(dim),
        "gridDim" => Expression::GridDim(dim),
        _ => unreachable!(),
    };
    Ok((input, expr))
}

/// Try cast `(type)expr` or parenthesised expression `(expr)`
fn parse_cast_or_paren(input: &str) -> IResult<&str, Expression> {
    let (input, _) = char('(')(input)?;
    let (input, _) = ws(input)?;

    // Try to parse as a type cast: (type)expr
    // We speculatively try parsing a type. If it succeeds and is immediately
    // followed by ')', treat it as a cast.
    let checkpoint = input;
    if let Ok((after_ty, (ty, _))) = parse_type(checkpoint) {
        let (after_ty, _) = ws(after_ty)?;
        if let Ok((after_close, _)) = char::<&str, nom::error::Error<&str>>(')')(after_ty) {
            // Check that it's actually a cast (what follows looks like an expression start)
            let (peek_rest, _) = ws(after_close)?;
            let looks_like_expr = peek_rest.starts_with('(')
                || peek_rest.starts_with(|c: char| c.is_alphanumeric() || c == '_' || c == '-' || c == '!' || c == '~' || c == '.');
            if looks_like_expr {
                // It's a cast only if the type is a real type (not just a variable name being subtracted)
                let is_real_type = matches!(ty,
                    Type::Void | Type::Bool | Type::Int(_) | Type::Float(_) | Type::Pointer(_)
                    | Type::Vector(_) | Type::Array(_, _));
                if is_real_type {
                    let (rest, expr) = parse_unary(after_close)?;
                    return Ok((rest, Expression::Cast { ty, expr: Box::new(expr) }));
                }
            }
        }
    }

    // Otherwise, parenthesised expression
    let (input, expr) = parse_expr(checkpoint)?;
    let (input, _) = ws(input)?;
    let (input, _) = char(')')(input)?;
    Ok((input, expr))
}

/// Identifier, function call, or __syncthreads()
fn parse_ident_or_call(input: &str) -> IResult<&str, Expression> {
    // __syncthreads()
    if let Ok((rest, _)) = tag::<&str, &str, nom::error::Error<&str>>("__syncthreads")(input) {
        let (rest, _) = ws(rest)?;
        if let Ok((rest, _)) = char::<&str, nom::error::Error<&str>>('(')(rest) {
            let (rest, _) = ws(rest)?;
            let (rest, _) = char(')')(rest)?;
            // We'll handle this as a special call that gets turned into SyncThreads statement
            return Ok((rest, Expression::Call { name: "__syncthreads".to_string(), args: vec![] }));
        }
    }

    let (input, name) = identifier(input)?;
    let (input, _) = ws(input)?;

    // Check for function call
    if let Ok((rest, _)) = char::<&str, nom::error::Error<&str>>('(')(input) {
        let (rest, _) = ws(rest)?;
        let (rest, args) = separated_list0(
            delimited(ws, char(','), ws),
            parse_expr,
        )(rest)?;
        let (rest, _) = ws(rest)?;
        let (rest, _) = char(')')(rest)?;

        // Detect warp primitives
        let expr = match name {
            "__shfl_sync" => Expression::WarpPrimitive { op: WarpOp::Shuffle, args },
            "__shfl_xor_sync" => Expression::WarpPrimitive { op: WarpOp::ShuffleXor, args },
            "__shfl_up_sync" => Expression::WarpPrimitive { op: WarpOp::ShuffleUp, args },
            "__shfl_down_sync" => Expression::WarpPrimitive { op: WarpOp::ShuffleDown, args },
            "__ballot_sync" => Expression::WarpPrimitive { op: WarpOp::Ballot, args },
            "__activemask" => Expression::WarpPrimitive { op: WarpOp::ActiveMask, args },
            _ => Expression::Call { name: name.to_string(), args },
        };
        return Ok((rest, expr));
    }

    Ok((input, Expression::Var(name.to_string())))
}

/// Postfix: a[i], a.field, a->field, a++, a--
fn parse_postfix(input: &str) -> IResult<&str, Expression> {
    let (mut rest, mut expr) = parse_primary(input)?;

    loop {
        let (r, _) = ws(rest)?;

        // Array index: expr[index]
        if let Ok((r2, _)) = char::<&str, nom::error::Error<&str>>('[')(r) {
            let (r2, _) = ws(r2)?;
            let (r2, index) = parse_expr(r2)?;
            let (r2, _) = ws(r2)?;
            let (r2, _) = char(']')(r2)?;
            expr = Expression::Index {
                array: Box::new(expr),
                index: Box::new(index),
            };
            rest = r2;
            continue;
        }

        // Member access: expr.field (but not after CUDA builtins which already consumed the dot)
        if let Ok((r2, _)) = char::<&str, nom::error::Error<&str>>('.')(r) {
            if let Ok((r3, field)) = identifier(r2) {
                // Make sure it's not a float literal like ".5"
                expr = Expression::Member {
                    object: Box::new(expr),
                    field: field.to_string(),
                };
                rest = r3;
                continue;
            }
        }

        // Arrow: expr->field
        if let Ok((r2, _)) = tag::<&str, &str, nom::error::Error<&str>>("->")(r) {
            let (r3, field) = identifier(r2)?;
            expr = Expression::Member {
                object: Box::new(expr),
                field: field.to_string(),
            };
            rest = r3;
            continue;
        }

        // Post-increment: expr++
        if let Ok((r2, _)) = tag::<&str, &str, nom::error::Error<&str>>("++")(r) {
            expr = Expression::Unary {
                op: UnaryOp::PostInc,
                expr: Box::new(expr),
            };
            rest = r2;
            continue;
        }

        // Post-decrement: expr--
        if let Ok((r2, _)) = tag::<&str, &str, nom::error::Error<&str>>("--")(r) {
            expr = Expression::Unary {
                op: UnaryOp::PostDec,
                expr: Box::new(expr),
            };
            rest = r2;
            continue;
        }

        rest = r;
        break;
    }

    Ok((rest, expr))
}

/// Unary prefix: ++x, --x, -x, !x, ~x, *x, &x
fn parse_unary(input: &str) -> IResult<&str, Expression> {
    let (input, _) = ws(input)?;

    // Pre-increment
    if let Ok((rest, _)) = tag::<&str, &str, nom::error::Error<&str>>("++")(input) {
        let (rest, expr) = parse_unary(rest)?;
        return Ok((rest, Expression::Unary { op: UnaryOp::PreInc, expr: Box::new(expr) }));
    }
    // Pre-decrement
    if let Ok((rest, _)) = tag::<&str, &str, nom::error::Error<&str>>("--")(input) {
        let (rest, expr) = parse_unary(rest)?;
        return Ok((rest, Expression::Unary { op: UnaryOp::PreDec, expr: Box::new(expr) }));
    }
    // Unary minus (not --> )
    if input.starts_with('-') && !input.starts_with("--") && !input.starts_with("->") {
        let (rest, _) = char('-')(input)?;
        let (rest, expr) = parse_unary(rest)?;
        return Ok((rest, Expression::Unary { op: UnaryOp::Neg, expr: Box::new(expr) }));
    }
    // Logical NOT
    if input.starts_with('!') && !input.starts_with("!=") {
        let (rest, _) = char('!')(input)?;
        let (rest, expr) = parse_unary(rest)?;
        return Ok((rest, Expression::Unary { op: UnaryOp::Not, expr: Box::new(expr) }));
    }
    // Bitwise NOT
    if input.starts_with('~') {
        let (rest, _) = char('~')(input)?;
        let (rest, expr) = parse_unary(rest)?;
        return Ok((rest, Expression::Unary { op: UnaryOp::BitNot, expr: Box::new(expr) }));
    }
    // Dereference
    if input.starts_with('*') && !input.starts_with("*=") {
        let (rest, _) = char('*')(input)?;
        let (rest, expr) = parse_unary(rest)?;
        return Ok((rest, Expression::Unary { op: UnaryOp::Deref, expr: Box::new(expr) }));
    }
    // Address-of
    if input.starts_with('&') && !input.starts_with("&&") && !input.starts_with("&=") {
        let (rest, _) = char('&')(input)?;
        let (rest, expr) = parse_unary(rest)?;
        return Ok((rest, Expression::Unary { op: UnaryOp::AddrOf, expr: Box::new(expr) }));
    }

    parse_postfix(input)
}

/// Binary expression using precedence climbing
fn parse_expr(input: &str) -> IResult<&str, Expression> {
    parse_assignment(input)
}

fn parse_assignment(input: &str) -> IResult<&str, Expression> {
    let (mut rest, mut left) = parse_ternary(input)?;

    loop {
        let (r, _) = ws(rest)?;

        // Try compound assignments first (longer tokens before shorter)
        let compound_op: Option<(usize, BinaryOp)> = if r.starts_with("<<=") {
            Some((3, BinaryOp::Shl))
        } else if r.starts_with(">>=") {
            Some((3, BinaryOp::Shr))
        } else if r.starts_with("+=") {
            Some((2, BinaryOp::Add))
        } else if r.starts_with("-=") {
            Some((2, BinaryOp::Sub))
        } else if r.starts_with("*=") {
            Some((2, BinaryOp::Mul))
        } else if r.starts_with("/=") {
            Some((2, BinaryOp::Div))
        } else if r.starts_with("%=") {
            Some((2, BinaryOp::Mod))
        } else if r.starts_with("&=") {
            Some((2, BinaryOp::And))
        } else if r.starts_with("|=") {
            Some((2, BinaryOp::Or))
        } else if r.starts_with("^=") {
            Some((2, BinaryOp::Xor))
        } else {
            None
        };

        if let Some((len, op)) = compound_op {
            let r2 = &r[len..];
            let (r2, right) = parse_assignment(r2)?;
            // Desugar: a += b  =>  a = a + b
            left = Expression::Binary {
                op: BinaryOp::Assign,
                left: Box::new(left.clone()),
                right: Box::new(Expression::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                }),
            };
            rest = r2;
            continue;
        }

        // Simple assignment: = but not ==
        if r.starts_with('=') && !r.starts_with("==") {
            let r2 = &r[1..];
            let (r2, right) = parse_assignment(r2)?;
            left = Expression::Binary { op: BinaryOp::Assign, left: Box::new(left), right: Box::new(right) };
            rest = r2;
            continue;
        }

        break;
    }

    Ok((rest, left))
}

fn parse_ternary(input: &str) -> IResult<&str, Expression> {
    let (rest, cond) = parse_logical_or(input)?;
    let (r, _) = ws(rest)?;
    if let Ok((r2, _)) = char::<&str, nom::error::Error<&str>>('?')(r) {
        let (r2, then_expr) = parse_expr(r2)?;
        let (r2, _) = ws(r2)?;
        let (r2, _) = char(':')(r2)?;
        let (r2, else_expr) = parse_ternary(r2)?;
        // Represent ternary as an if-like construct using Call for now
        // Actually, let's return it as a special call  __ternary__(cond, then, else)
        // The AST doesn't have ternary, so we use a synthetic representation
        Ok((r2, Expression::Call {
            name: "__ternary__".to_string(),
            args: vec![cond, then_expr, else_expr],
        }))
    } else {
        Ok((rest, cond))
    }
}

// ── Binary operator levels ──────────────────────────────────

fn parse_logical_or(input: &str) -> IResult<&str, Expression> {
    let (mut rest, mut left) = parse_logical_and(input)?;
    loop {
        let (r, _) = ws(rest)?;
        if let Ok((r2, _)) = tag::<&str, &str, nom::error::Error<&str>>("||")(r) {
            let (r2, right) = parse_logical_and(r2)?;
            left = Expression::Binary { op: BinaryOp::LogicalOr, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else {
            rest = r;
            break;
        }
    }
    Ok((rest, left))
}

fn parse_logical_and(input: &str) -> IResult<&str, Expression> {
    let (mut rest, mut left) = parse_bitwise_or(input)?;
    loop {
        let (r, _) = ws(rest)?;
        if let Ok((r2, _)) = tag::<&str, &str, nom::error::Error<&str>>("&&")(r) {
            let (r2, right) = parse_bitwise_or(r2)?;
            left = Expression::Binary { op: BinaryOp::LogicalAnd, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else {
            rest = r;
            break;
        }
    }
    Ok((rest, left))
}

fn parse_bitwise_or(input: &str) -> IResult<&str, Expression> {
    let (mut rest, mut left) = parse_bitwise_xor(input)?;
    loop {
        let (r, _) = ws(rest)?;
        // | but not ||
        if r.starts_with('|') && !r.starts_with("||") && !r.starts_with("|=") {
            let (r2, _) = char('|')(r)?;
            let (r2, right) = parse_bitwise_xor(r2)?;
            left = Expression::Binary { op: BinaryOp::Or, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else {
            rest = r;
            break;
        }
    }
    Ok((rest, left))
}

fn parse_bitwise_xor(input: &str) -> IResult<&str, Expression> {
    let (mut rest, mut left) = parse_bitwise_and(input)?;
    loop {
        let (r, _) = ws(rest)?;
        if r.starts_with('^') && !r.starts_with("^=") {
            let (r2, _) = char('^')(r)?;
            let (r2, right) = parse_bitwise_and(r2)?;
            left = Expression::Binary { op: BinaryOp::Xor, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else {
            rest = r;
            break;
        }
    }
    Ok((rest, left))
}

fn parse_bitwise_and(input: &str) -> IResult<&str, Expression> {
    let (mut rest, mut left) = parse_equality(input)?;
    loop {
        let (r, _) = ws(rest)?;
        // & but not && or &=
        if r.starts_with('&') && !r.starts_with("&&") && !r.starts_with("&=") {
            let (r2, _) = char('&')(r)?;
            let (r2, right) = parse_equality(r2)?;
            left = Expression::Binary { op: BinaryOp::And, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else {
            rest = r;
            break;
        }
    }
    Ok((rest, left))
}

fn parse_equality(input: &str) -> IResult<&str, Expression> {
    let (mut rest, mut left) = parse_relational(input)?;
    loop {
        let (r, _) = ws(rest)?;
        if let Ok((r2, _)) = tag::<&str, &str, nom::error::Error<&str>>("==")(r) {
            let (r2, right) = parse_relational(r2)?;
            left = Expression::Binary { op: BinaryOp::Eq, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else if let Ok((r2, _)) = tag::<&str, &str, nom::error::Error<&str>>("!=")(r) {
            let (r2, right) = parse_relational(r2)?;
            left = Expression::Binary { op: BinaryOp::Ne, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else {
            rest = r;
            break;
        }
    }
    Ok((rest, left))
}

fn parse_relational(input: &str) -> IResult<&str, Expression> {
    let (mut rest, mut left) = parse_shift(input)?;
    loop {
        let (r, _) = ws(rest)?;
        if r.starts_with("<=") {
            let r2 = &r[2..];
            let (r2, right) = parse_shift(r2)?;
            left = Expression::Binary { op: BinaryOp::Le, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else if r.starts_with(">=") {
            let r2 = &r[2..];
            let (r2, right) = parse_shift(r2)?;
            left = Expression::Binary { op: BinaryOp::Ge, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else if r.starts_with('<') && !r.starts_with("<<") {
            let r2 = &r[1..];
            let (r2, right) = parse_shift(r2)?;
            left = Expression::Binary { op: BinaryOp::Lt, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else if r.starts_with('>') && !r.starts_with(">>") {
            let r2 = &r[1..];
            let (r2, right) = parse_shift(r2)?;
            left = Expression::Binary { op: BinaryOp::Gt, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else {
            rest = r;
            break;
        }
    }
    Ok((rest, left))
}

fn parse_shift(input: &str) -> IResult<&str, Expression> {
    let (mut rest, mut left) = parse_additive(input)?;
    loop {
        let (r, _) = ws(rest)?;
        if r.starts_with("<<=") {
            rest = r;
            break;
        } else if r.starts_with(">>=") {
            rest = r;
            break;
        } else if r.starts_with("<<") {
            let r2 = &r[2..];
            let (r2, right) = parse_additive(r2)?;
            left = Expression::Binary { op: BinaryOp::Shl, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else if r.starts_with(">>") {
            let r2 = &r[2..];
            let (r2, right) = parse_additive(r2)?;
            left = Expression::Binary { op: BinaryOp::Shr, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else {
            rest = r;
            break;
        }
    }
    Ok((rest, left))
}

fn parse_additive(input: &str) -> IResult<&str, Expression> {
    let (mut rest, mut left) = parse_multiplicative(input)?;
    loop {
        let (r, _) = ws(rest)?;
        if r.starts_with('+') && !r.starts_with("++") && !r.starts_with("+=") {
            let (r2, _) = char('+')(r)?;
            let (r2, right) = parse_multiplicative(r2)?;
            left = Expression::Binary { op: BinaryOp::Add, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else if r.starts_with('-') && !r.starts_with("--") && !r.starts_with("-=") && !r.starts_with("->") {
            let (r2, _) = char('-')(r)?;
            let (r2, right) = parse_multiplicative(r2)?;
            left = Expression::Binary { op: BinaryOp::Sub, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else {
            rest = r;
            break;
        }
    }
    Ok((rest, left))
}

fn parse_multiplicative(input: &str) -> IResult<&str, Expression> {
    let (mut rest, mut left) = parse_unary(input)?;
    loop {
        let (r, _) = ws(rest)?;
        if r.starts_with('*') && !r.starts_with("*=") {
            let (r2, _) = char('*')(r)?;
            let (r2, right) = parse_unary(r2)?;
            left = Expression::Binary { op: BinaryOp::Mul, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else if r.starts_with('/') && !r.starts_with("/=") && !r.starts_with("//") && !r.starts_with("/*") {
            let (r2, _) = char('/')(r)?;
            let (r2, right) = parse_unary(r2)?;
            left = Expression::Binary { op: BinaryOp::Div, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else if r.starts_with('%') && !r.starts_with("%=") {
            let (r2, _) = char('%')(r)?;
            let (r2, right) = parse_unary(r2)?;
            left = Expression::Binary { op: BinaryOp::Mod, left: Box::new(left), right: Box::new(right) };
            rest = r2;
        } else {
            rest = r;
            break;
        }
    }
    Ok((rest, left))
}

// ═══════════════════════════════════════════════════════════════
//  Statement parsing
// ═══════════════════════════════════════════════════════════════

fn parse_block(input: &str) -> IResult<&str, Block> {
    let (input, _) = ws(input)?;
    let (input, _) = char('{')(input)?;
    let (input, stmts) = parse_statement_list(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char('}')(input)?;
    Ok((input, Block { statements: stmts }))
}

fn parse_statement_list(input: &str) -> IResult<&str, Vec<Statement>> {
    let mut stmts = Vec::new();
    let mut rest = input;
    loop {
        let (r, _) = ws(rest)?;
        if r.starts_with('}') || r.is_empty() {
            rest = r;
            break;
        }
        match parse_statement(r) {
            Ok((r2, stmt)) => {
                stmts.push(stmt);
                rest = r2;
            }
            Err(_) => {
                // Skip unrecognized token and try again (error recovery)
                if let Some(pos) = r.find(|c: char| c == ';' || c == '}') {
                    if r.as_bytes()[pos] == b';' {
                        rest = &r[pos + 1..];
                    } else {
                        rest = &r[pos..];
                    }
                } else {
                    break;
                }
            }
        }
    }
    Ok((rest, stmts))
}

fn parse_statement(input: &str) -> IResult<&str, Statement> {
    let (input, _) = ws(input)?;
    alt((
        parse_syncthreads_stmt,
        parse_return_stmt,
        parse_break_stmt,
        parse_continue_stmt,
        parse_if_stmt,
        parse_for_stmt,
        parse_while_stmt,
        parse_do_while_stmt,
        parse_block_stmt,
        parse_var_decl_stmt,
        parse_expr_stmt,
    ))(input)
}

fn parse_syncthreads_stmt(input: &str) -> IResult<&str, Statement> {
    let (input, _) = tag("__syncthreads")(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char('(')(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char(')')(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char(';')(input)?;
    Ok((input, Statement::SyncThreads))
}

fn parse_return_stmt(input: &str) -> IResult<&str, Statement> {
    let (input, _) = keyword("return")(input)?;
    let (input, _) = ws(input)?;
    if let Ok((rest, _)) = char::<&str, nom::error::Error<&str>>(';')(input) {
        return Ok((rest, Statement::Return(None)));
    }
    let (input, expr) = parse_expr(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char(';')(input)?;
    Ok((input, Statement::Return(Some(expr))))
}

fn parse_break_stmt(input: &str) -> IResult<&str, Statement> {
    let (input, _) = keyword("break")(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char(';')(input)?;
    Ok((input, Statement::Break))
}

fn parse_continue_stmt(input: &str) -> IResult<&str, Statement> {
    let (input, _) = keyword("continue")(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char(';')(input)?;
    Ok((input, Statement::Continue))
}

fn parse_if_stmt(input: &str) -> IResult<&str, Statement> {
    let (input, _) = keyword("if")(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char('(')(input)?;
    let (input, condition) = parse_expr(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char(')')(input)?;
    let (input, then_branch) = parse_statement(input)?;
    let (input, _) = ws(input)?;
    let (input, else_branch) = opt(preceded(
        pair(keyword("else"), ws),
        parse_statement,
    ))(input)?;

    Ok((input, Statement::If {
        condition,
        then_branch: Box::new(then_branch),
        else_branch: else_branch.map(Box::new),
    }))
}

fn parse_for_stmt(input: &str) -> IResult<&str, Statement> {
    let (input, _) = keyword("for")(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char('(')(input)?;

    // Init: either a var decl or an expression statement, or empty
    let (input, _) = ws(input)?;
    let (input, init) = if let Ok((r, _)) = char::<&str, nom::error::Error<&str>>(';')(input) {
        (r, None)
    } else if let Ok((r, stmt)) = parse_var_decl_stmt(input) {
        // var_decl_stmt already consumes the semicolon
        (r, Some(Box::new(stmt)))
    } else {
        let (r, expr) = parse_expr(input)?;
        let (r, _) = ws(r)?;
        let (r, _) = char(';')(r)?;
        (r, Some(Box::new(Statement::Expr(expr))))
    };

    // Condition
    let (input, _) = ws(input)?;
    let (input, condition) = if let Ok((r, _)) = char::<&str, nom::error::Error<&str>>(';')(input) {
        (r, None)
    } else {
        let (r, expr) = parse_expr(input)?;
        let (r, _) = ws(r)?;
        let (r, _) = char(';')(r)?;
        (r, Some(expr))
    };

    // Update
    let (input, _) = ws(input)?;
    let (input, update) = if let Ok((r, _)) = char::<&str, nom::error::Error<&str>>(')')(input) {
        (r, None)
    } else {
        let (r, expr) = parse_expr(input)?;
        let (r, _) = ws(r)?;
        let (r, _) = char(')')(r)?;
        (r, Some(expr))
    };

    let (input, body) = parse_statement(input)?;

    Ok((input, Statement::For {
        init,
        condition,
        update,
        body: Box::new(body),
    }))
}

fn parse_while_stmt(input: &str) -> IResult<&str, Statement> {
    let (input, _) = keyword("while")(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char('(')(input)?;
    let (input, condition) = parse_expr(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char(')')(input)?;
    let (input, body) = parse_statement(input)?;

    Ok((input, Statement::While {
        condition,
        body: Box::new(body),
    }))
}

fn parse_do_while_stmt(input: &str) -> IResult<&str, Statement> {
    let (input, _) = keyword("do")(input)?;
    let (input, body) = parse_statement(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = keyword("while")(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char('(')(input)?;
    let (input, condition) = parse_expr(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char(')')(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char(';')(input)?;

    Ok((input, Statement::While {
        condition,
        body: Box::new(body),
    }))
}

fn parse_block_stmt(input: &str) -> IResult<&str, Statement> {
    let (input, block) = parse_block(input)?;
    Ok((input, Statement::Block(block)))
}

/// Try to detect if the next tokens look like a variable declaration.
/// This is the key heuristic: we look for patterns like:
///   type name [= init] ;
///   type name [ size ] [= init] ;
///   extern __shared__ type name [] ;
///   __shared__ type name [ size ] ;
fn parse_var_decl_stmt(input: &str) -> IResult<&str, Statement> {
    let (input, _) = ws(input)?;

    // Storage class qualifiers
    let mut storage = StorageClass::Auto;
    let mut rest = input;
    let mut has_extern = false;

    // extern keyword
    if let Ok((r, _)) = keyword("extern")(rest) {
        has_extern = true;
        rest = r;
        let (r, _) = ws(rest)?;
        rest = r;
    }

    // __shared__, __constant__, register, static
    if let Ok((r, _)) = tag::<&str, &str, nom::error::Error<&str>>("__shared__")(rest) {
        storage = StorageClass::Shared;
        rest = r;
    } else if let Ok((r, _)) = tag::<&str, &str, nom::error::Error<&str>>("__constant__")(rest) {
        storage = StorageClass::Constant;
        rest = r;
    } else if let Ok((r, _)) = keyword("register")(rest) {
        storage = StorageClass::Register;
        rest = r;
    } else if let Ok((r, _)) = keyword("static")(rest) {
        // Keep Auto for static locals
        rest = r;
    } else if has_extern {
        // Just "extern" without __shared__/__constant__ - not a var decl we handle
        return Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag)));
    }

    // Parse the type
    let (rest, (mut ty, qualifiers)) = parse_type(rest)?;
    let (rest, _) = ws(rest)?;

    // Need an identifier here. Make sure it's not a keyword or `(` (which would be a function call).
    let (rest, name) = identifier(rest)?;

    // Check that name is not a keyword that starts a statement
    let kw_set = ["if", "else", "for", "while", "do", "return", "break", "continue",
                   "switch", "case", "default", "goto", "__syncthreads"];
    if kw_set.contains(&name) {
        return Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag)));
    }

    let (rest, _) = ws(rest)?;

    // Array suffix: [size] or []
    if let Ok((r, _)) = char::<&str, nom::error::Error<&str>>('[')(rest) {
        let (r, _) = ws(r)?;
        if let Ok((r, _)) = char::<&str, nom::error::Error<&str>>(']')(r) {
            ty = Type::Array(Box::new(ty), None);
            let (r, _) = ws(r)?;
            // Optional additional dimensions: [16][16]
            let mut r = r;
            while let Ok((r2, _)) = char::<&str, nom::error::Error<&str>>('[')(r) {
                let (r2, _) = ws(r2)?;
                let (r2, size_expr) = parse_expr(r2)?;
                let (r2, _) = ws(r2)?;
                let (r2, _) = char(']')(r2)?;
                let (r2, _) = ws(r2)?;
                r = r2;
                // We wrap in nested Array types
                // (simplification: we don't track multi-dim precisely)
            }
            let (r, _) = ws(r)?;
            let (r, _) = char(';')(r)?;
            return Ok((r, Statement::VarDecl {
                name: name.to_string(),
                ty,
                init: None,
                storage,
            }));
        } else {
            // [size] - possibly multi-dimensional: [16][16]
            let (r, size_expr) = parse_expr(r)?;
            let (r, _) = ws(r)?;
            let (r, _) = char(']')(r)?;
            let size = if let Expression::Literal(Literal::Int(n)) = &size_expr {
                Some(*n as usize)
            } else {
                None
            };
            ty = Type::Array(Box::new(ty), size);
            let mut r = r;
            let (r2, _) = ws(r)?;
            // Additional dimensions
            while let Ok((r3, _)) = char::<&str, nom::error::Error<&str>>('[')(r2) {
                let (r3, _) = ws(r3)?;
                let (r3, _size2) = parse_expr(r3)?;
                let (r3, _) = ws(r3)?;
                let (r3, _) = char(']')(r3)?;
                r = r3;
                let (r4, _) = ws(r)?;
                // Check for more dimensions
                if r4.starts_with('[') {
                    continue;
                }
                break;
            }
            let (r, _) = ws(r)?;
            // Optional initializer
            let (r, init) = if let Ok((r2, _)) = char::<&str, nom::error::Error<&str>>('=')(r) {
                let (r2, expr) = parse_expr(r2)?;
                (r2, Some(expr))
            } else {
                (r, None)
            };
            let (r, _) = ws(r)?;
            let (r, _) = char(';')(r)?;
            return Ok((r, Statement::VarDecl {
                name: name.to_string(),
                ty,
                init,
                storage,
            }));
        }
    }

    // Optional initializer: = expr
    let (rest, init) = if let Ok((r, _)) = char::<&str, nom::error::Error<&str>>('=')(rest) {
        let (r, expr) = parse_expr(r)?;
        (r, Some(expr))
    } else {
        (rest, None)
    };

    let (rest, _) = ws(rest)?;
    let (rest, _) = char(';')(rest)?;

    Ok((rest, Statement::VarDecl {
        name: name.to_string(),
        ty,
        init,
        storage,
    }))
}

fn parse_expr_stmt(input: &str) -> IResult<&str, Statement> {
    let (input, expr) = parse_expr(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char(';')(input)?;

    // Convert __syncthreads() call to SyncThreads statement
    if let Expression::Call { ref name, ref args } = expr {
        if name == "__syncthreads" && args.is_empty() {
            return Ok((input, Statement::SyncThreads));
        }
    }

    Ok((input, Statement::Expr(expr)))
}

// ═══════════════════════════════════════════════════════════════
//  Top-level item parsing
// ═══════════════════════════════════════════════════════════════

fn parse_parameter(input: &str) -> IResult<&str, Parameter> {
    let (input, _) = ws(input)?;
    let (input, (ty, qualifiers)) = parse_type(input)?;
    let (input, _) = ws(input)?;
    let (input, name) = identifier(input)?;
    // Optional array suffix on parameter: int arr[]
    let (input, _) = ws(input)?;
    let (input, _ty) = if let Ok((r, _)) = char::<&str, nom::error::Error<&str>>('[')(input) {
        let (r, _) = ws(r)?;
        if let Ok((r, _)) = char::<&str, nom::error::Error<&str>>(']')(r) {
            (r, Type::Pointer(Box::new(ty.clone())))
        } else {
            let (r, _) = parse_expr(r)?;
            let (r, _) = ws(r)?;
            let (r, _) = char(']')(r)?;
            (r, Type::Pointer(Box::new(ty.clone())))
        }
    } else {
        (input, ty.clone())
    };

    Ok((input, Parameter {
        name: name.to_string(),
        ty: _ty,
        qualifiers,
    }))
}

fn parse_param_list(input: &str) -> IResult<&str, Vec<Parameter>> {
    let (input, _) = ws(input)?;
    let (input, _) = char('(')(input)?;
    let (input, _) = ws(input)?;
    // Handle empty param list and void param list
    if let Ok((r, _)) = char::<&str, nom::error::Error<&str>>(')')(input) {
        return Ok((r, vec![]));
    }
    if let Ok((r, _)) = keyword("void")(input) {
        let (r, _) = ws(r)?;
        if let Ok((r, _)) = char::<&str, nom::error::Error<&str>>(')')(r) {
            return Ok((r, vec![]));
        }
    }
    let (input, params) = separated_list0(
        delimited(ws, char(','), ws),
        parse_parameter,
    )(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char(')')(input)?;
    Ok((input, params))
}

/// Parse a kernel definition: __global__ void name(params) { body }
fn parse_kernel_def(input: &str) -> IResult<&str, Item> {
    let (input, _) = ws(input)?;
    // Optional template<...> - skip it
    let input = skip_template(input);
    let (input, _) = ws(input)?;
    let (input, _) = tag("__global__")(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = keyword("void")(input)?;
    let (input, _) = ws(input)?;
    let (input, name) = identifier(input)?;
    let (input, params) = parse_param_list(input)?;
    let (input, body) = parse_block(input)?;

    Ok((input, Item::Kernel(KernelDef {
        name: name.to_string(),
        params,
        body,
        attributes: vec![],
    })))
}

/// Parse a __device__ function
fn parse_device_function(input: &str) -> IResult<&str, Item> {
    let (input, _) = ws(input)?;
    let input = skip_template(input);
    let (input, _) = ws(input)?;
    let (input, _) = tag("__device__")(input)?;
    let (input, _) = ws(input)?;
    // May also have __host__ qualifier
    let (input, also_host) = opt(preceded(tag("__host__"), ws))(input)?;
    // May have __forceinline__
    let (input, _) = opt(preceded(tag("__forceinline__"), ws))(input)?;
    let (input, _) = opt(preceded(keyword("inline"), ws))(input)?;
    let (input, (ret_ty, _)) = parse_type(input)?;
    let (input, _) = ws(input)?;
    let (input, name) = identifier(input)?;
    let (input, params) = parse_param_list(input)?;
    let (input, body) = parse_block(input)?;

    let mut qualifiers = vec![FunctionQualifier::Device];
    if also_host.is_some() {
        qualifiers.push(FunctionQualifier::Host);
    }

    Ok((input, Item::DeviceFunction(FunctionDef {
        name: name.to_string(),
        return_type: ret_ty,
        params,
        body,
        qualifiers,
    })))
}

/// Parse a __host__ function
fn parse_host_function(input: &str) -> IResult<&str, Item> {
    let (input, _) = ws(input)?;
    let input = skip_template(input);
    let (input, _) = ws(input)?;
    let (input, _) = tag("__host__")(input)?;
    let (input, _) = ws(input)?;
    // May also have __device__
    let (input, also_device) = opt(preceded(tag("__device__"), ws))(input)?;
    let (input, _) = opt(preceded(keyword("inline"), ws))(input)?;
    let (input, (ret_ty, _)) = parse_type(input)?;
    let (input, _) = ws(input)?;
    let (input, name) = identifier(input)?;
    let (input, params) = parse_param_list(input)?;
    let (input, body) = parse_block(input)?;

    let mut qualifiers = vec![FunctionQualifier::Host];
    if also_device.is_some() {
        qualifiers.push(FunctionQualifier::Device);
    }

    Ok((input, Item::HostFunction(FunctionDef {
        name: name.to_string(),
        return_type: ret_ty,
        params,
        body,
        qualifiers,
    })))
}

/// Parse an #include directive
fn parse_include(input: &str) -> IResult<&str, Item> {
    let (input, _) = ws(input)?;
    let (input, _) = char('#')(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = tag("include")(input)?;
    let (input, _) = take_while(|c: char| c == ' ' || c == '\t')(input)?;
    // Match <...> or "..."
    let (input, path) = alt((
        delimited(char('<'), take_until(">"), char('>')),
        delimited(char('"'), take_until("\""), char('"')),
    ))(input)?;
    // Consume rest of line
    let (input, _) = take_while(|c: char| c != '\n')(input)?;
    Ok((input, Item::Include(path.to_string())))
}

/// Parse a typedef: typedef type name;
fn parse_typedef(input: &str) -> IResult<&str, Item> {
    let (input, _) = ws(input)?;
    let (input, _) = keyword("typedef")(input)?;
    let (input, _) = ws(input)?;

    // Handle typedef struct { ... } Name;
    if let Ok((r, _)) = keyword("struct")(input) {
        let (r, _) = ws(r)?;
        // Optional struct name
        let (r, _struct_name) = opt(identifier)(r)?;
        let (r, _) = ws(r)?;
        // Struct body - just skip it
        if let Ok((r, _)) = char::<&str, nom::error::Error<&str>>('{')(r) {
            let r = skip_balanced_braces(r);
            let (r, _) = ws(r)?;
            let (r, name) = identifier(r)?;
            let (r, _) = ws(r)?;
            let (r, _) = char(';')(r)?;
            return Ok((r, Item::TypeDef(TypeDef {
                name: name.to_string(),
                ty: Type::Named(name.to_string()),
            })));
        }
    }

    let (input, (ty, _)) = parse_type(input)?;
    let (input, _) = ws(input)?;
    let (input, name) = identifier(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char(';')(input)?;
    Ok((input, Item::TypeDef(TypeDef {
        name: name.to_string(),
        ty,
    })))
}

/// Parse a struct definition (non-typedef)
fn parse_struct(input: &str) -> IResult<&str, Item> {
    let (input, _) = ws(input)?;
    let (input, _) = keyword("struct")(input)?;
    let (input, _) = ws(input)?;
    let (input, name) = identifier(input)?;
    let (input, _) = ws(input)?;
    let (input, _) = char('{')(input)?;
    let rest = skip_balanced_braces(input);
    let (rest, _) = ws(rest)?;
    let (rest, _) = char(';')(rest)?;
    Ok((rest, Item::TypeDef(TypeDef {
        name: name.to_string(),
        ty: Type::Named(name.to_string()),
    })))
}

/// Skip template<...> prefixes
fn skip_template(input: &str) -> &str {
    let trimmed = input.trim_start();
    if !trimmed.starts_with("template") {
        return input;
    }
    let rest = &trimmed[8..];
    let rest = rest.trim_start();
    if !rest.starts_with('<') {
        return input;
    }
    let mut depth = 0;
    for (i, c) in rest.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    return &rest[i + 1..];
                }
            }
            _ => {}
        }
    }
    input
}

/// Skip balanced braces, returning the rest after the closing '}'
fn skip_balanced_braces(input: &str) -> &str {
    let mut depth = 1;
    for (i, c) in input.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &input[i + 1..];
                }
            }
            _ => {}
        }
    }
    input
}

/// Skip a preprocessor directive (everything until end of line)
fn parse_preprocessor(input: &str) -> IResult<&str, ()> {
    let (input, _) = ws(input)?;
    let (input, _) = char('#')(input)?;
    let (input, _) = take_while(|c: char| c != '\n')(input)?;
    Ok((input, ()))
}

// ═══════════════════════════════════════════════════════════════
//  Top-level parser
// ═══════════════════════════════════════════════════════════════

/// Parse a top-level global variable declaration (__constant__ or __shared__)
fn parse_global_var_decl(input: &str) -> IResult<&str, Item> {
    let (input, _) = ws(input)?;
    // Only match __constant__ or __shared__ at top level
    let rest = input;
    let storage;
    let rest = if let Ok((r, _)) = tag::<&str, &str, nom::error::Error<&str>>("__constant__")(rest) {
        storage = StorageClass::Constant;
        r
    } else if let Ok((r, _)) = tag::<&str, &str, nom::error::Error<&str>>("__shared__")(rest) {
        storage = StorageClass::Shared;
        r
    } else {
        return Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag)));
    };
    let (rest, _) = ws(rest)?;
    let (rest, (ty, _qualifiers)) = parse_type(rest)?;
    let (rest, _) = ws(rest)?;
    let (rest, name) = identifier(rest)?;
    let (rest, _) = ws(rest)?;
    // Array suffix
    let (rest, ty) = if let Ok((r, _)) = char::<&str, nom::error::Error<&str>>('[')(rest) {
        let (r, _) = ws(r)?;
        let (r, size_expr) = parse_expr(r)?;
        let (r, _) = ws(r)?;
        let (r, _) = char(']')(r)?;
        let size = if let Expression::Literal(Literal::Int(n)) = &size_expr {
            Some(*n as usize)
        } else {
            None
        };
        (r, Type::Array(Box::new(ty), size))
    } else {
        (rest, ty)
    };
    let (rest, _) = ws(rest)?;
    // Optional initializer
    let (rest, init) = if let Ok((r, _)) = char::<&str, nom::error::Error<&str>>('=')(rest) {
        let (r, _) = ws(r)?;
        // Handle brace-enclosed initializers: {1.0, 2.0, ...}
        if r.starts_with('{') {
            // Skip to matching }
            let end = r.find('}').unwrap_or(r.len() - 1);
            let r = &r[end + 1..];
            (r, None) // We don't parse initializer lists into AST yet
        } else {
            let (r, expr) = parse_expr(r)?;
            (r, Some(expr))
        }
    } else {
        (rest, None)
    };
    let (rest, _) = ws(rest)?;
    let (rest, _) = char(';')(rest)?;
    Ok((rest, Item::GlobalVar(GlobalVar {
        name: name.to_string(),
        ty,
        storage,
        init,
    })))
}

fn parse_top_level_item(input: &str) -> IResult<&str, Option<Item>> {
    let (input, _) = ws(input)?;
    if input.is_empty() {
        return Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Eof)));
    }

    // Try each top-level construct
    if let Ok((r, item)) = parse_include(input) {
        return Ok((r, Some(item)));
    }
    // Global vars before kernels (so __constant__ is not skipped)
    if let Ok((r, item)) = parse_global_var_decl(input) {
        return Ok((r, Some(item)));
    }
    if let Ok((r, item)) = parse_kernel_def(input) {
        return Ok((r, Some(item)));
    }
    if let Ok((r, item)) = parse_device_function(input) {
        return Ok((r, Some(item)));
    }
    if let Ok((r, item)) = parse_host_function(input) {
        return Ok((r, Some(item)));
    }
    if let Ok((r, item)) = parse_typedef(input) {
        return Ok((r, Some(item)));
    }
    if let Ok((r, item)) = parse_struct(input) {
        return Ok((r, Some(item)));
    }

    // Skip preprocessor directives
    if input.starts_with('#') {
        let (r, _) = parse_preprocessor(input)?;
        return Ok((r, None));
    }

    // Skip unrecognized top-level constructs (free functions, etc.)
    // Try to skip to next semicolon or closing brace
    if let Some(pos) = input.find(|c: char| c == '{' || c == ';') {
        if input.as_bytes()[pos] == b'{' {
            let rest = skip_balanced_braces(&input[pos + 1..]);
            return Ok((rest, None));
        } else {
            return Ok((&input[pos + 1..], None));
        }
    }

    Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Eof)))
}

// ═══════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════

/// Main CUDA parser
pub struct CudaParser {
    // Parser state can be added here
}

impl CudaParser {
    /// Create a new CUDA parser
    pub fn new() -> Self {
        Self {}
    }

    /// Parse CUDA source code into AST
    pub fn parse(&self, source: &str) -> Result<Ast> {
        super::limits::validate_source(source)?;
        let mut items = Vec::new();
        let mut rest = source;

        loop {
            // Skip whitespace and comments
            match ws(rest) {
                Ok((r, _)) => rest = r,
                Err(_) => break,
            }
            if rest.is_empty() {
                break;
            }

            match parse_top_level_item(rest) {
                Ok((r, Some(item))) => {
                    items.push(item);
                    rest = r;
                }
                Ok((r, None)) => {
                    // Skipped preprocessor or unrecognized construct
                    rest = r;
                }
                Err(_) => {
                    // Skip one character and try again (error recovery)
                    if rest.is_empty() {
                        break;
                    }
                    // Try to find the next meaningful token
                    let first_len = rest.chars().next().unwrap().len_utf8();
                    if let Some(pos) = rest[first_len..].find(|c: char| {
                        c == '#' || c == '_' || c.is_alphabetic()
                    }) {
                        rest = &rest[pos + first_len..];
                    } else {
                        break;
                    }
                }
            }
        }

        Ok(Ast { items })
    }
}

impl Default for CudaParser {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_add() {
        let src = r#"
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
"#;
        let parser = CudaParser::new();
        let ast = parser.parse(src).unwrap();
        assert_eq!(ast.items.len(), 1);
        if let Item::Kernel(ref k) = ast.items[0] {
            assert_eq!(k.name, "vectorAdd");
            assert_eq!(k.params.len(), 4);
            assert_eq!(k.params[0].name, "a");
            assert_eq!(k.params[3].name, "n");
            // Body should have 2 statements: var decl + if
            assert_eq!(k.body.statements.len(), 2);
        } else {
            panic!("Expected kernel");
        }
    }

    #[test]
    fn test_mat_mul() {
        let src = r#"
__global__ void matMul(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sA[16][16];
    __shared__ float sB[16][16];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (K + 15) / 16; t++) {
        sA[threadIdx.y][threadIdx.x] = A[row * K + t * 16 + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < 16; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}
"#;
        let parser = CudaParser::new();
        let ast = parser.parse(src).unwrap();
        assert_eq!(ast.items.len(), 1);
        if let Item::Kernel(ref k) = ast.items[0] {
            assert_eq!(k.name, "matMul");
            assert_eq!(k.params.len(), 6);
        } else {
            panic!("Expected kernel");
        }
    }

    #[test]
    fn test_reduce() {
        let src = r#"
__global__ void reduce(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
"#;
        let parser = CudaParser::new();
        let ast = parser.parse(src).unwrap();
        assert_eq!(ast.items.len(), 1);
        if let Item::Kernel(ref k) = ast.items[0] {
            assert_eq!(k.name, "reduce");
            assert_eq!(k.params.len(), 3);
        } else {
            panic!("Expected kernel");
        }
    }

    #[test]
    fn test_include_directive() {
        let src = r#"#include <cuda_runtime.h>
#include "myheader.h"
"#;
        let parser = CudaParser::new();
        let ast = parser.parse(src).unwrap();
        assert!(ast.items.len() >= 2);
        assert!(matches!(&ast.items[0], Item::Include(p) if p == "cuda_runtime.h"));
        assert!(matches!(&ast.items[1], Item::Include(p) if p == "myheader.h"));
    }

    #[test]
    fn test_multiple_kernels() {
        let src = r#"
__global__ void kernel1(int* a) {
    a[threadIdx.x] = 0;
}
__global__ void kernel2(float* b, int n) {
    int i = threadIdx.x;
    if (i < n) b[i] = 1.0f;
}
"#;
        let parser = CudaParser::new();
        let ast = parser.parse(src).unwrap();
        assert_eq!(ast.items.len(), 2);
    }

    #[test]
    fn test_device_function() {
        let src = r#"
__device__ float clamp(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
"#;
        let parser = CudaParser::new();
        let ast = parser.parse(src).unwrap();
        assert_eq!(ast.items.len(), 1);
        assert!(matches!(&ast.items[0], Item::DeviceFunction(_)));
    }

    #[test]
    fn test_expressions() {
        // Test various expression types in isolation
        assert!(parse_expr("a + b * c").is_ok());
        assert!(parse_expr("a[i]").is_ok());
        assert!(parse_expr("threadIdx.x").is_ok());
        assert!(parse_expr("blockIdx.x * blockDim.x + threadIdx.x").is_ok());
        assert!(parse_expr("(float)x").is_ok());
        assert!(parse_expr("a < b && c > d").is_ok());
        assert!(parse_expr("i++").is_ok());
        assert!(parse_expr("++i").is_ok());
        assert!(parse_expr("atomicAdd(&x, 1)").is_ok());
    }

    #[test]
    fn test_type_parsing() {
        assert!(parse_type("float*").is_ok());
        assert!(parse_type("const float*").is_ok());
        assert!(parse_type("unsigned int").is_ok());
        assert!(parse_type("int").is_ok());
        assert!(parse_type("float4").is_ok());
        assert!(parse_type("double").is_ok());
    }
}
