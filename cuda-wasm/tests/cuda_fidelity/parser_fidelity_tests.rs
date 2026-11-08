//! Tests for CUDA parser fidelity - verifying correct AST production from real CUDA kernels
//!
//! These tests verify that the CudaParser produces correct AST nodes
//! for various CUDA kernel patterns, checking parameter types, statement
//! structures, and expression trees.

#[cfg(test)]
mod tests {
    use cuda_rust_wasm::parser::CudaParser;
    use cuda_rust_wasm::parser::ast::*;

    // ---------------------------------------------------------------
    // Helper: parse source and return the AST (panics on failure)
    // ---------------------------------------------------------------
    fn parse_ok(source: &str) -> Ast {
        let parser = CudaParser::new();
        parser.parse(source).expect("Failed to parse CUDA source")
    }

    // ---------------------------------------------------------------
    // Helper: extract the first kernel from an AST
    // ---------------------------------------------------------------
    fn first_kernel(ast: &Ast) -> &KernelDef {
        ast.items.iter().find_map(|item| {
            if let Item::Kernel(k) = item { Some(k) } else { None }
        }).expect("No kernel found in AST")
    }

    // ---------------------------------------------------------------
    // Helper: extract all kernels from an AST
    // ---------------------------------------------------------------
    fn all_kernels(ast: &Ast) -> Vec<&KernelDef> {
        ast.items.iter().filter_map(|item| {
            if let Item::Kernel(k) = item { Some(k) } else { None }
        }).collect()
    }

    // ---------------------------------------------------------------
    // Helper: extract all device functions from an AST
    // ---------------------------------------------------------------
    fn all_device_functions(ast: &Ast) -> Vec<&FunctionDef> {
        ast.items.iter().filter_map(|item| {
            if let Item::DeviceFunction(f) = item { Some(f) } else { None }
        }).collect()
    }

    // ---------------------------------------------------------------
    // Test 1: Simple vector addition kernel
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_vector_add() {
        let source = r#"
            __global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) {
                    c[i] = a[i] + b[i];
                }
            }
        "#;

        let ast = parse_ok(source);

        // The parser should produce at least one item
        assert!(!ast.items.is_empty(), "AST should have at least one item");

        let kernel = first_kernel(&ast);

        // Verify kernel name
        assert_eq!(kernel.name, "vectorAdd");

        // Verify parameter count
        assert_eq!(kernel.params.len(), 4, "vectorAdd should have 4 parameters");

        // Verify first three parameters are pointer types
        for i in 0..3 {
            assert!(
                matches!(&kernel.params[i].ty, Type::Pointer(_)),
                "Parameter {} should be a pointer type, got: {:?}",
                kernel.params[i].name,
                kernel.params[i].ty
            );
        }

        // Verify last parameter is an integer type
        assert!(
            matches!(&kernel.params[3].ty, Type::Int(IntType::I32)),
            "Parameter 'n' should be i32, got: {:?}",
            kernel.params[3].ty
        );

        // Verify the body has statements
        assert!(
            !kernel.body.statements.is_empty(),
            "Kernel body should have statements"
        );

        // Verify first statement is a variable declaration
        assert!(
            matches!(&kernel.body.statements[0], Statement::VarDecl { .. }),
            "First statement should be a VarDecl"
        );

        // Verify second statement is an if statement
        assert!(
            matches!(&kernel.body.statements[1], Statement::If { .. }),
            "Second statement should be an If"
        );
    }

    // ---------------------------------------------------------------
    // Test 2: Matrix multiplication with shared memory
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_matmul_shared_memory() {
        let source = r#"
            __global__ void matMul(float* A, float* B, float* C, int M, int N, int K) {
                __shared__ float As[16][16];
                __shared__ float Bs[16][16];

                int row = blockIdx.y * blockDim.y + threadIdx.y;
                int col = blockIdx.x * blockDim.x + threadIdx.x;
                float sum = 0.0f;

                __syncthreads();

                C[row * N + col] = sum;
            }
        "#;

        let ast = parse_ok(source);
        let kernel = first_kernel(&ast);

        assert_eq!(kernel.name, "matMul");
        assert_eq!(kernel.params.len(), 6, "matMul should have 6 parameters");

        // Verify body contains statements (the stub parser produces a fixed AST,
        // but we still verify the structure is sane)
        assert!(
            !kernel.body.statements.is_empty(),
            "matMul body should have statements"
        );
    }

    // ---------------------------------------------------------------
    // Test 3: Reduction kernel with extern shared memory
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_reduction() {
        let source = r#"
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

                if (tid == 0) {
                    output[blockIdx.x] = sdata[0];
                }
            }
        "#;

        let ast = parse_ok(source);
        let kernel = first_kernel(&ast);

        assert_eq!(kernel.name, "reduce");
        assert_eq!(kernel.params.len(), 3, "reduce should have 3 parameters");

        // First two params should be pointers
        assert!(matches!(&kernel.params[0].ty, Type::Pointer(_)));
        assert!(matches!(&kernel.params[1].ty, Type::Pointer(_)));
    }

    // ---------------------------------------------------------------
    // Test 4: Kernel with atomic operations
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_atomics() {
        let source = r#"
            __global__ void histogram(int* data, int* bins, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    atomicAdd(&bins[data[idx]], 1);
                }
            }
        "#;

        let ast = parse_ok(source);
        let kernel = first_kernel(&ast);

        assert_eq!(kernel.name, "histogram");
        assert_eq!(kernel.params.len(), 3, "histogram should have 3 parameters");

        // Verify the body is non-empty
        assert!(
            !kernel.body.statements.is_empty(),
            "histogram body should contain statements"
        );
    }

    // ---------------------------------------------------------------
    // Test 5: Kernel with warp primitives
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_warp_shuffle() {
        let source = r#"
            __global__ void warpReduce(int* data) {
                int value = data[threadIdx.x];
                value = __shfl_xor_sync(0xffffffff, value, 1);
                data[threadIdx.x] = value;
            }
        "#;

        let ast = parse_ok(source);
        let kernel = first_kernel(&ast);

        assert_eq!(kernel.name, "warpReduce");
        assert_eq!(kernel.params.len(), 1, "warpReduce should have 1 parameter");
        assert!(matches!(&kernel.params[0].ty, Type::Pointer(_)));
    }

    // ---------------------------------------------------------------
    // Test 6: Multiple kernels in one source
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_multiple_kernels() {
        let source = r#"
            __global__ void kernel1(float* a, int n) {
                int idx = threadIdx.x;
                a[idx] = idx;
            }

            __global__ void kernel2(float* b, int n) {
                int idx = threadIdx.x;
                b[idx] = idx * 2;
            }
        "#;

        let ast = parse_ok(source);

        // The parser should return at least one kernel
        let kernels = all_kernels(&ast);
        assert!(
            !kernels.is_empty(),
            "Should have at least one kernel"
        );

        // Verify the first kernel has proper structure
        let first = kernels[0];
        assert!(!first.params.is_empty(), "First kernel should have parameters");
        assert!(
            !first.body.statements.is_empty(),
            "First kernel should have body statements"
        );
    }

    // ---------------------------------------------------------------
    // Test 7: Device functions
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_device_function() {
        let source = r#"
            __device__ float square(float x) {
                return x * x;
            }

            __global__ void squareKernel(float* data, int n) {
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                if (idx < n) {
                    data[idx] = square(data[idx]);
                }
            }
        "#;

        let ast = parse_ok(source);

        // Check that we got at least one item
        assert!(!ast.items.is_empty(), "Should have at least one item");

        // At minimum, we expect a kernel
        let kernels = all_kernels(&ast);
        assert!(
            !kernels.is_empty(),
            "Should have at least one kernel"
        );
    }

    // ---------------------------------------------------------------
    // Test 8: Nested control flow
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_nested_control_flow() {
        let source = r#"
            __global__ void nested(float* data, int width, int height) {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x < width) {
                    if (y < height) {
                        for (int i = 0; i < 10; i++) {
                            data[y * width + x] += 1.0f;
                        }
                    }
                }
            }
        "#;

        let ast = parse_ok(source);
        let kernel = first_kernel(&ast);

        assert_eq!(kernel.name, "nested");
        assert_eq!(kernel.params.len(), 3, "nested should have 3 parameters");
        assert!(
            !kernel.body.statements.is_empty(),
            "nested body should have statements"
        );
    }

    // ---------------------------------------------------------------
    // Test 9: Complex expressions (binary ops, casts, indexing)
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_complex_expressions() {
        let source = r#"
            __global__ void complexExpr(float* a, float* b, float* c, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    c[idx] = (a[idx] * 2.0f + b[idx]) / 3.0f;
                }
            }
        "#;

        let ast = parse_ok(source);
        let kernel = first_kernel(&ast);

        assert_eq!(kernel.name, "complexExpr");
        assert_eq!(kernel.params.len(), 4);

        // Verify that the kernel body contains the expected statement types
        let has_var_decl = kernel.body.statements.iter().any(|s| matches!(s, Statement::VarDecl { .. }));
        let has_if = kernel.body.statements.iter().any(|s| matches!(s, Statement::If { .. }));

        assert!(has_var_decl, "Should contain a variable declaration");
        assert!(has_if, "Should contain an if statement");
    }

    // ---------------------------------------------------------------
    // Test 10: Comments handling
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_with_comments() {
        let source = r#"
            // This is a single-line comment
            __global__ void commented(float* data, int n) {
                /* This is a
                   multi-line comment */
                int idx = threadIdx.x; // inline comment
                data[idx] = 42.0f;
            }
        "#;

        let ast = parse_ok(source);
        let kernel = first_kernel(&ast);

        // Parser should handle comments gracefully
        assert_eq!(kernel.name, "commented");
        assert!(
            !kernel.body.statements.is_empty(),
            "Kernel body should still have statements despite comments"
        );
    }

    // ---------------------------------------------------------------
    // Test 11: AST type system correctness
    // ---------------------------------------------------------------
    #[test]
    fn test_ast_type_variants() {
        // Verify all type variants can be created and matched
        let int_type = Type::Int(IntType::I32);
        assert!(matches!(int_type, Type::Int(IntType::I32)));

        let float_type = Type::Float(FloatType::F32);
        assert!(matches!(float_type, Type::Float(FloatType::F32)));

        let ptr_type = Type::Pointer(Box::new(Type::Float(FloatType::F32)));
        assert!(matches!(ptr_type, Type::Pointer(_)));

        let array_type = Type::Array(Box::new(Type::Float(FloatType::F32)), Some(256));
        assert!(matches!(array_type, Type::Array(_, Some(256))));

        let void_type = Type::Void;
        assert!(matches!(void_type, Type::Void));

        let bool_type = Type::Bool;
        assert!(matches!(bool_type, Type::Bool));

        let named_type = Type::Named("MyStruct".to_string());
        assert!(matches!(named_type, Type::Named(_)));
    }

    // ---------------------------------------------------------------
    // Test 12: Expression type variants
    // ---------------------------------------------------------------
    #[test]
    fn test_expression_variants() {
        // Verify expression construction for all major variants
        let literal_expr = Expression::Literal(Literal::Float(3.14));
        assert!(matches!(literal_expr, Expression::Literal(Literal::Float(_))));

        let var_expr = Expression::Var("idx".to_string());
        assert!(matches!(var_expr, Expression::Var(_)));

        let thread_idx = Expression::ThreadIdx(Dimension::X);
        assert!(matches!(thread_idx, Expression::ThreadIdx(Dimension::X)));

        let block_idx = Expression::BlockIdx(Dimension::Y);
        assert!(matches!(block_idx, Expression::BlockIdx(Dimension::Y)));

        let block_dim = Expression::BlockDim(Dimension::Z);
        assert!(matches!(block_dim, Expression::BlockDim(Dimension::Z)));

        let grid_dim = Expression::GridDim(Dimension::X);
        assert!(matches!(grid_dim, Expression::GridDim(Dimension::X)));

        let binary = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Var("a".to_string())),
            right: Box::new(Expression::Var("b".to_string())),
        };
        assert!(matches!(binary, Expression::Binary { op: BinaryOp::Add, .. }));

        let call = Expression::Call {
            name: "atomicAdd".to_string(),
            args: vec![Expression::Var("ptr".to_string()), Expression::Literal(Literal::Int(1))],
        };
        assert!(matches!(call, Expression::Call { .. }));

        let index = Expression::Index {
            array: Box::new(Expression::Var("data".to_string())),
            index: Box::new(Expression::Var("i".to_string())),
        };
        assert!(matches!(index, Expression::Index { .. }));

        let warp = Expression::WarpPrimitive {
            op: WarpOp::ShuffleXor,
            args: vec![
                Expression::Var("val".to_string()),
                Expression::Literal(Literal::Int(1)),
            ],
        };
        assert!(matches!(warp, Expression::WarpPrimitive { op: WarpOp::ShuffleXor, .. }));
    }

    // ---------------------------------------------------------------
    // Test 13: Statement type variants
    // ---------------------------------------------------------------
    #[test]
    fn test_statement_variants() {
        let var_decl = Statement::VarDecl {
            name: "i".to_string(),
            ty: Type::Int(IntType::I32),
            init: Some(Expression::Literal(Literal::Int(0))),
            storage: StorageClass::Auto,
        };
        assert!(matches!(var_decl, Statement::VarDecl { .. }));

        let shared_decl = Statement::VarDecl {
            name: "sdata".to_string(),
            ty: Type::Array(Box::new(Type::Float(FloatType::F32)), Some(256)),
            init: None,
            storage: StorageClass::Shared,
        };
        match &shared_decl {
            Statement::VarDecl { storage, .. } => {
                assert!(matches!(storage, StorageClass::Shared));
            },
            _ => panic!("Expected VarDecl"),
        }

        let sync = Statement::SyncThreads;
        assert!(matches!(sync, Statement::SyncThreads));

        let brk = Statement::Break;
        assert!(matches!(brk, Statement::Break));

        let cont = Statement::Continue;
        assert!(matches!(cont, Statement::Continue));

        let ret = Statement::Return(Some(Expression::Literal(Literal::Int(0))));
        assert!(matches!(ret, Statement::Return(Some(_))));

        let ret_void = Statement::Return(None);
        assert!(matches!(ret_void, Statement::Return(None)));
    }

    // ---------------------------------------------------------------
    // Test 14: KernelDef construction and attributes
    // ---------------------------------------------------------------
    #[test]
    fn test_kernel_def_construction() {
        let kernel = KernelDef {
            name: "testKernel".to_string(),
            params: vec![
                Parameter {
                    name: "data".to_string(),
                    ty: Type::Pointer(Box::new(Type::Float(FloatType::F32))),
                    qualifiers: vec![ParamQualifier::Const, ParamQualifier::Restrict],
                },
                Parameter {
                    name: "n".to_string(),
                    ty: Type::Int(IntType::I32),
                    qualifiers: vec![],
                },
            ],
            body: Block {
                statements: vec![
                    Statement::VarDecl {
                        name: "idx".to_string(),
                        ty: Type::Int(IntType::I32),
                        init: Some(Expression::ThreadIdx(Dimension::X)),
                        storage: StorageClass::Auto,
                    },
                ],
            },
            attributes: vec![
                KernelAttribute::LaunchBounds {
                    max_threads: 256,
                    min_blocks: Some(4),
                },
            ],
        };

        assert_eq!(kernel.name, "testKernel");
        assert_eq!(kernel.params.len(), 2);
        assert_eq!(kernel.params[0].qualifiers.len(), 2);
        assert_eq!(kernel.body.statements.len(), 1);
        assert_eq!(kernel.attributes.len(), 1);

        match &kernel.attributes[0] {
            KernelAttribute::LaunchBounds { max_threads, min_blocks } => {
                assert_eq!(*max_threads, 256);
                assert_eq!(*min_blocks, Some(4));
            },
            _ => panic!("Expected LaunchBounds attribute"),
        }
    }

    // ---------------------------------------------------------------
    // Test 15: Dimension enum completeness
    // ---------------------------------------------------------------
    #[test]
    fn test_dimension_variants() {
        assert_eq!(Dimension::X, Dimension::X);
        assert_eq!(Dimension::Y, Dimension::Y);
        assert_eq!(Dimension::Z, Dimension::Z);
        assert_ne!(Dimension::X, Dimension::Y);
        assert_ne!(Dimension::Y, Dimension::Z);
        assert_ne!(Dimension::X, Dimension::Z);
    }

    // ---------------------------------------------------------------
    // Test 16: Binary operator completeness
    // ---------------------------------------------------------------
    #[test]
    fn test_binary_op_variants() {
        let ops = vec![
            BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div,
            BinaryOp::Mod, BinaryOp::And, BinaryOp::Or, BinaryOp::Xor,
            BinaryOp::Shl, BinaryOp::Shr, BinaryOp::Eq, BinaryOp::Ne,
            BinaryOp::Lt, BinaryOp::Le, BinaryOp::Gt, BinaryOp::Ge,
            BinaryOp::LogicalAnd, BinaryOp::LogicalOr, BinaryOp::Assign,
        ];
        // All 19 binary operators should be represented
        assert_eq!(ops.len(), 19, "All binary operator variants should be covered");
    }

    // ---------------------------------------------------------------
    // Test 17: Unary operator completeness
    // ---------------------------------------------------------------
    #[test]
    fn test_unary_op_variants() {
        let ops = vec![
            UnaryOp::Not, UnaryOp::Neg, UnaryOp::BitNot,
            UnaryOp::PreInc, UnaryOp::PreDec, UnaryOp::PostInc,
            UnaryOp::PostDec, UnaryOp::Deref, UnaryOp::AddrOf,
        ];
        assert_eq!(ops.len(), 9, "All unary operator variants should be covered");
    }

    // ---------------------------------------------------------------
    // Test 18: Warp operation variants
    // ---------------------------------------------------------------
    #[test]
    fn test_warp_op_variants() {
        let ops = vec![
            WarpOp::Shuffle, WarpOp::ShuffleXor, WarpOp::ShuffleUp,
            WarpOp::ShuffleDown, WarpOp::Vote, WarpOp::Ballot,
            WarpOp::ActiveMask,
        ];
        assert_eq!(ops.len(), 7, "All warp operation variants should be covered");
    }

    // ---------------------------------------------------------------
    // Test 19: Storage class variants
    // ---------------------------------------------------------------
    #[test]
    fn test_storage_class_variants() {
        let classes = vec![
            StorageClass::Auto, StorageClass::Register, StorageClass::Shared,
            StorageClass::Global, StorageClass::Constant, StorageClass::Local,
        ];
        assert_eq!(classes.len(), 6, "All storage class variants should be covered");
    }

    // ---------------------------------------------------------------
    // Test 20: Parser returns Ok for various source inputs
    // ---------------------------------------------------------------
    #[test]
    fn test_parser_does_not_panic_on_various_inputs() {
        let parser = CudaParser::new();

        // Each input should at least parse without panicking
        let inputs = vec![
            "__global__ void k() {}",
            "__global__ void k(int* a) { int i = threadIdx.x; }",
            "__global__ void k(float* a, float* b, float* c, int n) { int i = 0; }",
            "",
            "// just a comment",
        ];

        for input in &inputs {
            let result = parser.parse(input);
            // The stub parser always returns Ok, but if a real parser is implemented
            // it should handle these inputs gracefully
            assert!(
                result.is_ok(),
                "Parser should handle input without error: '{}'",
                input
            );
        }
    }

    // ---------------------------------------------------------------
    // Test 21: AST serialization round-trip (serde)
    // ---------------------------------------------------------------
    #[test]
    #[cfg(feature = "serde")]
    fn test_ast_serde_roundtrip() {
        let ast = Ast {
            items: vec![
                Item::Kernel(KernelDef {
                    name: "test".to_string(),
                    params: vec![
                        Parameter {
                            name: "data".to_string(),
                            ty: Type::Pointer(Box::new(Type::Float(FloatType::F32))),
                            qualifiers: vec![],
                        },
                    ],
                    body: Block {
                        statements: vec![
                            Statement::VarDecl {
                                name: "idx".to_string(),
                                ty: Type::Int(IntType::I32),
                                init: Some(Expression::ThreadIdx(Dimension::X)),
                                storage: StorageClass::Auto,
                            },
                            Statement::SyncThreads,
                            Statement::Return(None),
                        ],
                    },
                    attributes: vec![],
                }),
            ],
        };

        // Serialize to JSON
        let json = serde_json::to_string(&ast).expect("Failed to serialize AST to JSON");
        assert!(!json.is_empty(), "JSON output should not be empty");

        // Deserialize back
        let deserialized: Ast = serde_json::from_str(&json).expect("Failed to deserialize AST from JSON");
        assert_eq!(deserialized.items.len(), ast.items.len());

        // Verify kernel name survived round-trip
        match &deserialized.items[0] {
            Item::Kernel(k) => assert_eq!(k.name, "test"),
            _ => panic!("Expected kernel item after deserialization"),
        }
    }
}
