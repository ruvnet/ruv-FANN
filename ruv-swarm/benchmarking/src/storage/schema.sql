CREATE TABLE IF NOT EXISTS benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                instance_id TEXT NOT NULL,
                repository TEXT NOT NULL,
                issue_description TEXT,
                difficulty TEXT CHECK(difficulty IN ('easy', 'medium', 'hard')),
                execution_mode TEXT NOT NULL CHECK(execution_mode IN ('baseline', 'ml_optimized')),
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed', 'timeout')),
                claude_command TEXT NOT NULL,
                configuration TEXT NOT NULL,
                environment TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

CREATE TABLE IF NOT EXISTS stream_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_timestamp TIMESTAMP NOT NULL,
                relative_time_ms INTEGER NOT NULL,
                event_data TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );

CREATE TABLE IF NOT EXISTS tool_invocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                invocation_time TIMESTAMP NOT NULL,
                duration_ms INTEGER,
                parameters TEXT,
                result_size INTEGER,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                sequence_in_run INTEGER NOT NULL,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );

CREATE TABLE IF NOT EXISTS thinking_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                duration_ms INTEGER NOT NULL,
                token_count INTEGER,
                decision_points TEXT,
                context_before TEXT,
                context_after TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );

CREATE TABLE IF NOT EXISTS error_recovery_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                error_time TIMESTAMP NOT NULL,
                error_type TEXT NOT NULL,
                error_message TEXT,
                recovery_started TIMESTAMP,
                recovery_completed TIMESTAMP,
                recovery_strategy TEXT,
                recovery_success BOOLEAN,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );

CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                unit TEXT,
                timestamp TIMESTAMP NOT NULL,
                metadata TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );

CREATE TABLE IF NOT EXISTS code_quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                quality_score REAL NOT NULL,
                complexity_score INTEGER NOT NULL,
                test_coverage REAL,
                documentation_score REAL,
                security_score REAL,
                issues TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );

CREATE TABLE IF NOT EXISTS resource_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                cpu_usage REAL NOT NULL,
                memory_usage REAL NOT NULL,
                disk_io_read REAL,
                disk_io_write REAL,
                network_in REAL,
                network_out REAL,
                agent_count INTEGER,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );

CREATE TABLE IF NOT EXISTS swarm_coordination_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                active_agents INTEGER NOT NULL,
                messages_passed INTEGER NOT NULL,
                conflicts_resolved INTEGER,
                consensus_rounds INTEGER,
                coordination_efficiency REAL,
                task_distribution TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );

CREATE TABLE IF NOT EXISTS comparison_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                comparison_id TEXT UNIQUE NOT NULL,
                instance_id TEXT NOT NULL,
                baseline_run_id TEXT NOT NULL,
                ml_run_id TEXT NOT NULL,
                metric_improvements TEXT NOT NULL,
                statistical_analysis TEXT NOT NULL,
                patch_diff_analysis TEXT,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (baseline_run_id) REFERENCES benchmark_runs(run_id),
                FOREIGN KEY (ml_run_id) REFERENCES benchmark_runs(run_id)
            );

CREATE TABLE IF NOT EXISTS swe_bench_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                instance_id TEXT NOT NULL,
                tests_passed INTEGER NOT NULL,
                tests_failed INTEGER NOT NULL,
                patch_applied BOOLEAN NOT NULL,
                files_modified TEXT NOT NULL,
                patch_size_bytes INTEGER,
                validation_output TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );

CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON performance_metrics(run_id);

CREATE INDEX IF NOT EXISTS idx_metrics_type ON performance_metrics(metric_type);

CREATE INDEX IF NOT EXISTS idx_resource_run_timestamp ON resource_usage(run_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_comparison_instance ON comparison_results(instance_id);

CREATE INDEX IF NOT EXISTS idx_stream_events_run ON stream_events(run_id, sequence_number);

CREATE INDEX IF NOT EXISTS idx_tool_invocations_run ON tool_invocations(run_id, invocation_time);

CREATE INDEX IF NOT EXISTS idx_thinking_run ON thinking_sequences(run_id, start_time);

CREATE INDEX IF NOT EXISTS idx_error_recovery_run ON error_recovery_events(run_id, error_time);

CREATE INDEX IF NOT EXISTS idx_benchmark_instance ON benchmark_runs(instance_id, execution_mode);
