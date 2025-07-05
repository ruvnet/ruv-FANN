# RAN Intelligence Platform

## Overview

The RAN Intelligence Platform is a comprehensive AI-powered system for Radio Access Network (RAN) intelligence and automation. Built on the ruv-FANN neural network library, it provides predictive optimization, proactive service assurance, and deep network intelligence capabilities.

## 🏗️ Architecture

### Service Organization

The platform is organized into four main epics, each containing specialized microservices:

#### Epic 0: Platform Foundation Services (PFS)
Core infrastructure services that other components depend on:
- **pfs-data**: Data Ingestion & Normalization Service
- **pfs-feat**: Feature Engineering Service  
- **pfs-core**: ML Core Service (ruv-FANN wrapper)
- **pfs-reg**: Model Registry & Lifecycle Service

#### Epic 1: Predictive RAN Optimization
Services for proactive network efficiency and resource utilization:
- **opt-mob**: Dynamic Mobility & Load Management
- **opt-eng**: Energy Savings
- **opt-res**: Intelligent Resource Management

#### Epic 2: Proactive Service Assurance
Services for anticipating and mitigating network issues:
- **asa-int**: Uplink Interference Management
- **asa-5g**: 5G NSA/SA Service Assurance
- **asa-qos**: Quality of Service/Experience

#### Epic 3: Deep Network Intelligence & Strategic Planning
Services for data-driven insights and strategic planning:
- **dni-clus**: Cell Behavior Clustering
- **dni-cap**: Capacity & Coverage Planning
- **dni-slice**: Network Slice Management

### Technology Stack

- **ML Engine**: ruv-FANN (Rust-based Fast Artificial Neural Network library)
- **Backend**: Rust for performance-critical services
- **Communication**: gRPC with Protocol Buffers
- **Data Format**: Apache Parquet
- **Database**: PostgreSQL for metadata storage
- **Monitoring**: Prometheus, Grafana, Jaeger
- **Containerization**: Docker with Docker Compose

## 🚀 Quick Start

### Prerequisites

- Rust 1.75+ with cargo
- Protocol Buffer compiler (protoc)
- Docker and Docker Compose (optional)
- PostgreSQL (optional, for development)

### Setup

1. **Clone and navigate to the RAN platform**:
   ```bash
   cd examples/ran
   ```

2. **Run the setup script**:
   ```bash
   ./scripts/setup.sh
   ```

3. **Build all services**:
   ```bash
   cargo build --release --all-features
   ```

4. **Run tests**:
   ```bash
   cargo test --all-features
   ```

### Development with Docker

1. **Start all services**:
   ```bash
   docker-compose up -d
   ```

2. **View service logs**:
   ```bash
   docker-compose logs -f [service-name]
   ```

3. **Stop services**:
   ```bash
   docker-compose down
   ```

## 📊 Service APIs

### Core ML Service (pfs-core)
```proto
service MLCoreService {
    rpc CreateModel(CreateModelRequest) returns (CreateModelResponse);
    rpc TrainModel(TrainModelRequest) returns (TrainModelResponse);
    rpc Predict(PredictRequest) returns (PredictResponse);
    rpc GetModel(GetModelRequest) returns (GetModelResponse);
}
```

### Data Ingestion Service (pfs-data)
```proto
service DataIngestionService {
    rpc IngestFiles(IngestFilesRequest) returns (IngestFilesResponse);
    rpc IngestStream(stream StreamDataRequest) returns (stream StreamDataResponse);
    rpc ValidateData(ValidateDataRequest) returns (ValidateDataResponse);
}
```

### Predictive Optimization Service
```proto
service PredictiveOptimizationService {
    rpc PredictHandover(PredictHandoverRequest) returns (PredictHandoverResponse);
    rpc ForecastSleepWindow(ForecastSleepWindowRequest) returns (ForecastSleepWindowResponse);
    rpc PredictScellNeed(PredictScellNeedRequest) returns (PredictScellNeedResponse);
}
```

## 🛠️ Development

### Project Structure

```
examples/ran/
├── Cargo.toml                    # Workspace configuration
├── platform-foundation/         # Epic 0: Core services
├── predictive-optimization/      # Epic 1: Optimization services
├── service-assurance/           # Epic 2: Assurance services
├── network-intelligence/        # Epic 3: Intelligence services
├── shared/
│   ├── proto/                   # gRPC Protocol Buffers
│   └── common/                  # Shared utilities
├── docker/                      # Docker configurations
├── scripts/                     # Automation scripts
├── config/                      # Configuration files
└── docs/                        # Documentation
```

### Adding a New Service

1. **Create service directory**:
   ```bash
   mkdir -p epic-name/service-name/src
   ```

2. **Add to workspace** in `Cargo.toml`:
   ```toml
   members = [
       # ... existing members
       "epic-name/service-name",
   ]
   ```

3. **Create service Cargo.toml**:
   ```toml
   [package]
   name = "service-name"
   version = "0.1.0"
   edition = "2021"

   [dependencies]
   ran-common = { path = "../../shared/common" }
   ran-proto = { path = "../../shared/proto" }
   # ... other dependencies
   ```

4. **Implement the service** following the gRPC interface definitions.

### Testing

- **Unit tests**: `cargo test`
- **Integration tests**: `cargo test --test '*'`
- **Benchmarks**: `cargo bench`
- **Format code**: `cargo fmt`
- **Lint code**: `cargo clippy`

## 📋 Service Implementation Status

### Epic 0: Platform Foundation Services
- [ ] **PFS-DATA-01**: File-based ingestion agent
- [ ] **PFS-FEAT-01**: Time-series feature generation
- [ ] **PFS-CORE-01**: ruv-FANN gRPC wrapper
- [ ] **PFS-REG-01**: Model registry implementation

### Epic 1: Predictive RAN Optimization
- [ ] **OPT-MOB-01**: Predictive Handover Trigger Model
- [ ] **OPT-ENG-01**: Cell Sleep Mode Forecaster
- [ ] **OPT-RES-01**: Predictive Carrier Aggregation SCell Manager

### Epic 2: Proactive Service Assurance
- [ ] **ASA-INT-01**: Uplink Interference Classifier
- [ ] **ASA-5G-01**: ENDC Setup Failure Predictor
- [ ] **ASA-QOS-01**: Predictive VoLTE Jitter Forecaster

### Epic 3: Deep Network Intelligence
- [ ] **DNI-CLUS-01**: Automated Cell Profiling Agent
- [ ] **DNI-CAP-01**: Capacity Cliff Forecaster
- [ ] **DNI-SLICE-01**: Network Slice SLA Breach Predictor

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUST_LOG` | Logging level | `info` |
| `GRPC_PORT` | gRPC server port | `50051` |
| `METRICS_PORT` | Metrics endpoint port | `9090` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://localhost:5432/ran_intelligence` |
| `DATA_PATH` | Data storage path | `./data` |
| `MODEL_PATH` | Model storage path | `./models` |

### Configuration Files

- `config/development.toml`: Development environment configuration
- `config/production.toml`: Production environment configuration
- `docker/docker-compose.yml`: Container orchestration
- `.github/workflows/ci.yml`: CI/CD pipeline

## 📈 Monitoring

### Metrics
- **Prometheus**: Metrics collection at `:9090`
- **Grafana**: Dashboards at `:3000` (admin/admin)

### Tracing
- **Jaeger**: Distributed tracing at `:16686`

### Health Checks
Each service provides health check endpoints:
```bash
curl http://localhost:50051/health
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** following the code style
4. **Run tests**: `cargo test --all-features`
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Code Style

- Use `cargo fmt` for formatting
- Pass `cargo clippy` linting
- Write comprehensive tests
- Document public APIs
- Follow conventional commit messages

## 📝 License

This project is licensed under the MIT OR Apache-2.0 License - see the [LICENSE](../../LICENSE) file for details.

## 🔗 Related Projects

- [ruv-FANN](../../): Fast Artificial Neural Network library
- [ruv-swarm](../../ruv-swarm/): AI agent orchestration system

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ruvnet/ruv-FANN/issues)
- **Documentation**: [API Docs](https://docs.rs/ruv-fann)
- **Repository**: [GitHub](https://github.com/ruvnet/ruv-FANN)

---

*Built with ❤️ using Rust and ruv-FANN*