//! Database migrations for the Model Registry service

use sea_orm_migration::prelude::*;

mod m20240101_000001_create_models_table;
mod m20240101_000002_create_model_versions_table;
mod m20240101_000003_create_deployments_table;
mod m20240101_000004_create_model_metrics_table;
mod m20240101_000005_create_indexes;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20240101_000001_create_models_table::Migration),
            Box::new(m20240101_000002_create_model_versions_table::Migration),
            Box::new(m20240101_000003_create_deployments_table::Migration),
            Box::new(m20240101_000004_create_model_metrics_table::Migration),
            Box::new(m20240101_000005_create_indexes::Migration),
        ]
    }
}