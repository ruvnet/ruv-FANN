//! Model version entity definitions

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Eq, Serialize, Deserialize)]
#[sea_orm(table_name = "model_versions")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: String,
    pub model_id: String,
    pub version_number: String,
    pub description: Option<String>,
    pub is_active: bool,
    pub status: String,
    pub artifact_path: String,
    pub artifact_format: String,
    pub artifact_checksum: String,
    pub artifact_size_bytes: i64,
    pub performance_metrics: Option<Json>,
    pub training_data_info: Option<Json>,
    pub created_at: DateTime<Utc>,
    pub created_by: String,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::models::Entity",
        from = "Column::ModelId",
        to = "super::models::Column::Id"
    )]
    Model,
    #[sea_orm(has_many = "super::deployments::Entity")]
    Deployments,
    #[sea_orm(has_many = "super::metrics::Entity")]
    Metrics,
}

impl Related<super::models::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Model.def()
    }
}

impl Related<super::deployments::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Deployments.def()
    }
}

impl Related<super::metrics::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Metrics.def()
    }
}

impl ActiveModelBehavior for ActiveModel {
    fn new() -> Self {
        Self {
            id: Set(uuid::Uuid::new_v4().to_string()),
            created_at: Set(Utc::now()),
            ..ActiveModelTrait::default()
        }
    }
}