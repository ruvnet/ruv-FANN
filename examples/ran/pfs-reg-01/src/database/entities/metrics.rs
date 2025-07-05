//! Metrics entity definitions

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Eq, Serialize, Deserialize)]
#[sea_orm(table_name = "model_metrics")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: String,
    pub model_id: String,
    pub version_id: Option<String>,
    pub deployment_id: Option<String>,
    pub metric_type: String,
    pub metric_name: String,
    pub metric_value: f64,
    pub additional_data: Option<Json>,
    pub recorded_at: DateTime<Utc>,
    pub recorded_by: Option<String>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::models::Entity",
        from = "Column::ModelId",
        to = "super::models::Column::Id"
    )]
    Model,
    #[sea_orm(
        belongs_to = "super::model_versions::Entity",
        from = "Column::VersionId",
        to = "super::model_versions::Column::Id"
    )]
    ModelVersion,
    #[sea_orm(
        belongs_to = "super::deployments::Entity",
        from = "Column::DeploymentId",
        to = "super::deployments::Column::Id"
    )]
    Deployment,
}

impl Related<super::models::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Model.def()
    }
}

impl Related<super::model_versions::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::ModelVersion.def()
    }
}

impl Related<super::deployments::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Deployment.def()
    }
}

impl ActiveModelBehavior for ActiveModel {
    fn new() -> Self {
        Self {
            id: Set(uuid::Uuid::new_v4().to_string()),
            recorded_at: Set(Utc::now()),
            ..ActiveModelTrait::default()
        }
    }
}