//! Deployment entity definitions

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Eq, Serialize, Deserialize)]
#[sea_orm(table_name = "deployments")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: String,
    pub model_id: String,
    pub version_id: String,
    pub environment: String,
    pub status: String,
    pub config: Option<Json>,
    pub properties: Option<Json>,
    pub deployed_at: DateTime<Utc>,
    pub deployed_by: String,
    pub retired_at: Option<DateTime<Utc>>,
    pub retirement_reason: Option<String>,
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

impl ActiveModelBehavior for ActiveModel {
    fn new() -> Self {
        Self {
            id: Set(uuid::Uuid::new_v4().to_string()),
            deployed_at: Set(Utc::now()),
            ..ActiveModelTrait::default()
        }
    }
}