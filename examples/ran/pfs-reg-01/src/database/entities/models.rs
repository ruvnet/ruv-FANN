//! Model entity definitions

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Eq, Serialize, Deserialize)]
#[sea_orm(table_name = "models")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub category: String,
    pub model_type: String,
    pub status: String,
    pub tags: Option<Json>,
    pub capabilities: Option<Json>,
    pub configuration: Option<Json>,
    pub metadata: Option<Json>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub created_by: String,
    pub storage_path: Option<String>,
    pub checksum: Option<String>,
    pub size_bytes: Option<i64>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(has_many = "super::model_versions::Entity")]
    ModelVersions,
    #[sea_orm(has_many = "super::deployments::Entity")]
    Deployments,
    #[sea_orm(has_many = "super::metrics::Entity")]
    Metrics,
}

impl Related<super::model_versions::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::ModelVersions.def()
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
            updated_at: Set(Utc::now()),
            ..ActiveModelTrait::default()
        }
    }

    fn before_save<C>(mut self, _db: &C, _insert: bool) -> Result<Self, DbErr>
    where
        C: ConnectionTrait,
    {
        self.updated_at = Set(Utc::now());
        Ok(self)
    }
}