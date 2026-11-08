//! Public component tests. These exercise shipped code rather than test-local
//! replicas of unavailable architectures, optimizers, registries or dataframes.
mod support;
mod unit {
    pub mod core_tests;
    pub mod data_tests;
    #[cfg(feature = "registry")]
    pub mod registry_tests;
    pub mod training_tests;
    #[cfg(feature = "models")]
    pub mod models {
        pub mod advanced_tests;
        pub mod basic_tests;
        pub mod recurrent_tests;
        pub mod specialized_tests;
        pub mod transformer_tests;
    }
}
