//! Prompt generation for Claude Code CLI

use crate::loader::{DifficultyLevel, SWEBenchInstance};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Claude prompt generator for SWE-Bench instances
pub struct ClaudePromptGenerator {
    config: PromptGeneratorConfig,
    templates: HashMap<PromptTemplate, String>,
}

impl ClaudePromptGenerator {
    /// Create a new prompt generator
    pub fn new(config: crate::PromptConfig) -> Self {
        let generator_config = PromptGeneratorConfig::from(config);
        let templates = Self::initialize_templates();

        Self {
            config: generator_config,
            templates,
        }
    }

    /// Generate prompt for a SWE-Bench instance
    pub fn generate_prompt(&self, instance: &SWEBenchInstance) -> Result<GeneratedPrompt> {
        let template = self.select_template(&instance.difficulty);
        let base_prompt = self
            .templates
            .get(&template)
            .ok_or_else(|| anyhow::anyhow!("Template not found: {:?}", template))?;

        // Build context sections
        let issue_context = self.build_issue_context(instance);
        let test_context = self.build_test_context(instance);
        let hints_context = self.build_hints_context(instance);
        let constraints = self.build_constraints(instance);

        // Combine all sections
        let mut prompt_parts = vec![base_prompt.clone(), issue_context];

        if self.config.include_test_hints {
            prompt_parts.push(test_context);
        }

        if !instance.hints.is_empty() && self.config.include_hints {
            prompt_parts.push(hints_context);
        }

        prompt_parts.push(constraints);

        let full_prompt = prompt_parts.join("\n\n");
        let token_count = self.estimate_tokens(&full_prompt);

        // Truncate if necessary
        let content = if token_count > self.config.max_tokens {
            self.truncate_prompt(full_prompt, self.config.max_tokens)?
        } else {
            full_prompt
        };

        let final_token_count = self.estimate_tokens(&content);

        Ok(GeneratedPrompt {
            content,
            token_count: final_token_count,
            template_used: template,
            instance_id: instance.instance_id.clone(),
            metadata: self.build_metadata(instance),
        })
    }

    /// Generate batch prompts for multiple instances
    pub fn generate_batch(&self, instances: &[SWEBenchInstance]) -> Result<Vec<GeneratedPrompt>> {
        instances
            .iter()
            .map(|instance| self.generate_prompt(instance))
            .collect()
    }

    /// Select appropriate template based on difficulty
    fn select_template(&self, difficulty: &DifficultyLevel) -> PromptTemplate {
        match difficulty {
            DifficultyLevel::Easy => PromptTemplate::Simple,
            DifficultyLevel::Medium => PromptTemplate::Standard,
            DifficultyLevel::Hard => PromptTemplate::Detailed,
            DifficultyLevel::Expert => PromptTemplate::Expert,
        }
    }

    /// Initialize prompt templates
    fn initialize_templates() -> HashMap<PromptTemplate, String> {
        let mut templates = HashMap::new();

        templates.insert(
            PromptTemplate::Simple,
            r#"
You are working on fixing an issue in the {repo} repository.

## Issue Description
{issue_description}

## Your Task
Please analyze the issue and implement a fix. Focus on:
1. Understanding the root cause
2. Implementing a minimal, correct solution
3. Ensuring the fix doesn't break existing functionality

Use the available tools to explore the codebase, understand the issue, and implement your solution.
"#
            .to_string(),
        );

        templates.insert(
            PromptTemplate::Standard,
            r#"
You are an expert software engineer working on the {repo} repository (version {version}).

## Issue: {issue_title}

### Description
{issue_description}

### Your Task
1. **Investigate**: Use the provided tools to explore the codebase and understand the issue
2. **Analyze**: Identify the root cause and affected components
3. **Plan**: Develop a clear plan for the fix
4. **Implement**: Write the necessary code changes
5. **Verify**: Ensure your solution addresses the issue without introducing regressions

### Guidelines
- Make minimal changes necessary to fix the issue
- Follow the existing code style and conventions
- Consider edge cases and potential side effects
- Write clear, maintainable code

Begin by investigating the relevant files mentioned in the issue description.
"#
            .to_string(),
        );

        templates.insert(
            PromptTemplate::Detailed,
            r#"
You are an expert software engineer tasked with resolving a complex issue in the {repo} repository.

## Repository Context
- **Repository**: {repo}
- **Version**: {version}
- **Difficulty**: {difficulty}

## Issue Details
### Title: {issue_title}

### Full Description
{issue_description}

### Test Requirements
The following tests must pass after your implementation:
{test_directives}

### Your Mission
1. **Deep Investigation Phase**
   - Thoroughly explore the codebase structure
   - Identify all files and components related to the issue
   - Understand the existing implementation and architecture

2. **Root Cause Analysis**
   - Determine the exact cause of the issue
   - Identify why the current implementation fails
   - Consider all edge cases and scenarios

3. **Solution Design**
   - Design a robust solution that addresses the root cause
   - Ensure compatibility with existing code
   - Plan for maintainability and future extensions

4. **Implementation**
   - Implement your solution with clean, well-documented code
   - Follow the project's coding standards
   - Add appropriate error handling

5. **Verification**
   - Test your implementation thoroughly
   - Ensure all specified tests pass
   - Verify no regressions are introduced

### Important Considerations
- This is a {difficulty} level issue requiring careful attention
- Multiple files may need modification
- Consider performance implications
- Ensure thread safety if applicable

Start by examining the test patch to understand the expected behavior.
"#
            .to_string(),
        );

        templates.insert(
            PromptTemplate::Expert,
            r#"
You are a senior software architect working on a critical issue in the {repo} repository.

## Context
- **Repository**: {repo} (v{version})
- **Complexity**: Expert-level issue requiring deep system knowledge
- **Impact**: This fix may affect multiple subsystems

## The Challenge
### {issue_title}

### Detailed Problem Statement
{issue_description}

### Test Specifications
{test_patch}

### Expected Test Execution
{test_directives}

### Your Approach Should Include:

1. **Comprehensive System Analysis**
   - Map out the entire affected subsystem
   - Identify all dependencies and interactions
   - Understand the historical context (check git history if needed)

2. **Architecture-Level Thinking**
   - Consider the broader architectural implications
   - Evaluate multiple solution approaches
   - Choose the most maintainable and scalable solution

3. **Risk Assessment**
   - Identify potential breaking changes
   - Consider backward compatibility
   - Plan for migration if necessary

4. **Implementation Strategy**
   - Break down the implementation into logical steps
   - Consider creating abstractions if beneficial
   - Ensure code is self-documenting

5. **Quality Assurance**
   - Implement comprehensive error handling
   - Add logging for debugging
   - Consider adding new tests beyond requirements

6. **Performance Optimization**
   - Analyze performance implications
   - Optimize critical paths
   - Consider caching strategies if applicable

### Expert-Level Considerations
- This issue may reveal deeper architectural problems
- Your solution should be future-proof
- Consider submitting additional improvements
- Document your reasoning for future maintainers

### Available Hints (Use Judiciously)
{hints}

Remember: As an expert, you're not just fixing a bug—you're improving the system.

Begin with a thorough analysis of the codebase architecture.
"#
            .to_string(),
        );

        templates.insert(PromptTemplate::Custom("".to_string()), String::new());

        templates
    }

    /// Build issue context section
    fn build_issue_context(&self, instance: &SWEBenchInstance) -> String {
        format!(
            "## Issue Context\n\
             Repository: {}\n\
             Version: {}\n\
             Issue: {}\n\
             \n\
             Description:\n{}",
            instance.repo, instance.version, instance.issue_title, instance.issue_description
        )
    }

    /// Build test context section
    fn build_test_context(&self, instance: &SWEBenchInstance) -> String {
        let mut context = String::from("## Test Information\n");

        if !instance.test_directives.is_empty() {
            context.push_str("### Test Commands\n");
            for directive in &instance.test_directives {
                context.push_str(&format!("- `{}`\n", directive));
            }
        }

        if self.config.include_test_patch && !instance.test_patch.is_empty() {
            context.push_str("\n### Test Patch\n```diff\n");
            context.push_str(&instance.test_patch);
            context.push_str("\n```\n");
        }

        context
    }

    /// Build hints context section
    fn build_hints_context(&self, instance: &SWEBenchInstance) -> String {
        if instance.hints.is_empty() {
            return String::new();
        }

        let mut context = String::from("## Hints\n");
        for (i, hint) in instance.hints.iter().enumerate() {
            context.push_str(&format!("{}. {}\n", i + 1, hint));
        }

        context
    }

    /// Build constraints section
    fn build_constraints(&self, instance: &SWEBenchInstance) -> String {
        let mut constraints = vec![
            "## Constraints and Requirements".to_string(),
            "- Implement a working solution that makes all tests pass".to_string(),
            "- Make minimal changes necessary to fix the issue".to_string(),
            "- Do not modify test files unless explicitly required".to_string(),
            "- Follow the existing code style and conventions".to_string(),
        ];

        if instance.difficulty == DifficultyLevel::Expert {
            constraints.extend(vec![
                "- Consider performance implications of your changes".to_string(),
                "- Ensure thread safety and concurrency concerns are addressed".to_string(),
                "- Document complex logic with clear comments".to_string(),
            ]);
        }

        constraints.join("\n")
    }

    /// Estimate token count (rough approximation)
    fn estimate_tokens(&self, text: &str) -> usize {
        // Rough estimation: ~4 characters per token
        text.len() / 4
    }

    /// Truncate prompt to fit token limit
    fn truncate_prompt(&self, prompt: String, max_tokens: usize) -> Result<String> {
        let sections: Vec<&str> = prompt.split("\n\n").collect();
        let mut result = Vec::new();
        let mut current_tokens = 0;

        // Always include the first section (base template)
        if let Some(first) = sections.first() {
            result.push(*first);
            current_tokens += self.estimate_tokens(first);
        }

        // Add sections until we approach the limit
        for section in sections.iter().skip(1) {
            let section_tokens = self.estimate_tokens(section);
            if current_tokens + section_tokens > max_tokens * 9 / 10 {
                // Leave 10% buffer
                break;
            }
            result.push(*section);
            current_tokens += section_tokens;
        }

        // Add truncation notice
        if result.len() < sections.len() {
            result.push("... (Content truncated due to token limit)");
        }

        Ok(result.join("\n\n"))
    }

    /// Build metadata for the prompt
    fn build_metadata(&self, instance: &SWEBenchInstance) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("instance_id".to_string(), instance.instance_id.clone());
        metadata.insert("repo".to_string(), instance.repo.clone());
        metadata.insert("difficulty".to_string(), instance.difficulty.to_string());
        metadata.insert("version".to_string(), instance.version.clone());
        metadata
    }
}

/// Configuration for prompt generator
#[derive(Debug, Clone)]
struct PromptGeneratorConfig {
    max_tokens: usize,
    include_test_hints: bool,
    include_test_patch: bool,
    include_hints: bool,
    include_context_files: bool,
}

impl From<crate::PromptConfig> for PromptGeneratorConfig {
    fn from(config: crate::PromptConfig) -> Self {
        Self {
            max_tokens: config.max_tokens,
            include_test_hints: config.include_test_hints,
            include_test_patch: true,
            include_hints: true,
            include_context_files: config.include_context_files,
        }
    }
}

/// Prompt template types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PromptTemplate {
    Simple,
    Standard,
    Detailed,
    Expert,
    Custom(String),
}

/// Generated prompt with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedPrompt {
    pub content: String,
    pub token_count: usize,
    pub template_used: PromptTemplate,
    pub instance_id: String,
    pub metadata: HashMap<String, String>,
}

impl fmt::Display for GeneratedPrompt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.content)
    }
}

// Custom serialization for PromptTemplate
impl Serialize for PromptTemplate {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Simple => serializer.serialize_str("simple"),
            Self::Standard => serializer.serialize_str("standard"),
            Self::Detailed => serializer.serialize_str("detailed"),
            Self::Expert => serializer.serialize_str("expert"),
            Self::Custom(name) => serializer.serialize_str(&format!("custom:{}", name)),
        }
    }
}

impl<'de> Deserialize<'de> for PromptTemplate {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "simple" => Ok(Self::Simple),
            "standard" => Ok(Self::Standard),
            "detailed" => Ok(Self::Detailed),
            "expert" => Ok(Self::Expert),
            s if s.starts_with("custom:") => {
                Ok(Self::Custom(s.strip_prefix("custom:").unwrap().to_string()))
            }
            _ => Err(serde::de::Error::custom(format!("Unknown template: {}", s))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::InstanceMetrics;

    fn create_test_instance() -> SWEBenchInstance {
        SWEBenchInstance {
            instance_id: "test-001".to_string(),
            repo: "python/cpython".to_string(),
            version: "3.9".to_string(),
            issue_title: "Fix memory leak in dict".to_string(),
            issue_description: "Dict objects leak memory when...".to_string(),
            hints: vec!["Check the reference counting".to_string()],
            test_patch: "diff --git a/test.py b/test.py".to_string(),
            test_directives: vec!["pytest test_dict.py".to_string()],
            difficulty: DifficultyLevel::Medium,
            metrics: InstanceMetrics {
                files_changed: 2,
                lines_changed: 50,
                test_count: 3,
                dependencies: 1,
                complexity_score: 0.5,
                domain_specific: false,
                requires_external_api: false,
            },
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_prompt_generation() {
        let config = crate::PromptConfig::default();
        let generator = ClaudePromptGenerator::new(config);
        let instance = create_test_instance();

        let result = generator.generate_prompt(&instance);
        assert!(result.is_ok());

        let prompt = result.unwrap();
        assert!(prompt.content.contains("python/cpython"));
        assert!(prompt.content.contains("Fix memory leak in dict"));
        assert!(prompt.token_count > 0);
    }

    #[test]
    fn test_template_selection() {
        let config = crate::PromptConfig::default();
        let generator = ClaudePromptGenerator::new(config);

        assert_eq!(
            generator.select_template(&DifficultyLevel::Easy),
            PromptTemplate::Simple
        );
        assert_eq!(
            generator.select_template(&DifficultyLevel::Expert),
            PromptTemplate::Expert
        );
    }
}
