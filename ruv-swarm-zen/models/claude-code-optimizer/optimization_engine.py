#!/usr/bin/env python3
"""
Claude Code Optimization Engine

Implements prompt optimization for different SPARC modes and task types
with quality retention metrics.
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

@dataclass
class OptimizationResult:
    original_text: str
    optimized_text: str
    token_reduction: float
    quality_score: float
    sparc_mode: str
    task_type: str
    optimization_strategies: List[str]

class PromptOptimizer:
    """Main prompt optimization engine"""
    
    def __init__(self):
        self.load_optimization_patterns()
        self.load_sparc_templates()
    
    def load_optimization_patterns(self):
        """Load optimization patterns for different scenarios"""
        self.patterns = {
            # Common redundancy patterns
            'redundancy': [
                (r'\b(please|kindly|if you could|would you)\b', ''),
                (r'\b(make sure to|ensure that|be sure to)\b', ''),
                (r'\b(in order to|so as to)\b', 'to'),
                (r'\b(at this point in time|at the present time)\b', 'now'),
                (r'\b(due to the fact that|owing to the fact that)\b', 'because'),
                (r'\b(in the event that|in case)\b', 'if'),
                (r'\b(for the purpose of|with the aim of)\b', 'to'),
                (r'\b(prior to|previous to)\b', 'before'),
                (r'\b(subsequent to|following)\b', 'after'),
                (r'\b(in addition to)\b', 'and'),
                (r'\b(a number of|a variety of)\b', 'several'),
                (r'\b(it is important to note that)\b', ''),
                (r'\b(it should be mentioned that)\b', ''),
                (r'\b(as a matter of fact)\b', ''),
                (r'\b(the fact of the matter is)\b', ''),
            ],
            
            # Technical simplification
            'technical': [
                (r'\bimplement(ation)?\b', 'add'),
                (r'\bconfigure\b', 'set up'),
                (r'\butilize\b', 'use'),
                (r'\bestablish\b', 'create'),
                (r'\bfacilitate\b', 'enable'),
                (r'\bdemonstrate\b', 'show'),
                (r'\baccomplish\b', 'do'),
                (r'\binitialize\b', 'init'),
                (r'\bterminate\b', 'end'),
                (r'\bmodification\b', 'change'),
                (r'\boptimization\b', 'optimize'),
                (r'\bvalidation\b', 'validate'),
                (r'\bauthentication\b', 'auth'),
                (r'\bauthorization\b', 'authz'),
                (r'\bconfiguration\b', 'config'),
                (r'\bdocumentation\b', 'docs'),
            ],
            
            # Context compression
            'context': [
                (r'\bfor our (.*?) (application|system|project|platform)\b', r'for \1'),
                (r'\bin our (.*?) (application|system|project|platform)\b', r'in \1'),
                (r'\bof the (.*?) (application|system|project|platform)\b', r'of \1'),
                (r'\bthe (.*?) (application|system|project|platform)\b', r'\1'),
                (r'\bthat we are (working on|developing|building)\b', ''),
                (r'\bthat I am (working on|developing|building)\b', ''),
                (r'\bcurrently (working on|developing|building)\b', ''),
                (r'\bin the process of\b', ''),
                (r'\bin terms of\b', 'for'),
                (r'\bwith regard to\b', 'for'),
                (r'\bwith respect to\b', 'for'),
                (r'\bconcerning\b', 'for'),
                (r'\bregarding\b', 'for'),
            ]
        }
    
    def load_sparc_templates(self):
        """Load SPARC mode-specific templates"""
        self.sparc_templates = {
            'orchestrator': {
                'pattern': r'(coordinate|orchestrate|manage|organize)',
                'replacements': {
                    'coordinate multiple': 'coordinate',
                    'orchestrate the entire': 'orchestrate',
                    'manage all of the': 'manage',
                    'organize and structure': 'organize'
                },
                'keywords': ['coordinate', 'orchestrate', 'manage', 'organize', 'structure']
            },
            
            'coder': {
                'pattern': r'(implement|code|develop|build|create)',
                'replacements': {
                    'implement the functionality': 'implement',
                    'write the code for': 'code',
                    'develop the solution': 'develop',
                    'build the feature': 'build',
                    'create the implementation': 'create'
                },
                'keywords': ['implement', 'code', 'develop', 'build', 'create', 'function']
            },
            
            'researcher': {
                'pattern': r'(research|analyze|investigate|study|explore)',
                'replacements': {
                    'conduct research on': 'research',
                    'perform analysis of': 'analyze',
                    'investigate the details': 'investigate',
                    'study the behavior': 'study',
                    'explore the possibilities': 'explore'
                },
                'keywords': ['research', 'analyze', 'investigate', 'study', 'explore', 'compare']
            },
            
            'tdd': {
                'pattern': r'(test|testing|tdd|test-driven)',
                'replacements': {
                    'test-driven development': 'TDD',
                    'testing framework': 'test framework',
                    'unit testing': 'unit tests',
                    'integration testing': 'integration tests',
                    'end-to-end testing': 'e2e tests'
                },
                'keywords': ['test', 'TDD', 'unit', 'integration', 'e2e', 'mock']
            },
            
            'architect': {
                'pattern': r'(design|architecture|architect|structure|plan)',
                'replacements': {
                    'system architecture': 'architecture',
                    'architectural design': 'design',
                    'design patterns': 'patterns',
                    'structural design': 'structure'
                },
                'keywords': ['design', 'architecture', 'structure', 'pattern', 'scalable']
            },
            
            'reviewer': {
                'pattern': r'(review|audit|check|validate|verify)',
                'replacements': {
                    'code review': 'review',
                    'security audit': 'audit',
                    'quality check': 'check',
                    'validation process': 'validate'
                },
                'keywords': ['review', 'audit', 'check', 'validate', 'verify', 'quality']
            },
            
            'debugger': {
                'pattern': r'(debug|fix|troubleshoot|diagnose|resolve)',
                'replacements': {
                    'debug the issue': 'debug',
                    'fix the problem': 'fix',
                    'troubleshoot the error': 'troubleshoot',
                    'diagnose the cause': 'diagnose',
                    'resolve the bug': 'resolve'
                },
                'keywords': ['debug', 'fix', 'bug', 'error', 'issue', 'troubleshoot']
            },
            
            'tester': {
                'pattern': r'(test|testing|qa|quality)',
                'replacements': {
                    'quality assurance': 'QA',
                    'testing suite': 'test suite',
                    'test coverage': 'coverage',
                    'automated testing': 'auto tests'
                },
                'keywords': ['test', 'QA', 'coverage', 'automated', 'suite', 'scenario']
            },
            
            'analyzer': {
                'pattern': r'(analyze|analysis|performance|profile)',
                'replacements': {
                    'performance analysis': 'analysis',
                    'profiling the application': 'profiling',
                    'analyze the metrics': 'analyze metrics',
                    'bottleneck analysis': 'bottleneck check'
                },
                'keywords': ['analyze', 'performance', 'metrics', 'profile', 'bottleneck']
            }
        }
    
    def detect_sparc_mode(self, text: str) -> str:
        """Detect the most likely SPARC mode for the given text"""
        mode_scores = {}
        
        for mode, template in self.sparc_templates.items():
            score = 0
            
            # Check for pattern matches
            if re.search(template['pattern'], text, re.IGNORECASE):
                score += 3
            
            # Check for keywords
            for keyword in template['keywords']:
                if keyword.lower() in text.lower():
                    score += 1
            
            mode_scores[mode] = score
        
        # Return mode with highest score, default to 'general'
        if mode_scores:
            return max(mode_scores, key=mode_scores.get)
        return 'general'
    
    def detect_task_type(self, text: str) -> str:
        """Detect the task type from the text"""
        task_patterns = {
            'file_operations': r'(read|write|edit|create|delete|modify) (file|files)',
            'code_generation': r'(generate|create|write|implement) (code|function|class|method)',
            'debugging': r'(debug|fix|resolve|troubleshoot) (bug|error|issue)',
            'testing': r'(test|testing|unit test|integration test)',
            'documentation': r'(document|documentation|readme|docs)',
            'refactoring': r'(refactor|refactoring|optimize|improve)',
            'deployment': r'(deploy|deployment|ci/cd|pipeline)',
            'analysis': r'(analyze|analysis|performance|benchmark)'
        }
        
        for task_type, pattern in task_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return task_type
        
        return 'general'
    
    def apply_optimization_patterns(self, text: str, pattern_type: str) -> str:
        """Apply specific optimization patterns to text"""
        if pattern_type not in self.patterns:
            return text
        
        optimized_text = text
        for pattern, replacement in self.patterns[pattern_type]:
            optimized_text = re.sub(pattern, replacement, optimized_text, flags=re.IGNORECASE)
        
        return optimized_text
    
    def apply_sparc_optimizations(self, text: str, sparc_mode: str) -> str:
        """Apply SPARC mode-specific optimizations"""
        if sparc_mode not in self.sparc_templates:
            return text
        
        template = self.sparc_templates[sparc_mode]
        optimized_text = text
        
        # Apply replacements
        for phrase, replacement in template['replacements'].items():
            optimized_text = re.sub(
                re.escape(phrase), replacement, optimized_text, flags=re.IGNORECASE
            )
        
        return optimized_text
    
    def compress_lists_and_enumerations(self, text: str) -> str:
        """Compress lists and enumerations"""
        # Convert bullet points to comma-separated lists
        text = re.sub(r'(?:- |• |\d+\. )([^.\n]+)', r'\1', text)
        
        # Compress "including but not limited to"
        text = re.sub(r'including but not limited to', 'including', text, flags=re.IGNORECASE)
        
        # Compress "such as"
        text = re.sub(r'such as ([^.]+) and ([^.]+)', r'like \1, \2', text, flags=re.IGNORECASE)
        
        return text
    
    def remove_filler_words(self, text: str) -> str:
        """Remove filler words and phrases"""
        filler_patterns = [
            r'\b(obviously|clearly|definitely|certainly|of course)\b',
            r'\b(basically|essentially|fundamentally|generally)\b',
            r'\b(actually|really|quite|very|extremely|highly)\b',
            r'\b(I think|I believe|in my opinion|it seems)\b',
            r'\b(sort of|kind of|more or less|approximately)\b',
            r'\b(you know|you see|as you can see)\b',
            r'\b(let me|let us|allow me to)\b',
            r'\b(first of all|to begin with|in conclusion)\b'
        ]
        
        for pattern in filler_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def clean_whitespace(self, text: str) -> str:
        """Clean up whitespace and formatting"""
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def calculate_quality_score(self, original: str, optimized: str) -> float:
        """Calculate quality retention score"""
        # Simple heuristic based on keyword preservation
        original_words = set(re.findall(r'\b\w+\b', original.lower()))
        optimized_words = set(re.findall(r'\b\w+\b', optimized.lower()))
        
        # Important keywords that should be preserved
        important_keywords = {
            'function', 'class', 'method', 'variable', 'database', 'api', 'endpoint',
            'authentication', 'authorization', 'security', 'test', 'error', 'bug',
            'performance', 'optimization', 'react', 'node', 'python', 'javascript',
            'typescript', 'django', 'flask', 'postgresql', 'mongodb', 'redis'
        }
        
        original_important = original_words & important_keywords
        optimized_important = optimized_words & important_keywords
        
        if not original_important:
            return 0.95  # High score if no important keywords to preserve
        
        keyword_preservation = len(optimized_important) / len(original_important)
        
        # Adjust for length reduction
        length_ratio = len(optimized) / len(original)
        
        # Quality score balances keyword preservation and reasonable compression
        quality_score = (keyword_preservation * 0.7) + (min(length_ratio + 0.3, 1.0) * 0.3)
        
        return min(quality_score, 0.98)  # Cap at 98% to be realistic
    
    def optimize_prompt(self, text: str, sparc_mode: Optional[str] = None, task_type: Optional[str] = None) -> OptimizationResult:
        """Main optimization function"""
        original_text = text
        optimized_text = text
        strategies_used = []
        
        # Auto-detect SPARC mode and task type if not provided
        if not sparc_mode:
            sparc_mode = self.detect_sparc_mode(text)
        
        if not task_type:
            task_type = self.detect_task_type(text)
        
        # Apply optimization patterns in sequence
        
        # 1. Remove filler words
        optimized_text = self.remove_filler_words(optimized_text)
        strategies_used.append('filler_removal')
        
        # 2. Apply redundancy patterns
        optimized_text = self.apply_optimization_patterns(optimized_text, 'redundancy')
        strategies_used.append('redundancy_removal')
        
        # 3. Apply technical simplification
        optimized_text = self.apply_optimization_patterns(optimized_text, 'technical')
        strategies_used.append('technical_simplification')
        
        # 4. Apply context compression
        optimized_text = self.apply_optimization_patterns(optimized_text, 'context')
        strategies_used.append('context_compression')
        
        # 5. Apply SPARC-specific optimizations
        optimized_text = self.apply_sparc_optimizations(optimized_text, sparc_mode)
        strategies_used.append(f'sparc_{sparc_mode}_optimization')
        
        # 6. Compress lists and enumerations
        optimized_text = self.compress_lists_and_enumerations(optimized_text)
        strategies_used.append('list_compression')
        
        # 7. Clean whitespace
        optimized_text = self.clean_whitespace(optimized_text)
        strategies_used.append('whitespace_cleanup')
        
        # Calculate metrics
        original_tokens = len(original_text.split())
        optimized_tokens = len(optimized_text.split())
        token_reduction = 1 - (optimized_tokens / original_tokens) if original_tokens > 0 else 0
        quality_score = self.calculate_quality_score(original_text, optimized_text)
        
        return OptimizationResult(
            original_text=original_text,
            optimized_text=optimized_text,
            token_reduction=token_reduction,
            quality_score=quality_score,
            sparc_mode=sparc_mode,
            task_type=task_type,
            optimization_strategies=strategies_used
        )

class ValidationPipeline:
    """Pipeline for validating optimization results"""
    
    def __init__(self, target_reduction: float = 0.30, target_quality: float = 0.96):
        self.target_reduction = target_reduction
        self.target_quality = target_quality
        self.optimizer = PromptOptimizer()
    
    def validate_optimization(self, result: OptimizationResult) -> Dict:
        """Validate a single optimization result"""
        validation_result = {
            'meets_reduction_target': result.token_reduction >= self.target_reduction,
            'meets_quality_target': result.quality_score >= self.target_quality,
            'token_reduction': result.token_reduction,
            'quality_score': result.quality_score,
            'sparc_mode': result.sparc_mode,
            'task_type': result.task_type,
            'strategies_used': result.optimization_strategies
        }
        
        validation_result['overall_success'] = (
            validation_result['meets_reduction_target'] and 
            validation_result['meets_quality_target']
        )
        
        return validation_result
    
    def run_batch_validation(self, test_prompts: List[str]) -> Dict:
        """Run validation on a batch of test prompts"""
        results = []
        successful_optimizations = 0
        total_reduction = 0
        total_quality = 0
        
        for prompt in test_prompts:
            optimization_result = self.optimizer.optimize_prompt(prompt)
            validation_result = self.validate_optimization(optimization_result)
            
            results.append(validation_result)
            
            if validation_result['overall_success']:
                successful_optimizations += 1
            
            total_reduction += validation_result['token_reduction']
            total_quality += validation_result['quality_score']
        
        batch_metrics = {
            'total_prompts': len(test_prompts),
            'successful_optimizations': successful_optimizations,
            'success_rate': successful_optimizations / len(test_prompts) if test_prompts else 0,
            'average_token_reduction': total_reduction / len(test_prompts) if test_prompts else 0,
            'average_quality_score': total_quality / len(test_prompts) if test_prompts else 0,
            'target_reduction': self.target_reduction,
            'target_quality': self.target_quality,
            'detailed_results': results
        }
        
        return batch_metrics

def main():
    """Example usage of the optimization engine"""
    optimizer = PromptOptimizer()
    validator = ValidationPipeline()
    
    # Test prompts
    test_prompts = [
        "I need to implement a comprehensive user authentication system for our e-commerce platform that includes JWT token-based authentication with login, register, and password reset functionality.",
        "Please help me set up a complete CI/CD pipeline using GitHub Actions that automates the build, test, and deployment process for our microservices architecture.",
        "I would like you to analyze the performance bottlenecks in our React application that's experiencing slow rendering and poor user experience.",
        "Can you help me debug a memory leak in our Node.js application that's causing the server to crash after several hours of operation?"
    ]
    
    print("Claude Code Optimization Engine Test")
    print("=" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}:")
        print(f"Original: {prompt}")
        
        result = optimizer.optimize_prompt(prompt)
        
        print(f"Optimized: {result.optimized_text}")
        print(f"SPARC Mode: {result.sparc_mode}")
        print(f"Task Type: {result.task_type}")
        print(f"Token Reduction: {result.token_reduction:.1%}")
        print(f"Quality Score: {result.quality_score:.1%}")
        print(f"Strategies: {', '.join(result.optimization_strategies)}")
    
    # Run batch validation
    print(f"\n{'=' * 50}")
    print("Batch Validation Results")
    print("=" * 50)
    
    batch_results = validator.run_batch_validation(test_prompts)
    
    print(f"Total Prompts: {batch_results['total_prompts']}")
    print(f"Successful Optimizations: {batch_results['successful_optimizations']}")
    print(f"Success Rate: {batch_results['success_rate']:.1%}")
    print(f"Average Token Reduction: {batch_results['average_token_reduction']:.1%}")
    print(f"Average Quality Score: {batch_results['average_quality_score']:.1%}")
    print(f"Target Reduction: {batch_results['target_reduction']:.1%}")
    print(f"Target Quality: {batch_results['target_quality']:.1%}")

if __name__ == "__main__":
    main()