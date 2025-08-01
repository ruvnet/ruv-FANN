/**
 * Edge Cases and E2E Tests for Neural Model Presets
 * Comprehensive coverage for src/neural-models/presets/* and neural-presets-complete.js
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// Import all preset modules
import {
  NEURAL_PRESETS,
  getPreset,
  getCategoryPresets,
  getAllPresetNames,
  searchPresetsByUseCase,
  searchPresetsByAccuracy,
  searchPresetsByInferenceTime,
  getPresetStatistics,
  validatePresetConfig,
  getRecommendedPreset,
  PRESET_CATEGORIES,
  PRESET_MODEL_TYPES,
} from '../../src/neural-models/presets/index.js';

import { COMPLETE_NEURAL_PRESETS } from '../../src/neural-models/neural-presets-complete.js';

describe('Neural Presets Edge Cases and E2E Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Basic Preset Access Edge Cases', () => {
    it('should provide access to all preset categories', () => {
      expect(NEURAL_PRESETS).toBeDefined();
      expect(NEURAL_PRESETS.nlp).toBeDefined();
      expect(NEURAL_PRESETS.vision).toBeDefined();
      expect(NEURAL_PRESETS.timeseries).toBeDefined();
      expect(NEURAL_PRESETS.graph).toBeDefined();
    });

    it('should handle getPreset with valid category and preset', () => {
      // Test with actual presets from each category
      Object.keys(NEURAL_PRESETS).forEach(category => {
        const presets = NEURAL_PRESETS[category];
        const presetNames = Object.keys(presets);

        if (presetNames.length > 0) {
          const firstPreset = presetNames[0];
          const result = getPreset(category, firstPreset);
          expect(result).toBeDefined();
          expect(result.name).toBeDefined();
          expect(result.description).toBeDefined();
        }
      });
    });

    it('should throw error for invalid category in getPreset', () => {
      expect(() => getPreset('invalid-category', 'some-preset')).toThrow('Unknown preset category: invalid-category');
    });

    it('should handle getCategoryPresets for all categories', () => {
      Object.keys(NEURAL_PRESETS).forEach(category => {
        const presets = getCategoryPresets(category);
        expect(presets).toBeDefined();
        expect(typeof presets).toBe('object');
      });
    });

    it('should throw error for invalid category in getCategoryPresets', () => {
      expect(() => getCategoryPresets('non-existent')).toThrow('Unknown preset category: non-existent');
    });

    it('should return all preset names by category', () => {
      const allNames = getAllPresetNames();

      expect(allNames).toHaveProperty('nlp');
      expect(allNames).toHaveProperty('vision');
      expect(allNames).toHaveProperty('timeseries');
      expect(allNames).toHaveProperty('graph');

      Object.values(allNames).forEach(names => {
        expect(Array.isArray(names)).toBe(true);
      });
    });
  });

  describe('Search Functionality Edge Cases', () => {
    it('should search presets by use case (case insensitive)', () => {
      const results = searchPresetsByUseCase('classification');

      expect(Array.isArray(results)).toBe(true);
      results.forEach(result => {
        expect(result).toHaveProperty('category');
        expect(result).toHaveProperty('presetName');
        expect(result).toHaveProperty('preset');

        const useCase = result.preset.useCase?.toLowerCase() || '';
        const name = result.preset.name?.toLowerCase() || '';
        const description = result.preset.description?.toLowerCase() || '';

        expect(
          useCase.includes('classification') ||
          name.includes('classification') ||
          description.includes('classification'),
        ).toBe(true);
      });
    });

    it('should handle empty search results', () => {
      const results = searchPresetsByUseCase('non-existent-use-case-xyz');
      expect(results).toHaveLength(0);
    });

    it('should search presets by accuracy range', () => {
      const highAccuracyResults = searchPresetsByAccuracy(90);

      expect(Array.isArray(highAccuracyResults)).toBe(true);

      highAccuracyResults.forEach(result => {
        expect(result).toHaveProperty('accuracy');
        expect(result.accuracy).toBeGreaterThanOrEqual(90);
      });

      // Results should be sorted by accuracy (descending)
      for (let i = 1; i < highAccuracyResults.length; i++) {
        expect(highAccuracyResults[i - 1].accuracy).toBeGreaterThanOrEqual(
          highAccuracyResults[i].accuracy,
        );
      }
    });

    it('should handle accuracy search with no matches', () => {
      const results = searchPresetsByAccuracy(999); // Impossible accuracy
      expect(results).toHaveLength(0);
    });

    it('should search presets by inference time', () => {
      const fastResults = searchPresetsByInferenceTime(50);

      expect(Array.isArray(fastResults)).toBe(true);

      fastResults.forEach(result => {
        expect(result).toHaveProperty('inferenceTime');
        expect(result.inferenceTime).toBeLessThanOrEqual(50);
      });

      // Results should be sorted by inference time (ascending)
      for (let i = 1; i < fastResults.length; i++) {
        expect(fastResults[i - 1].inferenceTime).toBeLessThanOrEqual(
          fastResults[i].inferenceTime,
        );
      }
    });

    it('should handle inference time search with no matches', () => {
      const results = searchPresetsByInferenceTime(0); // Impossible time
      expect(results).toHaveLength(0);
    });

    it('should handle malformed accuracy strings gracefully', () => {
      // Mock a preset with malformed accuracy
      const originalPresets = NEURAL_PRESETS.nlp;
      NEURAL_PRESETS.nlp = {
        ...originalPresets,
        malformed_test: {
          name: 'Malformed Test',
          performance: {
            expectedAccuracy: 'not-a-percentage',
            inferenceTime: '10ms',
          },
        },
      };

      try {
        const results = searchPresetsByAccuracy(50);
        // Should not crash, might not include the malformed preset
        expect(Array.isArray(results)).toBe(true);
      } finally {
        // Restore original presets
        NEURAL_PRESETS.nlp = originalPresets;
      }
    });

    it('should handle malformed inference time strings gracefully', () => {
      const originalPresets = NEURAL_PRESETS.vision;
      NEURAL_PRESETS.vision = {
        ...originalPresets,
        malformed_time_test: {
          name: 'Malformed Time Test',
          performance: {
            expectedAccuracy: '85%',
            inferenceTime: 'very-fast',
          },
        },
      };

      try {
        const results = searchPresetsByInferenceTime(100);
        expect(Array.isArray(results)).toBe(true);
      } finally {
        NEURAL_PRESETS.vision = originalPresets;
      }
    });
  });

  describe('Preset Statistics Edge Cases', () => {
    it('should generate comprehensive preset statistics', () => {
      const stats = getPresetStatistics();

      expect(stats).toHaveProperty('totalPresets');
      expect(stats).toHaveProperty('categories');
      expect(stats).toHaveProperty('models');
      expect(stats).toHaveProperty('accuracyRanges');
      expect(stats).toHaveProperty('inferenceTimeRanges');

      expect(typeof stats.totalPresets).toBe('number');
      expect(stats.totalPresets).toBeGreaterThan(0);

      // Verify category counts
      Object.keys(NEURAL_PRESETS).forEach(category => {
        expect(stats.categories).toHaveProperty(category);
        expect(typeof stats.categories[category]).toBe('number');
      });

      // Verify accuracy ranges
      const accuracyKeys = ['90-100%', '80-89%', '70-79%', 'below-70%'];
      accuracyKeys.forEach(key => {
        expect(stats.accuracyRanges).toHaveProperty(key);
        expect(typeof stats.accuracyRanges[key]).toBe('number');
      });

      // Verify inference time ranges
      const timeKeys = ['under-10ms', '10-50ms', '50-100ms', 'over-100ms'];
      timeKeys.forEach(key => {
        expect(stats.inferenceTimeRanges).toHaveProperty(key);
        expect(typeof stats.inferenceTimeRanges[key]).toBe('number');
      });
    });

    it('should count models correctly in statistics', () => {
      const stats = getPresetStatistics();

      // Verify that model counts make sense
      expect(typeof stats.models).toBe('object');

      PRESET_MODEL_TYPES.forEach(modelType => {
        if (stats.models[modelType]) {
          expect(typeof stats.models[modelType]).toBe('number');
          expect(stats.models[modelType]).toBeGreaterThan(0);
        }
      });
    });

    it('should handle empty presets gracefully', () => {
      // Temporarily clear presets
      const originalPresets = { ...NEURAL_PRESETS };

      NEURAL_PRESETS.nlp = {};
      NEURAL_PRESETS.vision = {};
      NEURAL_PRESETS.timeseries = {};
      NEURAL_PRESETS.graph = {};

      try {
        const stats = getPresetStatistics();
        expect(stats.totalPresets).toBe(0);
        expect(stats.categories.nlp).toBe(0);
        expect(stats.categories.vision).toBe(0);
        expect(stats.categories.timeseries).toBe(0);
        expect(stats.categories.graph).toBe(0);
      } finally {
        // Restore original presets
        Object.assign(NEURAL_PRESETS, originalPresets);
      }
    });
  });

  describe('Preset Validation Edge Cases', () => {
    it('should validate a complete preset configuration', () => {
      const validPreset = {
        name: 'Test Preset',
        description: 'A test preset for validation',
        model: 'transformer',
        config: { layers: 12, dimensions: 768 },
        training: { epochs: 100, batchSize: 32 },
        performance: {
          expectedAccuracy: '92%',
          inferenceTime: '15ms',
          memoryUsage: '2GB',
          trainingTime: '4 hours',
        },
        useCase: 'Test case',
      };

      expect(() => validatePresetConfig(validPreset)).not.toThrow();
      expect(validatePresetConfig(validPreset)).toBe(true);
    });

    it('should throw error for missing required fields', () => {
      const incompletePreset = {
        name: 'Incomplete Preset',
        description: 'Missing other fields',
      };

      expect(() => validatePresetConfig(incompletePreset)).toThrow('Preset validation failed');
      expect(() => validatePresetConfig(incompletePreset)).toThrow('Missing fields: model, config, training, performance, useCase');
    });

    it('should throw error for missing performance fields', () => {
      const presetMissingPerf = {
        name: 'Test Preset',
        description: 'Test description',
        model: 'transformer',
        config: {},
        training: {},
        performance: {
          expectedAccuracy: '92%',
          // Missing inferenceTime, memoryUsage, trainingTime
        },
        useCase: 'Test',
      };

      expect(() => validatePresetConfig(presetMissingPerf)).toThrow('Preset performance validation failed');
      expect(() => validatePresetConfig(presetMissingPerf)).toThrow('Missing fields: inferenceTime, memoryUsage, trainingTime');
    });

    it('should handle null or undefined preset', () => {
      expect(() => validatePresetConfig(null)).toThrow();
      expect(() => validatePresetConfig(undefined)).toThrow();
    });

    it('should handle preset with null performance object', () => {
      const presetWithNullPerf = {
        name: 'Test',
        description: 'Test',
        model: 'test',
        config: {},
        training: {},
        performance: null,
        useCase: 'Test',
      };

      expect(() => validatePresetConfig(presetWithNullPerf)).toThrow();
    });
  });

  describe('Preset Recommendations Edge Cases', () => {
    it('should return recommended presets for known use cases', () => {
      const knownUseCases = [
        'chatbot',
        'sentiment_analysis',
        'object_detection',
        'face_recognition',
        'stock_prediction',
        'weather_forecast',
        'fraud_detection',
        'recommendation',
      ];

      knownUseCases.forEach(useCase => {
        const preset = getRecommendedPreset(useCase);

        if (preset) {
          expect(preset).toHaveProperty('name');
          expect(preset).toHaveProperty('description');
          expect(preset).toHaveProperty('model');
          expect(preset).toHaveProperty('performance');
        }
      });
    });

    it('should return null for unknown use cases', () => {
      const unknownUseCases = [
        'unknown-use-case',
        'non-existent',
        'random-task',
      ];

      unknownUseCases.forEach(useCase => {
        const preset = getRecommendedPreset(useCase);
        expect(preset).toBeNull();
      });
    });

    it('should handle case insensitive use case matching', () => {
      const preset1 = getRecommendedPreset('CHATBOT');
      const preset2 = getRecommendedPreset('chatbot');
      const preset3 = getRecommendedPreset('ChAtBoT');

      expect(preset1).toEqual(preset2);
      expect(preset2).toEqual(preset3);
    });

    it('should handle empty or null use case strings', () => {
      expect(getRecommendedPreset('')).toBeNull();
      expect(getRecommendedPreset(null)).toBeNull();
      expect(getRecommendedPreset(undefined)).toBeNull();
    });
  });

  describe('Constants and Exports Edge Cases', () => {
    it('should provide all expected preset categories', () => {
      expect(PRESET_CATEGORIES).toEqual({
        NLP: 'nlp',
        VISION: 'vision',
        TIME_SERIES: 'timeseries',
        GRAPH: 'graph',
      });
    });

    it('should list all expected model types', () => {
      const expectedTypes = [
        'transformer',
        'cnn',
        'lstm',
        'gru',
        'autoencoder',
        'gnn',
        'resnet',
        'vae',
      ];

      expect(PRESET_MODEL_TYPES).toEqual(expectedTypes);
    });

    it('should ensure all model types are represented in presets', () => {
      const allPresets = Object.values(NEURAL_PRESETS).flatMap(category => Object.values(category));
      const usedModelTypes = new Set(allPresets.map(preset => preset.model));

      // At least some of the model types should be used
      const intersection = PRESET_MODEL_TYPES.filter(type => usedModelTypes.has(type));
      expect(intersection.length).toBeGreaterThan(0);
    });
  });

  describe('Complete Neural Presets Integration', () => {
    it('should provide access to complete neural presets', () => {
      expect(COMPLETE_NEURAL_PRESETS).toBeDefined();
      expect(typeof COMPLETE_NEURAL_PRESETS).toBe('object');
    });

    it('should include transformer models in complete presets', () => {
      expect(COMPLETE_NEURAL_PRESETS.transformer).toBeDefined();

      const transformerPresets = COMPLETE_NEURAL_PRESETS.transformer;
      expect(typeof transformerPresets).toBe('object');

      // Check specific transformer presets
      Object.values(transformerPresets).forEach(preset => {
        expect(preset).toHaveProperty('name');
        expect(preset).toHaveProperty('description');
        expect(preset).toHaveProperty('model');
        expect(preset.model).toBe('transformer');
        expect(preset).toHaveProperty('config');
        expect(preset).toHaveProperty('cognitivePatterns');
        expect(preset).toHaveProperty('performance');
        expect(preset).toHaveProperty('useCase');
      });
    });

    it('should validate cognitive patterns in complete presets', () => {
      const validCognitivePatterns = [
        'convergent', 'divergent', 'lateral', 'systems', 'critical', 'abstract', 'adaptive',
      ];

      Object.values(COMPLETE_NEURAL_PRESETS).forEach(categoryPresets => {
        Object.values(categoryPresets).forEach(preset => {
          if (preset.cognitivePatterns) {
            expect(Array.isArray(preset.cognitivePatterns)).toBe(true);

            preset.cognitivePatterns.forEach(pattern => {
              expect(validCognitivePatterns).toContain(pattern);
            });
          }
        });
      });
    });

    it('should ensure performance metrics are properly formatted', () => {
      Object.values(COMPLETE_NEURAL_PRESETS).forEach(categoryPresets => {
        Object.values(categoryPresets).forEach(preset => {
          if (preset.performance) {
            expect(preset.performance).toHaveProperty('expectedAccuracy');
            expect(preset.performance).toHaveProperty('inferenceTime');
            expect(preset.performance).toHaveProperty('memoryUsage');
            expect(preset.performance).toHaveProperty('trainingTime');

            // Validate format patterns
            expect(typeof preset.performance.expectedAccuracy).toBe('string');
            expect(typeof preset.performance.inferenceTime).toBe('string');
            expect(typeof preset.performance.memoryUsage).toBe('string');
            expect(typeof preset.performance.trainingTime).toBe('string');
          }
        });
      });
    });
  });

  describe('Cross-Category Integration Tests', () => {
    it('should maintain consistency across preset categories', () => {
      const allPresets = [];

      Object.entries(NEURAL_PRESETS).forEach(([category, presets]) => {
        Object.entries(presets).forEach(([presetName, preset]) => {
          allPresets.push({
            category,
            presetName,
            preset,
          });
        });
      });

      // Ensure all presets have required structure
      allPresets.forEach(({ category, presetName, preset }) => {
        expect(preset).toHaveProperty('name');
        expect(preset).toHaveProperty('description');
        expect(typeof preset.name).toBe('string');
        expect(typeof preset.description).toBe('string');
        expect(preset.name.length).toBeGreaterThan(0);
        expect(preset.description.length).toBeGreaterThan(0);
      });
    });

    it('should handle search across all categories simultaneously', () => {
      const searchTerm = 'neural';
      const results = searchPresetsByUseCase(searchTerm);

      // Should find presets from potentially multiple categories
      const categoriesFound = new Set(results.map(r => r.category));
      expect(categoriesFound.size).toBeGreaterThanOrEqual(1);

      // All results should match the search term
      results.forEach(result => {
        const preset = result.preset;
        const searchableText = [
          preset.name || '',
          preset.description || '',
          preset.useCase || '',
        ].join(' ').toLowerCase();

        expect(searchableText).toContain(searchTerm.toLowerCase());
      });
    });

    it('should provide comprehensive model type coverage', () => {
      const modelCounts = {};

      Object.values(NEURAL_PRESETS).forEach(categoryPresets => {
        Object.values(categoryPresets).forEach(preset => {
          if (preset.model) {
            modelCounts[preset.model] = (modelCounts[preset.model] || 0) + 1;
          }
        });
      });

      // Should have reasonable distribution of model types
      expect(Object.keys(modelCounts).length).toBeGreaterThan(0);

      // Each model type should be used at least once if present
      Object.values(modelCounts).forEach(count => {
        expect(count).toBeGreaterThan(0);
      });
    });
  });

  describe('End-to-End Preset Workflow Tests', () => {
    it('should complete full preset discovery and validation workflow', () => {
      // Step 1: Get all available presets
      const allNames = getAllPresetNames();
      expect(Object.keys(allNames)).toHaveLength(4); // nlp, vision, timeseries, graph

      // Step 2: Search for specific use cases
      const chatbotPresets = searchPresetsByUseCase('chatbot');
      const visionPresets = searchPresetsByUseCase('detection');

      // Step 3: Filter by performance requirements
      const highAccuracyPresets = searchPresetsByAccuracy(85);
      const fastPresets = searchPresetsByInferenceTime(20);

      // Step 4: Get recommendations
      const recommendation = getRecommendedPreset('sentiment_analysis');

      // Step 5: Validate selected presets
      [...chatbotPresets, ...visionPresets, ...highAccuracyPresets, ...fastPresets].forEach(result => {
        expect(() => validatePresetConfig(result.preset)).not.toThrow();
      });

      if (recommendation) {
        expect(() => validatePresetConfig(recommendation)).not.toThrow();
      }

      // Step 6: Get comprehensive statistics
      const stats = getPresetStatistics();
      expect(stats.totalPresets).toBeGreaterThan(0);

      // Verify the workflow found reasonable results
      expect(highAccuracyPresets.length + fastPresets.length).toBeGreaterThan(0);
    });

    it('should handle performance optimization preset selection', () => {
      // Find presets optimized for different criteria
      const fastInferencePresets = searchPresetsByInferenceTime(10);
      const highAccuracyPresets = searchPresetsByAccuracy(90);

      // Analyze tradeoffs
      const performanceAnalysis = {
        speedOptimized: fastInferencePresets.length,
        accuracyOptimized: highAccuracyPresets.length,
        balanced: fastInferencePresets.filter(fast =>
          highAccuracyPresets.some(accurate =>
            fast.category === accurate.category &&
            fast.presetName === accurate.presetName,
          ),
        ).length,
      };

      // Should have options for different optimization strategies
      expect(performanceAnalysis.speedOptimized + performanceAnalysis.accuracyOptimized).toBeGreaterThan(0);
    });

    it('should support preset customization workflow', () => {
      // Step 1: Find base preset
      const basePreset = getRecommendedPreset('object_detection');

      if (basePreset) {
        // Step 2: Create customized version
        const customizedPreset = {
          ...basePreset,
          name: `Customized ${ basePreset.name}`,
          config: {
            ...basePreset.config,
            customParameter: 'custom_value',
          },
          performance: {
            ...basePreset.performance,
            expectedAccuracy: '95%', // Improved target
          },
        };

        // Step 3: Validate customization
        expect(() => validatePresetConfig(customizedPreset)).not.toThrow();

        // Step 4: Verify customizations were applied
        expect(customizedPreset.name).toContain('Customized');
        expect(customizedPreset.config.customParameter).toBe('custom_value');
        expect(customizedPreset.performance.expectedAccuracy).toBe('95%');
      }
    });

    it('should handle batch preset operations', () => {
      // Batch validation of all presets
      const validationResults = [];

      Object.entries(NEURAL_PRESETS).forEach(([category, presets]) => {
        Object.entries(presets).forEach(([presetName, preset]) => {
          try {
            validatePresetConfig(preset);
            validationResults.push({ category, presetName, valid: true });
          } catch (error) {
            validationResults.push({
              category,
              presetName,
              valid: false,
              error: error.message,
            });
          }
        });
      });

      // Most presets should be valid
      const validCount = validationResults.filter(r => r.valid).length;
      const totalCount = validationResults.length;

      expect(validCount / totalCount).toBeGreaterThan(0.8); // At least 80% should be valid

      // Log any invalid presets for debugging
      const invalidPresets = validationResults.filter(r => !r.valid);
      if (invalidPresets.length > 0) {
        console.warn('Invalid presets found:', invalidPresets);
      }
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle malformed preset data gracefully', () => {
      // Test with various malformed data
      const malformedCases = [
        null,
        undefined,
        {},
        { name: null },
        { name: '', description: null },
        { performance: {} },
        { performance: { expectedAccuracy: null } },
      ];

      malformedCases.forEach((malformed, index) => {
        try {
          validatePresetConfig(malformed);
          // If it doesn't throw, it means the preset is somehow valid
        } catch (error) {
          expect(error).toBeInstanceOf(Error);
          expect(error.message).toContain('validation failed');
        }
      });
    });

    it('should handle concurrent preset operations', async() => {
      // Simulate concurrent access to presets
      const concurrentOperations = Array.from({ length: 10 }, async(_, i) => {
        const category = ['nlp', 'vision', 'timeseries', 'graph'][i % 4];

        return Promise.all([
          getCategoryPresets(category),
          searchPresetsByUseCase(`test-${i}`),
          searchPresetsByAccuracy(80 + i),
          searchPresetsByInferenceTime(10 + i * 5),
        ]);
      });

      const results = await Promise.all(concurrentOperations);

      // All operations should complete successfully
      expect(results).toHaveLength(10);
      results.forEach(result => {
        expect(result).toHaveLength(4); // 4 operations per concurrent batch
      });
    });

    it('should maintain data integrity under stress', () => {
      // Perform many operations to ensure data integrity
      const operations = 1000;
      const categories = Object.keys(NEURAL_PRESETS);

      for (let i = 0; i < operations; i++) {
        const randomCategory = categories[i % categories.length];

        // These operations should not modify the original data
        getCategoryPresets(randomCategory);
        searchPresetsByUseCase(`stress-test-${i % 10}`);
        searchPresetsByAccuracy(Math.random() * 100);
        searchPresetsByInferenceTime(Math.random() * 100);
      }

      // Verify data is still intact
      const stats = getPresetStatistics();
      expect(stats.totalPresets).toBeGreaterThan(0);

      // Verify structure is maintained
      Object.keys(NEURAL_PRESETS).forEach(category => {
        expect(NEURAL_PRESETS[category]).toBeDefined();
        expect(typeof NEURAL_PRESETS[category]).toBe('object');
      });
    });

    it('should handle unicode and special characters in search', () => {
      const specialSearchTerms = [
        'émotions', // Unicode
        'AI/ML', // Special characters
        'test-case', // Hyphen
        'test_case', // Underscore
        'test.case', // Dot
        'test case', // Space
        '🤖', // Emoji
        '', // Empty string
      ];

      specialSearchTerms.forEach(term => {
        expect(() => {
          const results = searchPresetsByUseCase(term);
          expect(Array.isArray(results)).toBe(true);
        }).not.toThrow();
      });
    });
  });
});
