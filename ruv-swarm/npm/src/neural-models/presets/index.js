/**
 * Neural Network Presets Index
 * Centralized access to all production-ready neural network configurations
 */

import { nlpPresets, getNLPPreset, availableNLPPresets } from './nlp.js';
import { visionPresets, getVisionPreset, availableVisionPresets } from './vision.js';
import { timeSeriesPresets, getTimeSeriesPreset, availableTimeSeriesPresets } from './timeseries.js';
import { graphPresets, getGraphPreset, availableGraphPresets } from './graph.js';

// Combined presets object
export const NEURAL_PRESETS = {
  nlp: nlpPresets,
  vision: visionPresets,
  timeseries: timeSeriesPresets,
  graph: graphPresets,
};

// Category-specific getters
export {
  getNLPPreset,
  getVisionPreset,
  getTimeSeriesPreset,
  getGraphPreset,
};

// Available presets lists
export {
  availableNLPPresets,
  availableVisionPresets,
  availableTimeSeriesPresets,
  availableGraphPresets,
};

// Universal preset getter function
export const getPreset = (category, presetName) => {
  const categoryMap = {
    nlp: getNLPPreset,
    vision: getVisionPreset,
    timeseries: getTimeSeriesPreset,
    graph: getGraphPreset,
  };

  if (!categoryMap[category]) {
    throw new Error(`Unknown preset category: ${category}. Available categories: ${Object.keys(categoryMap).join(', ')}`);
  }

  return categoryMap[category](presetName);
};

// Get all presets for a category
export const getCategoryPresets = (category) => {
  const categoryMap = {
    nlp: nlpPresets,
    vision: visionPresets,
    timeseries: timeSeriesPresets,
    graph: graphPresets,
  };

  if (!categoryMap[category]) {
    throw new Error(`Unknown preset category: ${category}. Available categories: ${Object.keys(categoryMap).join(', ')}`);
  }

  return categoryMap[category];
};

// Get all available preset names by category
export const getAllPresetNames = () => {
  return {
    nlp: availableNLPPresets,
    vision: availableVisionPresets,
    timeseries: availableTimeSeriesPresets,
    graph: availableGraphPresets,
  };
};

// Search presets by use case
export const searchPresetsByUseCase = (searchTerm) => {
  const results = [];
  const searchLower = searchTerm.toLowerCase();

  Object.entries(NEURAL_PRESETS).forEach(([category, presets]) => {
    Object.entries(presets).forEach(([presetName, preset]) => {
      if (
        preset.useCase.toLowerCase().includes(searchLower) ||
        preset.name.toLowerCase().includes(searchLower) ||
        preset.description.toLowerCase().includes(searchLower)
      ) {
        results.push({
          category,
          presetName,
          preset,
        });
      }
    });
  });

  return results;
};

// Search presets by accuracy range
export const searchPresetsByAccuracy = (minAccuracy) => {
  const results = [];

  Object.entries(NEURAL_PRESETS).forEach(([category, presets]) => {
    Object.entries(presets).forEach(([presetName, preset]) => {
      const accuracyStr = preset.performance.expectedAccuracy;
      const accuracyMatch = accuracyStr.match(/(\d+)-?(\d+)?%/);

      if (accuracyMatch) {
        const minAcc = parseInt(accuracyMatch[1], 10);
        if (minAcc >= minAccuracy) {
          results.push({
            category,
            presetName,
            preset,
            accuracy: minAcc,
          });
        }
      }
    });
  });

  return results.sort((a, b) => b.accuracy - a.accuracy);
};

// Search presets by inference time
export const searchPresetsByInferenceTime = (maxTimeMs) => {
  const results = [];

  Object.entries(NEURAL_PRESETS).forEach(([category, presets]) => {
    Object.entries(presets).forEach(([presetName, preset]) => {
      const timeStr = preset.performance.inferenceTime;
      const timeMatch = timeStr.match(/(\d+)ms/);

      if (timeMatch) {
        const timeMs = parseInt(timeMatch[1], 10);
        if (timeMs <= maxTimeMs) {
          results.push({
            category,
            presetName,
            preset,
            inferenceTime: timeMs,
          });
        }
      }
    });
  });

  return results.sort((a, b) => a.inferenceTime - b.inferenceTime);
};

// Get preset statistics
export const getPresetStatistics = () => {
  const stats = {
    totalPresets: 0,
    categories: {},
    models: {},
    accuracyRanges: {
      '90-100%': 0,
      '80-89%': 0,
      '70-79%': 0,
      'below-70%': 0,
    },
    inferenceTimeRanges: {
      'under-10ms': 0,
      '10-50ms': 0,
      '50-100ms': 0,
      'over-100ms': 0,
    },
  };

  Object.entries(NEURAL_PRESETS).forEach(([category, presets]) => {
    stats.categories[category] = Object.keys(presets).length;
    stats.totalPresets += Object.keys(presets).length;

    Object.values(presets).forEach(preset => {
      // Count model types
      const modelType = preset.model;
      stats.models[modelType] = (stats.models[modelType] || 0) + 1;

      // Categorize accuracy
      const accuracyStr = preset.performance.expectedAccuracy;
      const accuracyMatch = accuracyStr.match(/(\d+)-?(\d+)?%/);
      if (accuracyMatch) {
        const minAcc = parseInt(accuracyMatch[1], 10);
        if (minAcc >= 90) {
          stats.accuracyRanges['90-100%']++;
        } else if (minAcc >= 80) {
          stats.accuracyRanges['80-89%']++;
        } else if (minAcc >= 70) {
          stats.accuracyRanges['70-79%']++;
        } else {
          stats.accuracyRanges['below-70%']++;
        }
      }

      // Categorize inference time
      const timeStr = preset.performance.inferenceTime;
      const timeMatch = timeStr.match(/(\d+)ms/);
      if (timeMatch) {
        const timeMs = parseInt(timeMatch[1], 10);
        if (timeMs < 10) {
          stats.inferenceTimeRanges['under-10ms']++;
        } else if (timeMs < 50) {
          stats.inferenceTimeRanges['10-50ms']++;
        } else if (timeMs < 100) {
          stats.inferenceTimeRanges['50-100ms']++;
        } else {
          stats.inferenceTimeRanges['over-100ms']++;
        }
      }
    });
  });

  return stats;
};

// Export preset categories for easy reference
export const PRESET_CATEGORIES = {
  NLP: 'nlp',
  VISION: 'vision',
  TIME_SERIES: 'timeseries',
  GRAPH: 'graph',
};

// Export model types used in presets
export const PRESET_MODEL_TYPES = [
  'transformer',
  'cnn',
  'lstm',
  'gru',
  'autoencoder',
  'gnn',
  'resnet',
  'vae',
];

// Utility function to validate preset configuration
export const validatePresetConfig = (preset) => {
  const requiredFields = ['name', 'description', 'model', 'config', 'training', 'performance', 'useCase'];
  const missingFields = requiredFields.filter(field => !preset[field]);

  if (missingFields.length > 0) {
    throw new Error(`Preset validation failed. Missing fields: ${missingFields.join(', ')}`);
  }

  // Validate performance fields
  const requiredPerformanceFields = ['expectedAccuracy', 'inferenceTime', 'memoryUsage', 'trainingTime'];
  const missingPerfFields = requiredPerformanceFields.filter(field => !preset.performance[field]);

  if (missingPerfFields.length > 0) {
    throw new Error(`Preset performance validation failed. Missing fields: ${missingPerfFields.join(', ')}`);
  }

  return true;
};

// Export default preset recommendations by use case
export const DEFAULT_RECOMMENDATIONS = {
  'chatbot': { category: 'nlp', preset: 'conversational_ai' },
  'sentiment_analysis': { category: 'nlp', preset: 'sentiment_analysis_social' },
  'object_detection': { category: 'vision', preset: 'object_detection_realtime' },
  'face_recognition': { category: 'vision', preset: 'facial_recognition_secure' },
  'stock_prediction': { category: 'timeseries', preset: 'stock_market_prediction' },
  'weather_forecast': { category: 'timeseries', preset: 'weather_forecasting' },
  'fraud_detection': { category: 'graph', preset: 'fraud_detection_financial' },
  'recommendation': { category: 'graph', preset: 'recommendation_engine' },
};

// Get recommended preset for a use case
export const getRecommendedPreset = (useCase) => {
  const recommendation = DEFAULT_RECOMMENDATIONS[useCase.toLowerCase()];
  if (!recommendation) {
    return null;
  }

  return getPreset(recommendation.category, recommendation.preset);
};