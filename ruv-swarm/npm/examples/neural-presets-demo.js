#!/usr/bin/env node

/**
 * Neural Presets Demonstration
 * Shows how to use the 40+ production-ready neural network presets
 */

import { NeuralNetworkManager } from '../src/neural-network-manager.js';
import { WasmLoader } from '../src/wasm-loader.js';

// Create mock WASM loader for demonstration
const mockWasmLoader = {
  async loadModule() {
    return { isPlaceholder: true };
  }
};

async function demonstratePresets() {
  console.log('🧠 Neural Network Presets Demonstration\n');
  
  const neuralManager = new NeuralNetworkManager(mockWasmLoader);

  // 1. Show available presets by category
  console.log('📊 Available Preset Categories:');
  const summary = neuralManager.getPresetSummary();
  Object.entries(summary).forEach(([category, info]) => {
    console.log(`  ${category.toUpperCase()}: ${info.count} presets`);
    info.presets.slice(0, 3).forEach(preset => console.log(`    - ${preset}`));
    if (info.presets.length > 3) {
      console.log(`    ... and ${info.presets.length - 3} more`);
    }
  });
  console.log();

  // 2. Create agents from specific presets
  console.log('🚀 Creating Agents from Presets:\n');

  try {
    // NLP: Sentiment Analysis
    console.log('Creating sentiment analysis agent...');
    const sentimentAgent = await neuralManager.createAgentFromPreset(
      'sentiment-analyzer',
      'nlp',
      'sentiment_analysis_social'
    );
    console.log('✅ Sentiment Analyzer created');
    console.log(`   Performance: ${neuralManager.getPresetPerformance('nlp', 'sentiment_analysis_social').expectedAccuracy}`);
    console.log();

    // Vision: Object Detection
    console.log('Creating object detection agent...');
    const detectionAgent = await neuralManager.createAgentFromPreset(
      'object-detector',
      'vision',
      'object_detection_realtime'
    );
    console.log('✅ Object Detector created');
    console.log(`   Performance: ${neuralManager.getPresetPerformance('vision', 'object_detection_realtime').expectedAccuracy}`);
    console.log();

    // Time Series: Stock Prediction
    console.log('Creating stock prediction agent...');
    const stockAgent = await neuralManager.createAgentFromPreset(
      'stock-predictor',
      'timeseries',
      'stock_market_prediction'
    );
    console.log('✅ Stock Predictor created');
    console.log(`   Performance: ${neuralManager.getPresetPerformance('timeseries', 'stock_market_prediction').expectedAccuracy}`);
    console.log();

    // Graph: Fraud Detection
    console.log('Creating fraud detection agent...');
    const fraudAgent = await neuralManager.createAgentFromPreset(
      'fraud-detector',
      'graph',
      'fraud_detection_financial'
    );
    console.log('✅ Fraud Detector created');
    console.log(`   Performance: ${neuralManager.getPresetPerformance('graph', 'fraud_detection_financial').expectedAccuracy}`);
    console.log();

  } catch (error) {
    console.error('❌ Error creating agents:', error.message);
  }

  // 3. Create agents by use case
  console.log('🎯 Creating Agents by Use Case:\n');

  try {
    const chatbotAgent = await neuralManager.createAgentForUseCase(
      'chatbot-assistant',
      'chatbot'
    );
    console.log('✅ Chatbot Assistant created from recommended preset');
    console.log();

    const recommendationAgent = await neuralManager.createAgentForUseCase(
      'recommendation-engine',
      'recommendation'
    );
    console.log('✅ Recommendation Engine created from recommended preset');
    console.log();

  } catch (error) {
    console.error('❌ Error creating agents by use case:', error.message);
  }

  // 4. Search presets by use case
  console.log('🔍 Searching Presets by Use Case:\n');

  const weatherPresets = neuralManager.searchPresets('weather');
  console.log('Weather-related presets:');
  weatherPresets.forEach(result => {
    console.log(`  - ${result.preset.name} (${result.category})`);
    console.log(`    Use case: ${result.preset.useCase}`);
  });
  console.log();

  const medicalPresets = neuralManager.searchPresets('medical');
  console.log('Medical-related presets:');
  medicalPresets.forEach(result => {
    console.log(`  - ${result.preset.name} (${result.category})`);
    console.log(`    Use case: ${result.preset.useCase}`);
  });
  console.log();

  // 5. Batch create agents
  console.log('📦 Batch Creating Agents:\n');

  const batchConfigs = [
    { agentId: 'nlp-translator', category: 'nlp', presetName: 'language_translation' },
    { agentId: 'vision-enhancer', category: 'vision', presetName: 'image_enhancement' },
    { agentId: 'timeseries-energy', category: 'timeseries', presetName: 'energy_consumption' },
    { agentId: 'graph-social', category: 'graph', presetName: 'social_network_influence' }
  ];

  try {
    const batchResults = await neuralManager.batchCreateAgentsFromPresets(batchConfigs);
    
    console.log(`✅ Successfully created ${batchResults.results.length} agents:`);
    batchResults.results.forEach(result => {
      const presetInfo = neuralManager.getAgentPresetInfo(result.agentId);
      console.log(`  - ${result.agentId}: ${presetInfo.name}`);
    });

    if (batchResults.errors.length > 0) {
      console.log(`❌ Failed to create ${batchResults.errors.length} agents:`);
      batchResults.errors.forEach(error => {
        console.log(`  - ${error.agentId}: ${error.error}`);
      });
    }
    console.log();

  } catch (error) {
    console.error('❌ Error in batch creation:', error.message);
  }

  // 6. Show agent preset information
  console.log('📋 Agent Preset Information:\n');

  const agentIds = ['sentiment-analyzer', 'object-detector', 'stock-predictor', 'fraud-detector'];
  agentIds.forEach(agentId => {
    const presetInfo = neuralManager.getAgentPresetInfo(agentId);
    if (presetInfo) {
      console.log(`${agentId}:`);
      console.log(`  Name: ${presetInfo.name}`);
      console.log(`  Category: ${presetInfo.category}`);
      console.log(`  Expected Accuracy: ${presetInfo.performance.expectedAccuracy}`);
      console.log(`  Inference Time: ${presetInfo.performance.inferenceTime}`);
      console.log(`  Memory Usage: ${presetInfo.performance.memoryUsage}`);
      console.log(`  Use Case: ${presetInfo.useCase}`);
      console.log();
    }
  });

  // 7. Performance comparison
  console.log('⚡ Performance Comparison by Category:\n');

  const categories = ['nlp', 'vision', 'timeseries', 'graph'];
  categories.forEach(category => {
    const presets = neuralManager.getAvailablePresets(category);
    console.log(`${category.toUpperCase()} Presets (${Object.keys(presets).length} total):`);
    
    Object.entries(presets).slice(0, 3).forEach(([name, preset]) => {
      console.log(`  ${preset.name}:`);
      console.log(`    Accuracy: ${preset.performance.expectedAccuracy}`);
      console.log(`    Speed: ${preset.performance.inferenceTime}`);
      console.log(`    Memory: ${preset.performance.memoryUsage}`);
    });
    console.log();
  });

  console.log('🎉 Neural Presets Demonstration Complete!');
  console.log('\n📚 Key Features:');
  console.log('  ✓ 40+ production-ready neural network presets');
  console.log('  ✓ 4 categories: NLP, Vision, Time Series, Graph Analysis');
  console.log('  ✓ Easy creation by preset name or use case');
  console.log('  ✓ Batch agent creation capabilities');
  console.log('  ✓ Performance metrics and configuration details');
  console.log('  ✓ Search and recommendation functionality');
}

// Run demonstration if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  demonstratePresets().catch(console.error);
}

export { demonstratePresets };