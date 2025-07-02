/**
 * Computer Vision Neural Network Presets
 * Production-ready configurations for image and video processing tasks
 */

export const visionPresets = {
  // Real-time Object Detection
  object_detection_realtime: {
    name: 'Real-time Object Detector',
    description: 'Optimized for real-time object detection in video streams',
    model: 'cnn',
    config: {
      inputShape: [416, 416, 3],
      architecture: 'yolo_v5',
      convLayers: [
        { filters: 32, kernelSize: 3, stride: 1, activation: 'mish' },
        { filters: 64, kernelSize: 3, stride: 2, activation: 'mish' },
        { filters: 128, kernelSize: 3, stride: 1, activation: 'mish' },
        { filters: 256, kernelSize: 3, stride: 2, activation: 'mish' },
      ],
      anchors: [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]],
      numClasses: 80,
      dropoutRate: 0.2,
    },
    training: {
      batchSize: 16,
      learningRate: 1e-3,
      epochs: 100,
      optimizer: 'sgd',
      momentum: 0.9,
      augmentation: {
        rotation: 15,
        zoom: 0.2,
        flip: true,
        colorJitter: 0.2,
      },
    },
    performance: {
      expectedAccuracy: '85-88% mAP',
      inferenceTime: '8ms (30+ FPS)',
      memoryUsage: '150MB',
      trainingTime: '24-48 hours on GPU',
    },
    useCase: 'Security cameras, autonomous vehicles, robotics',
  },

  // Facial Recognition
  facial_recognition_secure: {
    name: 'Secure Facial Recognition',
    description: 'High-accuracy facial recognition with privacy features',
    model: 'resnet',
    config: {
      inputShape: [160, 160, 3],
      architecture: 'facenet',
      numBlocks: 8,
      blockDepth: 3,
      hiddenDimensions: 512,
      initialChannels: 64,
      embeddingSize: 128,
      useArcFaceLoss: true,
    },
    training: {
      batchSize: 128,
      learningRate: 5e-4,
      epochs: 200,
      optimizer: 'adam',
      scheduler: 'cosine',
      margin: 0.5,
      scale: 30,
    },
    performance: {
      expectedAccuracy: '99.2% on LFW',
      inferenceTime: '5ms',
      memoryUsage: '200MB',
      trainingTime: '3-5 days on GPU',
    },
    useCase: 'Access control, identity verification, secure authentication',
  },

  // Medical Image Analysis
  medical_imaging_analysis: {
    name: 'Medical Image Analyzer',
    description: 'Analyze medical images for diagnosis support',
    model: 'cnn',
    config: {
      inputShape: [512, 512, 1], // Grayscale medical images
      architecture: 'unet_3d',
      convLayers: [
        { filters: 64, kernelSize: 3, stride: 1, activation: 'relu', batchNorm: true },
        { filters: 128, kernelSize: 3, stride: 1, activation: 'relu', batchNorm: true },
        { filters: 256, kernelSize: 3, stride: 1, activation: 'relu', batchNorm: true },
        { filters: 512, kernelSize: 3, stride: 1, activation: 'relu', batchNorm: true },
      ],
      skipConnections: true,
      attentionGates: true,
      dropoutRate: 0.3,
    },
    training: {
      batchSize: 8,
      learningRate: 1e-4,
      epochs: 150,
      optimizer: 'adamw',
      lossFunction: 'dice_bce',
      classWeights: 'auto',
      augmentation: {
        rotation: 20,
        elasticDeformation: true,
        intensityShift: 0.1,
      },
    },
    performance: {
      expectedAccuracy: '93-95% Dice Score',
      inferenceTime: '200ms',
      memoryUsage: '2GB',
      trainingTime: '48-72 hours on GPU',
    },
    useCase: 'Tumor detection, organ segmentation, disease classification',
  },

  // Autonomous Driving
  autonomous_driving: {
    name: 'Autonomous Driving Vision',
    description: 'Multi-task vision for autonomous vehicles',
    model: 'cnn',
    config: {
      inputShape: [640, 480, 3],
      architecture: 'multitask_network',
      backboneNetwork: 'efficientnet_b4',
      tasks: {
        segmentation: { numClasses: 19 },
        detection: { numClasses: 10 },
        depthEstimation: { outputChannels: 1 },
        laneDetection: { numLanes: 4 },
      },
      featurePyramid: true,
      dropoutRate: 0.2,
    },
    training: {
      batchSize: 4,
      learningRate: 2e-4,
      epochs: 80,
      optimizer: 'adam',
      multiTaskWeights: {
        segmentation: 1.0,
        detection: 1.0,
        depth: 0.5,
        lanes: 0.8,
      },
      mixedPrecision: true,
    },
    performance: {
      expectedAccuracy: '88-91% mIoU',
      inferenceTime: '25ms',
      memoryUsage: '500MB',
      trainingTime: '5-7 days on multi-GPU',
    },
    useCase: 'Self-driving cars, ADAS systems, robotics navigation',
  },

  // Quality Inspection
  quality_inspection: {
    name: 'Industrial Quality Inspector',
    description: 'Detect defects in manufacturing',
    model: 'cnn',
    config: {
      inputShape: [224, 224, 3],
      architecture: 'siamese_network',
      backbone: 'resnet50',
      metricLearning: true,
      embeddingDimension: 256,
      anomalyThreshold: 0.85,
      dropoutRate: 0.3,
    },
    training: {
      batchSize: 32,
      learningRate: 1e-3,
      epochs: 100,
      optimizer: 'adam',
      contrastiveLoss: true,
      hardNegativeMining: true,
      augmentation: {
        rotation: 360,
        brightness: 0.3,
        contrast: 0.3,
        noise: 0.05,
      },
    },
    performance: {
      expectedAccuracy: '96-98% defect detection',
      inferenceTime: '10ms',
      memoryUsage: '300MB',
      trainingTime: '12-24 hours on GPU',
    },
    useCase: 'Manufacturing QC, PCB inspection, surface defect detection',
  },

  // Satellite Image Analysis
  satellite_image_analysis: {
    name: 'Satellite Image Analyzer',
    description: 'Analyze satellite imagery for various applications',
    model: 'cnn',
    config: {
      inputShape: [512, 512, 8], // Multispectral channels
      architecture: 'deeplab_v3_plus',
      backbone: 'xception',
      outputStride: 16,
      numClasses: 15,
      asppDilationRates: [6, 12, 18],
      dropoutRate: 0.3,
    },
    training: {
      batchSize: 8,
      learningRate: 5e-4,
      epochs: 120,
      optimizer: 'sgd',
      momentum: 0.9,
      polynomialDecay: true,
      augmentation: {
        randomCrop: 448,
        horizontalFlip: true,
        verticalFlip: true,
        gaussianNoise: 0.01,
      },
    },
    performance: {
      expectedAccuracy: '89-92% pixel accuracy',
      inferenceTime: '150ms',
      memoryUsage: '1.5GB',
      trainingTime: '36-48 hours on GPU',
    },
    useCase: 'Land use classification, change detection, disaster response',
  },

  // Document Scanner
  document_scanner: {
    name: 'Document Scanner and OCR',
    description: 'Scan and extract text from documents',
    model: 'cnn',
    config: {
      inputShape: [768, 1024, 3],
      architecture: 'crnn',
      cnnBackbone: 'mobilenet_v3',
      rnnHiddenSize: 256,
      rnnLayers: 2,
      vocabSize: 95, // Printable ASCII
      ctcBeamWidth: 100,
      dropoutRate: 0.3,
    },
    training: {
      batchSize: 16,
      learningRate: 1e-3,
      epochs: 50,
      optimizer: 'adam',
      ctcLoss: true,
      augmentation: {
        perspective: true,
        rotation: 5,
        shear: 0.2,
        blur: 0.5,
      },
    },
    performance: {
      expectedAccuracy: '98-99% character accuracy',
      inferenceTime: '50ms',
      memoryUsage: '400MB',
      trainingTime: '24-36 hours on GPU',
    },
    useCase: 'Document digitization, receipt scanning, form processing',
  },

  // Video Action Recognition
  video_action_recognition: {
    name: 'Video Action Recognizer',
    description: 'Recognize human actions in video sequences',
    model: 'cnn',
    config: {
      inputShape: [16, 224, 224, 3], // 16 frames
      architecture: 'i3d',
      inflatedKernels: true,
      temporalKernelSize: 3,
      numClasses: 400,
      includeOpticalFlow: false,
      dropoutRate: 0.5,
    },
    training: {
      batchSize: 8,
      learningRate: 1e-3,
      epochs: 80,
      optimizer: 'sgd',
      momentum: 0.9,
      clipGradientNorm: 40,
      augmentation: {
        temporalJitter: 4,
        spatialCrop: 'random',
        colorJitter: 0.2,
      },
    },
    performance: {
      expectedAccuracy: '82-85% top-1',
      inferenceTime: '100ms per clip',
      memoryUsage: '800MB',
      trainingTime: '3-5 days on GPU',
    },
    useCase: 'Sports analysis, surveillance, human-computer interaction',
  },

  // Image Enhancement
  image_enhancement: {
    name: 'AI Image Enhancer',
    description: 'Enhance image quality and resolution',
    model: 'autoencoder',
    config: {
      inputSize: 65536, // 256x256
      encoderLayers: [32768, 16384, 8192, 4096],
      bottleneckSize: 2048,
      decoderLayers: [4096, 8192, 16384, 32768],
      skipConnections: true,
      residualLearning: true,
      perceptualLoss: true,
      activation: 'prelu',
    },
    training: {
      batchSize: 16,
      learningRate: 2e-4,
      epochs: 200,
      optimizer: 'adam',
      lossWeights: {
        reconstruction: 1.0,
        perceptual: 0.1,
        adversarial: 0.001,
      },
      scheduler: 'reduceLROnPlateau',
    },
    performance: {
      expectedAccuracy: '32-35 PSNR',
      inferenceTime: '80ms',
      memoryUsage: '600MB',
      trainingTime: '48-72 hours on GPU',
    },
    useCase: 'Photo restoration, super-resolution, denoising',
  },

  // Style Transfer
  style_transfer: {
    name: 'Neural Style Transfer',
    description: 'Apply artistic styles to images',
    model: 'cnn',
    config: {
      inputShape: [512, 512, 3],
      architecture: 'style_transfer_net',
      encoderBackbone: 'vgg19',
      decoderDepth: 5,
      instanceNormalization: true,
      styleEmbeddingSize: 256,
      numStyles: 10,
      dropoutRate: 0.0,
    },
    training: {
      batchSize: 8,
      learningRate: 1e-3,
      epochs: 40,
      optimizer: 'adam',
      contentWeight: 1.0,
      styleWeight: 100000,
      tvWeight: 1e-6,
      useMultipleStyleLayers: true,
    },
    performance: {
      expectedAccuracy: 'Subjective quality',
      inferenceTime: '100ms',
      memoryUsage: '500MB',
      trainingTime: '12-24 hours on GPU',
    },
    useCase: 'Artistic applications, photo filters, content creation',
  },
};

// Export utility function to get preset by name
export const getVisionPreset = (presetName) => {
  if (!visionPresets[presetName]) {
    throw new Error(`Vision preset '${presetName}' not found. Available presets: ${Object.keys(visionPresets).join(', ')}`);
  }
  return visionPresets[presetName];
};

// Export list of available presets
export const availableVisionPresets = Object.keys(visionPresets);