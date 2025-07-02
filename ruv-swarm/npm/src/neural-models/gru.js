/**
 * Gated Recurrent Unit (GRU) Model
 * Alternative to LSTM with fewer parameters
 */

import { NeuralModel } from './base.js';

class GRUModel extends NeuralModel {
  constructor(config = {}) {
    super('gru');

    // GRU configuration
    this.config = {
      inputSize: config.inputSize || 128,
      hiddenSize: config.hiddenSize || 256,
      numLayers: config.numLayers || 2,
      outputSize: config.outputSize || 10,
      dropoutRate: config.dropoutRate || 0.2,
      bidirectional: config.bidirectional || false,
      ...config,
    };

    // Initialize GRU gates and weights
    this.gates = [];
    this.outputLayer = null;

    this.initializeWeights();
  }

  initializeWeights() {
    const directions = this.config.bidirectional ? 2 : 1;

    // Initialize weights for each layer and direction
    for (let layer = 0; layer < this.config.numLayers; layer++) {
      const layerGates = [];

      for (let dir = 0; dir < directions; dir++) {
        const inputSize = layer === 0 ? this.config.inputSize :
          this.config.hiddenSize * directions;

        // GRU has 3 gates: reset, update, and candidate
        const gates = {
          // Reset gate
          resetInput: this.createWeight([inputSize, this.config.hiddenSize]),
          resetHidden: this.createWeight([this.config.hiddenSize, this.config.hiddenSize]),
          resetBias: new Float32Array(this.config.hiddenSize).fill(0),

          // Update gate
          updateInput: this.createWeight([inputSize, this.config.hiddenSize]),
          updateHidden: this.createWeight([this.config.hiddenSize, this.config.hiddenSize]),
          updateBias: new Float32Array(this.config.hiddenSize).fill(0),

          // Candidate hidden state
          candidateInput: this.createWeight([inputSize, this.config.hiddenSize]),
          candidateHidden: this.createWeight([this.config.hiddenSize, this.config.hiddenSize]),
          candidateBias: new Float32Array(this.config.hiddenSize).fill(0),

          direction: dir === 0 ? 'forward' : 'backward',
        };

        layerGates.push(gates);
      }

      this.gates.push(layerGates);
    }

    // Output layer
    const outputInputSize = this.config.hiddenSize * directions;
    this.outputLayer = {
      weight: this.createWeight([outputInputSize, this.config.outputSize]),
      bias: new Float32Array(this.config.outputSize).fill(0),
    };
  }

  createWeight(shape) {
    const size = shape.reduce((a, b) => a * b, 1);
    const weight = new Float32Array(size);

    // Xavier initialization
    const scale = Math.sqrt(2.0 / (shape[0] + shape[1]));

    for (let i = 0; i < size; i++) {
      weight[i] = (Math.random() * 2 - 1) * scale;
    }

    weight.shape = shape;
    return weight;
  }

  async forward(input, training = false) {
    const batchSize = input.shape[0];
    const sequenceLength = input.shape[1];

    // Initialize hidden states for all layers
    const hiddenStates = this.initializeHiddenStates(batchSize);

    // Process through GRU layers
    let layerInput = input;

    for (let layer = 0; layer < this.config.numLayers; layer++) {
      const layerOutput = await this.processLayer(
        layerInput,
        hiddenStates[layer],
        layer,
        training,
      );

      layerInput = layerOutput.output;
      hiddenStates[layer] = layerOutput.finalHidden;
    }

    // Apply output layer to final hidden states
    const output = this.applyOutputLayer(layerInput);

    return output;
  }

  initializeHiddenStates(batchSize) {
    const hiddenStates = [];
    const directions = this.config.bidirectional ? 2 : 1;

    for (let layer = 0; layer < this.config.numLayers; layer++) {
      const layerHidden = [];

      for (let dir = 0; dir < directions; dir++) {
        const hidden = new Float32Array(batchSize * this.config.hiddenSize);
        hidden.shape = [batchSize, this.config.hiddenSize];
        layerHidden.push(hidden);
      }

      hiddenStates.push(layerHidden);
    }

    return hiddenStates;
  }

  async processLayer(input, hiddenStates, layerIndex, training) {
    const batchSize = input.shape[0];
    const sequenceLength = input.shape[1];
    const inputSize = input.shape[2];

    const directions = this.config.bidirectional ? 2 : 1;
    const outputs = [];

    for (let dir = 0; dir < directions; dir++) {
      const gates = this.gates[layerIndex][dir];
      const isBackward = dir === 1;

      // Process sequence in appropriate direction
      const sequenceOutput = new Float32Array(
        batchSize * sequenceLength * this.config.hiddenSize,
      );

      let hidden = hiddenStates[dir];

      for (let t = 0; t < sequenceLength; t++) {
        const timeStep = isBackward ? sequenceLength - 1 - t : t;

        // Extract input at current time step
        const xt = new Float32Array(batchSize * inputSize);
        for (let b = 0; b < batchSize; b++) {
          for (let i = 0; i < inputSize; i++) {
            xt[b * inputSize + i] = input[b * sequenceLength * inputSize +
                                         timeStep * inputSize + i];
          }
        }
        xt.shape = [batchSize, inputSize];

        // GRU computation
        const gruOutput = this.gruCell(xt, hidden, gates);
        hidden = gruOutput;

        // Store output
        for (let b = 0; b < batchSize; b++) {
          for (let h = 0; h < this.config.hiddenSize; h++) {
            sequenceOutput[b * sequenceLength * this.config.hiddenSize +
                         timeStep * this.config.hiddenSize + h] =
              hidden[b * this.config.hiddenSize + h];
          }
        }
      }

      sequenceOutput.shape = [batchSize, sequenceLength, this.config.hiddenSize];
      outputs.push(sequenceOutput);
      hiddenStates[dir] = hidden;
    }

    // Concatenate outputs if bidirectional
    let finalOutput;
    if (this.config.bidirectional) {
      finalOutput = this.concatenateBidirectional(outputs[0], outputs[1]);
    } else {
      finalOutput = outputs[0];
    }

    // Apply dropout if training
    if (training && this.config.dropoutRate > 0 && layerIndex < this.config.numLayers - 1) {
      finalOutput = this.dropout(finalOutput, this.config.dropoutRate);
    }

    return {
      output: finalOutput,
      finalHidden: hiddenStates,
    };
  }

  gruCell(input, hidden, gates) {
    const batchSize = input.shape[0];
    const inputSize = input.shape[1];
    const { hiddenSize } = this.config;

    // Reset gate: r = σ(W_ir @ x + W_hr @ h + b_r)
    const resetGate = new Float32Array(batchSize * hiddenSize);
    for (let b = 0; b < batchSize; b++) {
      for (let h = 0; h < hiddenSize; h++) {
        let sum = gates.resetBias[h];

        // Input contribution
        for (let i = 0; i < inputSize; i++) {
          sum += input[b * inputSize + i] *
                 gates.resetInput[i * hiddenSize + h];
        }

        // Hidden contribution
        for (let hh = 0; hh < hiddenSize; hh++) {
          sum += hidden[b * hiddenSize + hh] *
                 gates.resetHidden[hh * hiddenSize + h];
        }

        resetGate[b * hiddenSize + h] = 1 / (1 + Math.exp(-sum)); // sigmoid
      }
    }

    // Update gate: z = σ(W_iz @ x + W_hz @ h + b_z)
    const updateGate = new Float32Array(batchSize * hiddenSize);
    for (let b = 0; b < batchSize; b++) {
      for (let h = 0; h < hiddenSize; h++) {
        let sum = gates.updateBias[h];

        // Input contribution
        for (let i = 0; i < inputSize; i++) {
          sum += input[b * inputSize + i] *
                 gates.updateInput[i * hiddenSize + h];
        }

        // Hidden contribution
        for (let hh = 0; hh < hiddenSize; hh++) {
          sum += hidden[b * hiddenSize + hh] *
                 gates.updateHidden[hh * hiddenSize + h];
        }

        updateGate[b * hiddenSize + h] = 1 / (1 + Math.exp(-sum)); // sigmoid
      }
    }

    // Candidate hidden state: h_tilde = tanh(W_ih @ x + W_hh @ (r * h) + b_h)
    const candidateHidden = new Float32Array(batchSize * hiddenSize);
    for (let b = 0; b < batchSize; b++) {
      for (let h = 0; h < hiddenSize; h++) {
        let sum = gates.candidateBias[h];

        // Input contribution
        for (let i = 0; i < inputSize; i++) {
          sum += input[b * inputSize + i] *
                 gates.candidateInput[i * hiddenSize + h];
        }

        // Hidden contribution (modulated by reset gate)
        for (let hh = 0; hh < hiddenSize; hh++) {
          const modulatedHidden = resetGate[b * hiddenSize + hh] *
                                 hidden[b * hiddenSize + hh];
          sum += modulatedHidden * gates.candidateHidden[hh * hiddenSize + h];
        }

        candidateHidden[b * hiddenSize + h] = Math.tanh(sum);
      }
    }

    // New hidden state: h_t = z * h_{t-1} + (1 - z) * h_tilde
    const newHidden = new Float32Array(batchSize * hiddenSize);
    for (let b = 0; b < batchSize; b++) {
      for (let h = 0; h < hiddenSize; h++) {
        const idx = b * hiddenSize + h;
        const z = updateGate[idx];
        newHidden[idx] = z * hidden[idx] + (1 - z) * candidateHidden[idx];
      }
    }

    newHidden.shape = [batchSize, hiddenSize];
    return newHidden;
  }

  concatenateBidirectional(forward, backward) {
    const [batchSize, sequenceLength, hiddenSize] = forward.shape;
    const output = new Float32Array(batchSize * sequenceLength * hiddenSize * 2);

    for (let b = 0; b < batchSize; b++) {
      for (let t = 0; t < sequenceLength; t++) {
        // Copy forward direction
        for (let h = 0; h < hiddenSize; h++) {
          output[b * sequenceLength * hiddenSize * 2 +
                t * hiddenSize * 2 + h] =
            forward[b * sequenceLength * hiddenSize +
                   t * hiddenSize + h];
        }

        // Copy backward direction
        for (let h = 0; h < hiddenSize; h++) {
          output[b * sequenceLength * hiddenSize * 2 +
                t * hiddenSize * 2 + hiddenSize + h] =
            backward[b * sequenceLength * hiddenSize +
                    t * hiddenSize + h];
        }
      }
    }

    output.shape = [batchSize, sequenceLength, hiddenSize * 2];
    return output;
  }

  applyOutputLayer(input) {
    const [batchSize, sequenceLength, hiddenSize] = input.shape;

    // Apply output layer to last time step
    const lastTimeStep = new Float32Array(batchSize * hiddenSize);

    for (let b = 0; b < batchSize; b++) {
      for (let h = 0; h < hiddenSize; h++) {
        lastTimeStep[b * hiddenSize + h] =
          input[b * sequenceLength * hiddenSize +
               (sequenceLength - 1) * hiddenSize + h];
      }
    }

    lastTimeStep.shape = [batchSize, hiddenSize];

    // Linear transformation
    const output = new Float32Array(batchSize * this.config.outputSize);

    for (let b = 0; b < batchSize; b++) {
      for (let o = 0; o < this.config.outputSize; o++) {
        let sum = this.outputLayer.bias[o];

        for (let h = 0; h < hiddenSize; h++) {
          sum += lastTimeStep[b * hiddenSize + h] *
                 this.outputLayer.weight[h * this.config.outputSize + o];
        }

        output[b * this.config.outputSize + o] = sum;
      }
    }

    output.shape = [batchSize, this.config.outputSize];
    return output;
  }

  async train(trainingData, options = {}) {
    const {
      epochs = 10,
      batchSize = 32,
      learningRate = 0.001,
      gradientClipping = 5.0,
      validationSplit = 0.1,
    } = options;

    const trainingHistory = [];

    // Split data
    const splitIndex = Math.floor(trainingData.length * (1 - validationSplit));
    const trainData = trainingData.slice(0, splitIndex);
    const valData = trainingData.slice(splitIndex);

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      let epochAccuracy = 0;
      let batchCount = 0;

      // Shuffle training data
      const shuffled = this.shuffle(trainData);

      // Process batches
      for (let i = 0; i < shuffled.length; i += batchSize) {
        const batch = shuffled.slice(i, Math.min(i + batchSize, shuffled.length));

        // Forward pass
        const predictions = await this.forward(batch.inputs, true);

        // Calculate loss
        const loss = this.crossEntropyLoss(predictions, batch.targets);
        epochLoss += loss;

        // Calculate accuracy for classification
        if (this.config.outputSize > 1) {
          const accuracy = this.calculateAccuracy(predictions, batch.targets);
          epochAccuracy += accuracy;
        }

        // Backward pass with gradient clipping
        await this.backward(loss, learningRate, gradientClipping);

        batchCount++;
      }

      // Validation
      const valMetrics = await this.evaluate(valData);

      const avgTrainLoss = epochLoss / batchCount;
      const avgTrainAccuracy = epochAccuracy / batchCount;

      const historyEntry = {
        epoch: epoch + 1,
        trainLoss: avgTrainLoss,
        valLoss: valMetrics.loss,
      };

      if (this.config.outputSize > 1) {
        historyEntry.trainAccuracy = avgTrainAccuracy;
        historyEntry.valAccuracy = valMetrics.accuracy;
      }

      trainingHistory.push(historyEntry);

      console.log(
        `Epoch ${epoch + 1}/${epochs} - ` +
        `Train Loss: ${avgTrainLoss.toFixed(4)}, ${
          this.config.outputSize > 1 ?
            `Train Acc: ${(avgTrainAccuracy * 100).toFixed(2)}%, ` : ''
        }Val Loss: ${valMetrics.loss.toFixed(4)}${
          this.config.outputSize > 1 ?
            `, Val Acc: ${(valMetrics.accuracy * 100).toFixed(2)}%` : ''}`,
      );

      this.updateMetrics(avgTrainLoss, avgTrainAccuracy);
    }

    return {
      history: trainingHistory,
      finalLoss: trainingHistory[trainingHistory.length - 1].trainLoss,
      modelType: 'gru',
    };
  }

  async evaluate(data) {
    let totalLoss = 0;
    let totalAccuracy = 0;
    let batchCount = 0;

    for (const batch of data) {
      const predictions = await this.forward(batch.inputs, false);
      const loss = this.crossEntropyLoss(predictions, batch.targets);

      totalLoss += loss;

      if (this.config.outputSize > 1) {
        const accuracy = this.calculateAccuracy(predictions, batch.targets);
        totalAccuracy += accuracy;
      }

      batchCount++;
    }

    const metrics = {
      loss: totalLoss / batchCount,
    };

    if (this.config.outputSize > 1) {
      metrics.accuracy = totalAccuracy / batchCount;
    }

    return metrics;
  }

  calculateAccuracy(predictions, targets) {
    const batchSize = predictions.shape[0];
    let correct = 0;

    for (let b = 0; b < batchSize; b++) {
      let maxIdx = 0;
      let maxVal = -Infinity;

      for (let i = 0; i < this.config.outputSize; i++) {
        const val = predictions[b * this.config.outputSize + i];
        if (val > maxVal) {
          maxVal = val;
          maxIdx = i;
        }
      }

      if (targets[b * this.config.outputSize + maxIdx] === 1) {
        correct++;
      }
    }

    return correct / batchSize;
  }

  getConfig() {
    return {
      type: 'gru',
      ...this.config,
      parameters: this.countParameters(),
    };
  }

  countParameters() {
    let count = 0;

    // GRU gates parameters
    for (const layer of this.gates) {
      for (const gates of layer) {
        // Reset gate
        count += gates.resetInput.length;
        count += gates.resetHidden.length;
        count += gates.resetBias.length;

        // Update gate
        count += gates.updateInput.length;
        count += gates.updateHidden.length;
        count += gates.updateBias.length;

        // Candidate
        count += gates.candidateInput.length;
        count += gates.candidateHidden.length;
        count += gates.candidateBias.length;
      }
    }

    // Output layer
    count += this.outputLayer.weight.length;
    count += this.outputLayer.bias.length;

    return count;
  }
}

export { GRUModel };