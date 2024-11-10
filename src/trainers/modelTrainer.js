import * as tf from '@tensorflow/tfjs';
import { setupLogger } from '../utils/logger.js';
import path from 'path';
import fs from 'fs/promises';

const logger = setupLogger();

export class ModelTrainer {
  constructor() {
    this.model = null;
    this.tokenizer = null;
    this.trainingConfig = {
      learningRate: 1e-5,
      weightDecay: 0.01,
      warmupSteps: 500,
      maxSeqLength: 1024,
      gradientClipNorm: 1.0,
      validationSplit: 0.1
    };
  }

  async train({ epochs, batchSize }) {
    try {
      logger.info('Starting training process', { epochs, batchSize });
      
      await this.loadModel();
      const trainingData = await this.loadTrainingData();
      
      const splitIndex = Math.floor(trainingData.length * (1 - this.trainingConfig.validationSplit));
      const trainData = trainingData.slice(0, splitIndex);
      const validData = trainingData.slice(splitIndex);
      
      const trainDataset = await this.prepareDataset(trainData, batchSize);
      const validDataset = await this.prepareDataset(validData, batchSize);
      
      await this.trainingLoop(trainDataset, validDataset, epochs, batchSize);
      await this.saveModel();
      
      logger.info('Training completed');
      return true;
    } catch (error) {
      logger.error('Error in training:', error);
      throw error;
    }
  }

  async loadModel() {
    try {
      logger.info('Loading model...');
      
      this.model = tf.sequential({
        layers: [
          tf.layers.embedding({
            inputDim: 10000,
            outputDim: 256,
            inputLength: this.trainingConfig.maxSeqLength,
            maskZero: true
          }),
          tf.layers.layerNormalization(),
          tf.layers.dropout(0.1),
          tf.layers.lstm({
            units: 128,
            returnSequences: true,
            recurrentDropout: 0.1
          }),
          tf.layers.layerNormalization(),
          tf.layers.dropout(0.1),
          tf.layers.lstm({
            units: 64,
            recurrentDropout: 0.1
          }),
          tf.layers.layerNormalization(),
          tf.layers.dropout(0.1),
          tf.layers.dense({
            units: 10000,
            activation: 'softmax'
          })
        ]
      });

      const optimizer = tf.train.adam(this.trainingConfig.learningRate);
      
      this.model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });
      
      logger.info('Model loaded successfully');
    } catch (error) {
      logger.error('Error loading model:', error);
      throw error;
    }
  }

  async prepareDataset(data, batchSize) {
    try {
      const { inputTensors, targetTensors } = await this.tokenizeAndPadSequences(data);
      
      return tf.data.zip({
        inputs: tf.data.array(inputTensors).map(t => tf.tensor(t)),
        targets: tf.data.array(targetTensors).map(t => tf.tensor(t))
      }).shuffle(1000).batch(batchSize).prefetch(2);
    } catch (error) {
      logger.error('Error preparing dataset:', error);
      throw error;
    }
  }

  async tokenizeAndPadSequences(data) {
    const inputTensors = [];
    const targetTensors = [];

    for (const item of data) {
      const text = `${item.instruction} ${item.input} ${item.output}`;
      const tokens = Array.from(
        { length: this.trainingConfig.maxSeqLength },
        () => Math.floor(Math.random() * 10000)
      );
      
      inputTensors.push(tokens.slice(0, -1));
      targetTensors.push(tokens.slice(1));
    }

    return { inputTensors, targetTensors };
  }

  async trainingLoop(trainDataset, validDataset, epochs, batchSize) {
    try {
      let bestLoss = Infinity;
      let patienceCount = 0;
      const maxPatience = 3;

      for (let epoch = 1; epoch <= epochs; epoch++) {
        logger.info(`Starting epoch ${epoch}/${epochs}`);
        
        const trainMetrics = await this.trainEpoch(trainDataset, epoch);
        const validMetrics = await this.validateEpoch(validDataset, epoch);
        
        if (validMetrics.loss < bestLoss) {
          bestLoss = validMetrics.loss;
          patienceCount = 0;
          await this.saveCheckpoint(epoch, validMetrics);
        } else {
          patienceCount++;
          if (patienceCount >= maxPatience) {
            logger.info('Early stopping triggered');
            break;
          }
        }

        this.adjustLearningRate(epoch, validMetrics.loss);
      }
    } catch (error) {
      logger.error('Error in training loop:', error);
      throw error;
    }
  }

  async trainEpoch(dataset, epoch) {
    let totalLoss = 0;
    let totalAccuracy = 0;
    let batchCount = 0;

    await dataset.forEachAsync(async (batch) => {
      const history = await this.model.trainOnBatch(batch.inputs, batch.targets);
      totalLoss += history[0];
      totalAccuracy += history[1];
      batchCount++;

      if (batchCount % 10 === 0) {
        this.logTrainingProgress(epoch, batchCount, {
          loss: totalLoss / batchCount,
          accuracy: totalAccuracy / batchCount
        });
      }
    });

    return {
      loss: totalLoss / batchCount,
      accuracy: totalAccuracy / batchCount
    };
  }

  async validateEpoch(dataset, epoch) {
    let totalLoss = 0;
    let totalAccuracy = 0;
    let batchCount = 0;

    await dataset.forEachAsync(async (batch) => {
      const evaluation = await this.model.evaluateOnBatch(batch.inputs, batch.targets);
      totalLoss += evaluation[0];
      totalAccuracy += evaluation[1];
      batchCount++;
    });

    const metrics = {
      loss: totalLoss / batchCount,
      accuracy: totalAccuracy / batchCount
    };

    logger.info(`Validation metrics for epoch ${epoch}:`, metrics);
    return metrics;
  }

  adjustLearningRate(epoch, validationLoss) {
    if (epoch > this.trainingConfig.warmupSteps) {
      const newLR = this.trainingConfig.learningRate * 
        Math.pow(0.95, Math.floor(epoch / 2));
      this.model.optimizer.learningRate = newLR;
      logger.info(`Adjusted learning rate to ${newLR}`);
    }
  }

  logTrainingProgress(epoch, batchCount, metrics) {
    logger.info(
      `Epoch ${epoch}, Batch ${batchCount}: ` +
      `Loss: ${metrics.loss.toFixed(4)}, ` +
      `Accuracy: ${metrics.accuracy.toFixed(4)}`
    );
  }

  async saveCheckpoint(epoch, metrics) {
    const checkpointDir = path.join(process.env.OUTPUT_DIR || './output', 'checkpoints');
    const checkpointPath = path.join(checkpointDir, `checkpoint-epoch-${epoch}`);
    
    try {
      await fs.mkdir(checkpointDir, { recursive: true });
      await this.model.save(`file://${checkpointPath}`);
      
      const metadata = {
        epoch,
        timestamp: new Date().toISOString(),
        metrics,
        config: this.trainingConfig
      };
      
      await fs.writeFile(
        path.join(checkpointPath, 'metadata.json'),
        JSON.stringify(metadata, null, 2)
      );
      
      logger.info(`Checkpoint saved for epoch ${epoch}`);
    } catch (error) {
      logger.error('Error saving checkpoint:', error);
      throw error;
    }
  }

  async saveModel() {
    const outputPath = path.join(process.env.OUTPUT_DIR || './output', 'final_model');
    try {
      await fs.mkdir(outputPath, { recursive: true });
      await this.model.save(`file://${outputPath}`);
      logger.info(`Final model saved to ${outputPath}`);
    } catch (error) {
      logger.error('Error saving final model:', error);
      throw error;
    }
  }
}