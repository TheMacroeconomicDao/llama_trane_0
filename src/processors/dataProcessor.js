import { readFile, writeFile } from 'fs/promises';
import { setupLogger } from '../utils/logger.js';
import { OpenAI } from 'openai';
import path from 'path';

const logger = setupLogger();

export class DataProcessor {
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });
    this.processedData = [];
    this.augmentationConfig = {
      temperature: [0.7, 1.0, 1.2],
      maxTokens: [100, 150, 200]
    };
  }

  async processData() {
    try {
      logger.info('Starting data processing');
      
      const rawData = await this.loadRawData();
      const cleanedData = await this.cleanData(rawData);
      const augmentedData = await this.augmentData(cleanedData);
      const gpt4Responses = await this.getGPT4Responses(augmentedData);
      const balancedData = await this.balanceDataset(gpt4Responses);
      const llamaFormatData = await this.convertToLLaMAFormat(balancedData);
      
      await this.saveProcessedData(llamaFormatData);
      await this.generateDatasetStats(llamaFormatData);
      
      logger.info('Data processing completed');
      return true;
    } catch (error) {
      logger.error('Error in data processing:', error);
      throw error;
    }
  }

  async loadRawData() {
    const dataPath = process.env.TRAINING_DATA_PATH;
    try {
      const rawData = await readFile(dataPath, 'utf-8');
      return JSON.parse(rawData);
    } catch (error) {
      logger.error('Error loading raw data:', error);
      throw error;
    }
  }

  async cleanData(data) {
    return data.filter(item => {
      // Удаление пустых или слишком коротких примеров
      if (!item.prompt || item.prompt.length < 10) return false;
      
      // Удаление дубликатов
      const isDuplicate = this.processedData.some(
        processed => processed.prompt === item.prompt
      );
      if (isDuplicate) return false;
      
      // Базовая очистка текста
      item.prompt = item.prompt
        .trim()
        .replace(/\s+/g, ' ')
        .replace(/[^\w\s.,?!-]/g, '');
      
      return true;
    });
  }

  async augmentData(data) {
    const augmentedData = [];
    
    for (const item of data) {
      augmentedData.push(item); // Оригинальный пример
      
      // Создание вариаций с разными параметрами
      for (const temp of this.augmentationConfig.temperature) {
        for (const maxTokens of this.augmentationConfig.maxTokens) {
          const augmentedPrompt = await this.createPromptVariation(
            item.prompt,
            temp,
            maxTokens
          );
          if (augmentedPrompt) {
            augmentedData.push({ prompt: augmentedPrompt });
          }
        }
      }
    }
    
    return augmentedData;
  }

  async createPromptVariation(prompt, temperature, maxTokens) {
    try {
      const completion = await this.openai.chat.completions.create({
        model: "gpt-4",
        messages: [
          {
            role: "system",
            content: "Create a variation of the following prompt while preserving its meaning and intent."
          },
          { role: "user", content: prompt }
        ],
        temperature,
        max_tokens: maxTokens
      });
      
      return completion.choices[0].message.content;
    } catch (error) {
      logger.error('Error creating prompt variation:', error);
      return null;
    }
  }

  async getGPT4Responses(data) {
    const responses = [];
    const batchSize = 5; // Обработка батчами для избежания rate limits
    
    for (let i = 0; i < data.length; i += batchSize) {
      const batch = data.slice(i, i + batchSize);
      const batchPromises = batch.map(async (item) => {
        try {
          const completion = await this.openai.chat.completions.create({
            model: "gpt-4",
            messages: [
              {
                role: "system",
                content: "You are a helpful assistant providing detailed and accurate responses."
              },
              { role: "user", content: item.prompt }
            ],
            temperature: 0.7,
            max_tokens: 150
          });
          
          return {
            prompt: item.prompt,
            response: completion.choices[0].message.content
          };
        } catch (error) {
          logger.error('Error getting GPT-4 response:', error);
          return null;
        }
      });
      
      const batchResponses = await Promise.all(batchPromises);
      responses.push(...batchResponses.filter(r => r !== null));
      
      // Задержка между батчами
      if (i + batchSize < data.length) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    
    return responses;
  }

  async balanceDataset(data) {
    // Анализ длины ответов
    const responseLengths = data.map(item => item.response.length);
    const avgLength = responseLengths.reduce((a, b) => a + b, 0) / responseLengths.length;
    const stdDev = Math.sqrt(
      responseLengths.reduce((a, b) => a + Math.pow(b - avgLength, 2), 0) / responseLengths.length
    );
    
    // Фильтрация экстремальных значений
    return data.filter(item => {
      const length = item.response.length;
      return length > avgLength - 2 * stdDev && length < avgLength + 2 * stdDev;
    });
  }

  async convertToLLaMAFormat(data) {
    return data.map(item => ({
      instruction: item.prompt,
      input: "",
      output: item.response,
      metadata: {
        responseLength: item.response.length,
        promptTokens: item.prompt.split(/\s+/).length,
        responseTokens: item.response.split(/\s+/).length
      }
    }));
  }

  async generateDatasetStats(data) {
    const stats = {
      totalExamples: data.length,
      averagePromptLength: 0,
      averageResponseLength: 0,
      promptLengthDistribution: {},
      responseLengthDistribution: {}
    };
    
    data.forEach(item => {
      const promptTokens = item.metadata.promptTokens;
      const responseTokens = item.metadata.responseTokens;
      
      stats.averagePromptLength += promptTokens;
      stats.averageResponseLength += responseTokens;
      
      // Распределение длин
      stats.promptLengthDistribution[promptTokens] = 
        (stats.promptLengthDistribution[promptTokens] || 0) + 1;
      stats.responseLengthDistribution[responseTokens] = 
        (stats.responseLengthDistribution[responseTokens] || 0) + 1;
    });
    
    stats.averagePromptLength /= data.length;
    stats.averageResponseLength /= data.length;
    
    const statsPath = path.join(process.env.OUTPUT_DIR, 'dataset_stats.json');
    await writeFile(statsPath, JSON.stringify(stats, null, 2));
    logger.info('Dataset statistics generated');
  }

  async saveProcessedData(processedData) {
    const outputPath = path.join(process.env.OUTPUT_DIR, 'processed_data.json');
    try {
      await writeFile(outputPath, JSON.stringify(processedData, null, 2));
      logger.info(`Processed data saved to ${outputPath}`);
      this.processedData = processedData;
    } catch (error) {
      logger.error('Error saving processed data:', error);
      throw error;
    }
  }
}