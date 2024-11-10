import { Command } from 'commander';
import { DataProcessor } from './processors/dataProcessor.js';
import { ModelTrainer } from './trainers/modelTrainer.js';
import { setupLogger } from './utils/logger.js';
import dotenv from 'dotenv';

dotenv.config();
const logger = setupLogger();
const program = new Command();

program
  .name('llama-finetuning')
  .description('LLaMA 3.1 Fine-tuning CLI')
  .version('1.0.0');

program
  .command('process-data')
  .description('Process and prepare GPT-4 data for fine-tuning')
  .action(async () => {
    try {
      const processor = new DataProcessor();
      await processor.processData();
      logger.info('Data processing completed successfully');
    } catch (error) {
      logger.error('Error processing data:', error);
      process.exit(1);
    }
  });

program
  .command('train')
  .description('Start fine-tuning LLaMA 3.1')
  .option('-e, --epochs <number>', 'number of training epochs', '10')
  .option('-b, --batch-size <number>', 'batch size', '32')
  .action(async (options) => {
    try {
      const trainer = new ModelTrainer();
      await trainer.train({
        epochs: parseInt(options.epochs),
        batchSize: parseInt(options.batchSize)
      });
      logger.info('Training completed successfully');
    } catch (error) {
      logger.error('Error during training:', error);
      process.exit(1);
    }
  });

program.parse();