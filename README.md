# LLaMA 3.1 Fine-tuning Project

This project provides tools for fine-tuning LLaMA 3.1 using GPT-4 outputs.

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Copy `.env.example` to `.env` and configure your environment variables

## Usage

### Process Training Data
```bash
npm start process-data
```

### Start Fine-tuning
```bash
npm start train --epochs 3 --batch-size 8
```

## Project Structure

- `src/`
  - `index.js` - Main entry point
  - `processors/` - Data processing utilities
  - `trainers/` - Model training implementation
  - `utils/` - Helper functions and logging

## Logging

Logs are stored in:
- `error.log` - Error messages
- `combined.log` - All log levels