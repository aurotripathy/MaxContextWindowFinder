#### Note: Derived from https://github.com/Scionero/MaxContextFinder/tree/main with many simplifications and minor changes
# OpenAI-Compatible LLM Server Context Window Size Discovery

A tool to determine the maximum usable context size for models served from an OpenAI-compatible endpoint while monitoring performance. This tool helps you find the optimal balance between context size and performance for your specific hardware setup.

## Overview

This tool tests increasing context sizes with your chosen model on an OpenAI-compatible server to find the maximum size that maintains acceptable performance. It monitors:
- Token processing speed (tokens per second)
- Response times
- Model behavior at different context lengths

## Prerequisites

### Linux
- Python 3.8+
- An OpenAI-compatible endpoint running

## Installation

1. Clone this repository:
```bash
git clone https://github.com/scionero/maxcontextfinder
cd maxcontextfinder
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python main.py
```

Example:
```bash
python main.py codellama:latest
```
Example with a custom server:
```bash
python main.py codellama:latest --endpoint_url http://your-server --port <your port number>
```

### Command Line Options

- `model`: (Required) The model name (e.g., 'codellama:latest', 'llama2:13b')
- `--endpoint_url`: OpenAI-compatible server endpoint URL (default: http://localhost)
- `--port`: OpenAI-compatible server port (default: 8000)
- `--min_token_rate`: Minimum acceptable tokens per second (default: 10)
- `--start`: Starting context size (default: 1024)
- `--step`: Step size for context increments (default: 1024)
- `--tests`: Number of tests per context size (default: 3)

Example with all options:
```bash
python main.py mistral:7b --min_token_rate 15 --start 2048 --step 2048 --tests 5
```

### Output

The tool generates detailed logs including:
- Test parameters and configuration
- Performance metrics for each test
- Token processing speeds
- Final recommended context size

Logs are saved to the `logs` directory with names: `context_test_MODEL_TIMESTAMP.log`

## Important Notes

1. **Framework Specificity**: Results are specific to OpenAI-compatible servers and may differ from other frameworks like:
   - Pure llama.cpp
   - vLLM
   - Different quantization methods
   - Other serving frameworks

2. **Hardware Dependence**: Results depend on your hardware:
   - GPU memory and performance
   - CPU capabilities
   - System memory
   - Storage speed

## Understanding Results

The tool stops testing larger context sizes when:
- Token processing speed drops below the minimum threshold
- Model encounters errors or timeouts

The "maximum recommended context size" is the largest size that maintained acceptable performance across all metrics.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional GPU support
- More performance metrics
- Support for other frameworks
- Better token counting accuracy
- Alternative testing methodologies

Please feel free to:
- Open issues for bugs or feature requests
- Submit pull requests with improvements
- Share your testing results
- Suggest better testing methodologies

## Disclaimer

Results should be considered approximate. Real-world performance may vary based on:
- Specific prompt content
- Model implementation details
- System load and conditions
- Hardware configuration
