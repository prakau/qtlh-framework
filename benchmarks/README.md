# QTL-H Framework Benchmarking Tools

This directory contains tools and utilities for benchmarking the QTL-H Framework's performance across different modules and configurations.

## Directory Structure

```
benchmarks/
├── benchmark_config.yaml    # Configuration for benchmark runs
├── run_benchmarks.py       # Main benchmark runner script
├── report_generator.py     # Benchmark report generation utilities
├── README.md              # This file
└── results/              # Benchmark results directory
    └── README.md         # Results documentation
```

## Running Benchmarks

### Basic Usage

```bash
# Run all benchmarks
python -m pytest benchmarks/run_benchmarks.py --benchmark-only

# Run specific module benchmarks
python -m pytest benchmarks/run_benchmarks.py --benchmark-only -k "quantum"
python -m pytest benchmarks/run_benchmarks.py --benchmark-only -k "hd"
python -m pytest benchmarks/run_benchmarks.py --benchmark-only -k "topology"
```

### With Configuration

```bash
# Use custom configuration
BENCHMARK_CONFIG=path/to/custom_config.yaml python -m pytest benchmarks/run_benchmarks.py

# Enable GPU tests
USE_GPU=true python -m pytest benchmarks/run_benchmarks.py -m "gpu"
```

## Generating Reports

```bash
# Generate comprehensive report
python benchmarks/report_generator.py

# Generate report with baseline comparison
python benchmarks/report_generator.py --baseline results/baseline.json
```

## Configuration Options

Edit `benchmark_config.yaml` to customize:
- Test parameters
- Data sizes
- Hardware settings
- Output formats
- Performance thresholds

## Report Types

1. **HTML Report** (`report.html`)
   - Interactive visualizations
   - Performance comparisons
   - Memory usage analysis
   - Regression detection

2. **CSV Data** (`results.csv`)
   - Raw benchmark timings
   - Memory usage data
   - System metrics

3. **JSON Results** (`results.json`)
   - Complete benchmark data
   - Configuration used
   - System information

## Performance Tracking

- Results are automatically saved in the `results/` directory
- Each run creates a timestamped subdirectory
- Baseline results are maintained for regression testing
- Historical trends are tracked and visualized

## Best Practices

1. **Running Benchmarks**
   - Use a dedicated machine
   - Close unnecessary applications
   - Run multiple iterations
   - Include warmup rounds

2. **Comparing Results**
   - Use consistent hardware
   - Compare against baseline
   - Check for regressions
   - Document environment

3. **Interpreting Results**
   - Consider variance
   - Look for patterns
   - Check resource usage
   - Validate against thresholds

## GPU Testing

GPU benchmarks require:
- CUDA-capable GPU
- PyTorch with CUDA support
- Environment variable: `USE_GPU=true`

## Result Analysis

The report generator provides:
- Performance metrics
- Statistical analysis
- Regression detection
- Resource profiling
- Comparative visualizations

## Common Issues

1. **High Variance**
   - Increase number of runs
   - Check system load
   - Use warmup rounds

2. **Memory Issues**
   - Adjust batch sizes
   - Monitor swap usage
   - Check memory leaks

3. **GPU Problems**
   - Verify CUDA setup
   - Check GPU memory
   - Monitor temperatures

## Contributing

When adding new benchmarks:
1. Follow existing patterns
2. Add configuration options
3. Include documentation
4. Provide baseline results

## Support

For benchmark-related issues:
- Check the troubleshooting guide
- Review existing issues
- Contact maintainers

## Future Plans

- Automated performance tracking
- Cloud benchmarking support
- Advanced profiling tools
- Distributed testing
