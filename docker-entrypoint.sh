#!/bin/bash
set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Initialize environment
init_environment() {
    echo "Initializing QTL-H environment..."
    mkdir -p /app/.cache /app/data /app/results
}

# Setup development tools if in development mode
setup_dev_environment() {
    if [ "$BUILD_ENV" = "development" ]; then
        echo "Setting up development environment..."
        if command_exists jupyter; then
            jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
        fi
    fi
}

# Function to handle GPU setup
setup_gpu() {
    if [ "$USE_GPU" = "true" ]; then
        echo "Checking GPU availability..."
        if command_exists nvidia-smi; then
            nvidia-smi
            export CUDA_VISIBLE_DEVICES=0
        else
            echo "Warning: GPU requested but NVIDIA drivers not found"
            export USE_GPU=false
        fi
    fi
}

# Function to run tests
run_tests() {
    echo "Running tests..."
    pytest_args="-v tests/"
    
    if [ "$USE_GPU" = "true" ]; then
        pytest_args="$pytest_args -m 'not no_gpu'"
    else
        pytest_args="$pytest_args -m 'not gpu'"
    fi
    
    if [ "$RUN_SLOW_TESTS" != "true" ]; then
        pytest_args="$pytest_args -m 'not slow'"
    fi
    
    pytest $pytest_args
}

# Function to start Jupyter services
start_jupyter() {
    local service=$1
    echo "Starting Jupyter $service..."
    if [ "$service" = "lab" ]; then
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    else
        jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    fi
}

# Print environment information
print_info() {
    echo "QTL-H Framework Environment Information"
    echo "======================================"
    echo "Python version: $(python --version)"
    echo "CUDA available: $USE_GPU"
    echo "Development mode: $BUILD_ENV"
    echo "Working directory: $(pwd)"
    echo "Cache directory: $QTLH_CACHE_DIR"
    echo "Data directory: $QTLH_DATA_DIR"
    echo "Results directory: $QTLH_RESULTS_DIR"
}

# Main entrypoint logic
main() {
    init_environment
    setup_gpu
    print_info
    
    case "$1" in
        test)
            run_tests
            ;;
        notebook)
            start_jupyter "notebook"
            ;;
        lab)
            start_jupyter "lab"
            ;;
        shell)
            /bin/bash
            ;;
        dev)
            setup_dev_environment
            /bin/bash
            ;;
        help)
            echo "Available commands:"
            echo "  test      - Run test suite"
            echo "  notebook  - Start Jupyter notebook"
            echo "  lab       - Start JupyterLab"
            echo "  shell     - Start bash shell"
            echo "  dev       - Start development environment"
            echo "  help      - Show this help message"
            ;;
        *)
            if [ -z "$1" ]; then
                python -m qtlh
            else
                exec "$@"
            fi
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
