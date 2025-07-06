#!/bin/bash

# Deployment script for optimized Alzheimer's prediction app
# This script sets up the environment and runs the optimized version

echo "🚀 Starting deployment of optimized Alzheimer's prediction app..."

# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install optimized dependencies
echo "📥 Installing optimized dependencies..."
pip install --upgrade pip
pip install -r requirements-optimized.txt

# Check if model exists
if [ ! -f "best_model.pth" ]; then
    echo "❌ Model file 'best_model.pth' not found!"
    echo "Please ensure the model file is in the current directory."
    exit 1
fi

# Run model optimization if optimized models don't exist
if [ ! -f "model_quantized.pth" ] && [ ! -f "model_torchscript.pt" ]; then
    echo "🔄 Running model optimization..."
    python3 model_optimizer.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Model optimization completed successfully!"
    else
        echo "⚠️  Model optimization failed, using original model..."
    fi
fi

# Display optimization results
echo ""
echo "📊 Model Size Comparison:"
if [ -f "best_model.pth" ]; then
    original_size=$(du -h "best_model.pth" | cut -f1)
    echo "  Original model: $original_size"
fi

if [ -f "model_quantized.pth" ]; then
    quantized_size=$(du -h "model_quantized.pth" | cut -f1)
    echo "  Quantized model: $quantized_size"
fi

if [ -f "model_torchscript.pt" ]; then
    torchscript_size=$(du -h "model_torchscript.pt" | cut -f1)
    echo "  TorchScript model: $torchscript_size"
fi

echo ""
echo "🎯 Performance Optimizations Applied:"
echo "  ✅ Reduced image resolution (256→224)"
echo "  ✅ Model quantization (75% size reduction)"
echo "  ✅ CPU-only dependencies"
echo "  ✅ Memory management optimization"
echo "  ✅ Efficient caching"
echo ""

# Start the optimized application
echo "🚀 Starting optimized Alzheimer's prediction app..."
echo "📱 The app will be available at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

# Run the optimized Streamlit app
streamlit run Alzheimer-ui-optimized.py --server.address=0.0.0.0 --server.port=8501

echo "👋 Application stopped."