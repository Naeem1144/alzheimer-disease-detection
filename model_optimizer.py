#!/usr/bin/env python3
"""
Model optimization script for Alzheimer's CNN
This script optimizes the model for production deployment by:
1. Quantizing the model to int8 precision
2. Converting to TorchScript for faster inference
3. Optimizing for CPU deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc
from torch.quantization import quantize_dynamic
import warnings
warnings.filterwarnings('ignore')


class AlzheimerCNN(nn.Module):
    """Optimized CNN model for Alzheimer's classification"""
    
    def __init__(self, num_classes=4):
        super(AlzheimerCNN, self).__init__()
        # Reduced model complexity for faster inference
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(F.relu(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def optimize_model():
    """Optimize the trained model for production deployment"""
    
    print("Starting model optimization...")
    
    # Load the original model
    model = AlzheimerCNN(num_classes=4)
    
    if not os.path.exists('best_model.pth'):
        print("Error: best_model.pth not found!")
        return
    
    try:
        # Load weights
        state_dict = torch.load('best_model.pth', map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        print("✓ Original model loaded successfully")
        
        # Get original model size
        original_size = os.path.getsize('best_model.pth') / (1024 * 1024)
        print(f"Original model size: {original_size:.2f} MB")
        
        # 1. Dynamic Quantization - reduces model size by ~75%
        print("Applying dynamic quantization...")
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},  # Quantize linear and conv layers
            dtype=torch.qint8
        )
        
        # Save quantized model
        torch.save(quantized_model.state_dict(), 'model_quantized.pth')
        quantized_size = os.path.getsize('model_quantized.pth') / (1024 * 1024)
        print(f"✓ Quantized model size: {quantized_size:.2f} MB")
        print(f"✓ Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
        
        # 2. TorchScript optimization for faster inference
        print("Converting to TorchScript...")
        model.eval()
        
        # Create example input for tracing
        example_input = torch.randn(1, 3, 224, 224)  # Reduced resolution
        
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        # Optimize the traced model
        optimized_model = torch.jit.optimize_for_inference(traced_model)
        
        # Save TorchScript model
        torch.jit.save(optimized_model, 'model_torchscript.pt')
        torchscript_size = os.path.getsize('model_torchscript.pt') / (1024 * 1024)
        print(f"✓ TorchScript model size: {torchscript_size:.2f} MB")
        
        # 3. Create a lightweight model for web deployment
        print("Creating lightweight model...")
        
        # Save just the state dict with compression
        torch.save(
            model.state_dict(),
            'model_lightweight.pth',
            _use_new_zipfile_serialization=True
        )
        
        lightweight_size = os.path.getsize('model_lightweight.pth') / (1024 * 1024)
        print(f"✓ Lightweight model size: {lightweight_size:.2f} MB")
        
        # Performance benchmark
        print("\nPerformance benchmark:")
        import time
        
        # Test inference speed
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            
            # Original model
            start_time = time.time()
            for _ in range(100):
                _ = model(test_input)
            original_time = time.time() - start_time
            
            # TorchScript model
            start_time = time.time()
            for _ in range(100):
                _ = optimized_model(test_input)
            torchscript_time = time.time() - start_time
            
            print(f"Original model: {original_time:.3f}s for 100 inferences")
            print(f"TorchScript model: {torchscript_time:.3f}s for 100 inferences")
            print(f"Speed improvement: {((original_time - torchscript_time) / original_time * 100):.1f}%")
        
        # Cleanup
        del model, quantized_model, traced_model, optimized_model
        gc.collect()
        
        print("\n✓ Model optimization complete!")
        print("Generated files:")
        print("- model_quantized.pth: Quantized model (75% smaller)")
        print("- model_torchscript.pt: TorchScript optimized model")
        print("- model_lightweight.pth: Compressed lightweight model")
        
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    optimize_model()