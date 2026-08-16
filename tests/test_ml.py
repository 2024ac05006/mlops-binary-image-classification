import pytest
import torch
from PIL import Image
import sys
import os

# Test the image transform pipeline and the SimpleCNN model's forward pass logic from src/app.py

# Add the project root to the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.app import SimpleCNN, transform

def test_data_preprocessing():
    """Test the data pre-processing function (transforms)."""
    # Create a dummy image (e.g., 500x500 RGB)
    dummy_img = Image.new('RGB', (500, 500), color='red')
    
    # Apply the transform used in our API
    tensor = transform(dummy_img)
    
    # Assert the output is a tensor with the required 224x224 size and 3 channels
    assert torch.is_tensor(tensor)
    assert tensor.shape == (3, 224, 224)

def test_model_inference():
    """Test the model utility/inference function."""
    model = SimpleCNN()
    model.eval()
    
    # Create a dummy tensor simulating a pre-processed image batch (1 image, 3 channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Run inference
    with torch.no_grad():
        output = model(dummy_input)
        
    # Assert the output shape matches the binary classification classes (1 batch, 2 classes)
    assert output.shape == (1, 2)