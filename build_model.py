import torch
import torch.onnx
from model.u2net_refactor import (
    U2NET_lite,
)  # Import the model definition from your file

print("Loading U-2-Net Lite model architecture...")
# Create an instance of the model
model = U2NET_lite()

print("Loading pre-trained weights (u2netp.pth)...")
# Load the weights from the .pth file
# We use map_location='cpu' to ensure it runs even without a GPU
model.load_state_dict(
    torch.load("saved_models/u2net/u2net.pth", map_location=torch.device("cpu"))
)

# Set the model to evaluation mode (important for conversion)
model.eval()

# Create a dummy input tensor with the expected shape
# (batch_size, channels, height, width)
# 320x320 is a standard size for this model.
dummy_input = torch.randn(1, 3, 320, 320)

# The refactored model returns a list of 7 output maps.
# We need to name them for the ONNX file.
output_names = [
    "output_0",  # This is the main, fused output mask
    "output_1",
    "output_2",
    "output_3",
    "output_4",
    "output_5",
    "output_6",
]

print("Exporting to ONNX (u2net_lite.onnx)...")

torch.onnx.export(
    model,
    dummy_input,
    "u2net_lite.onnx",
    export_params=True,  # Store the weights in the model file
    opset_version=11,  # A stable version for web compatibility
    do_constant_folding=True,  # Optimizes the model
    input_names=["input"],  # Name the input
    output_names=output_names,  # Name the outputs
)

print("-" * 50)
print("Successfully converted to u2net_lite.onnx")
print("You can now use this file with onnxruntime-web in JavaScript.")
print(f"Your model has 1 input: 'input'")
print(f"Your model has 7 outputs: {', '.join(output_names)}")
print("NOTE: 'output_0' is the final fused mask you most likely want to use.")
