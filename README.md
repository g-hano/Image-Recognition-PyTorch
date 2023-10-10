# Fruit and Vegetable Recognition Model

## Overview

This deep learning model is designed to recognize and classify fruits and vegetables from images. It is implemented using PyTorch and consists of convolutional neural networks (CNNs) for image feature extraction and fully connected layers for classification.

## Model Architecture

The model architecture is defined as follows:

- **Input Layer**: This model expects color images with 3 channels (RGB).

- **Convolutional Layer 1**: The first convolutional layer with 6 output channels and a 5x5 kernel. It is followed by a ReLU activation function.

- **Max Pooling Layer 1**: A 2x2 max pooling layer to downsample the feature maps.

- **Convolutional Layer 2**: The second convolutional layer with 16 output channels and a 5x5 kernel, followed by a ReLU activation function.

- **Max Pooling Layer 2**: Another 2x2 max pooling layer for downsampling.

- **Fully Connected Layer 1 (fc1)**: A fully connected layer with 120 output neurons and ReLU activation.

- **Fully Connected Layer 2 (fc2)**: A fully connected layer with 84 output neurons and ReLU activation.

- **Fully Connected Layer 3 (Output Layer)**: The final fully connected layer with an output size equal to the number of classes (n_classes). It uses softmax activation for multi-class classification.

## Training

The model is trained using the following hyperparameters:

- Learning Rate: 0.001
- Number of Epochs: 10

Loss Function: Cross-Entropy Loss
Optimizer: Adam

During training, the model's weights are updated to minimize the cross-entropy loss between predicted labels and actual labels.

## Evaluation

After training, the model achieved the following performance:

- Training Loss: 0.3327
- Test Accuracy: 92.71%

The model's accuracy on the test dataset indicates its ability to correctly classify fruits and vegetables from real-world images.

## Usage

To use this model for fruit and vegetable recognition, you can follow these steps:

1. Initialize an instance of the `Model` class.
2. Load the trained weights (if not already loaded).
3. Prepare your input image by resizing it to the expected input dimensions (3x224x224 for RGB images).
4. Convert the image to a PyTorch tensor.
5. Pass the tensor through the model using the `forward` method to obtain classification probabilities.
6. Interpret the output to determine the class of the recognized fruit or vegetable.

```python
# Example usage
model = Model().to(device)

checkpoint_path = "model_last.pth"

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.eval()

# Prepare and preprocess your input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match your model's input size
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

image_path = "image_path.jpg"

image = Image.open(image_path)
image = transform(image)
image = image.unsqueeze(0)

# Pass the image through the model
with torch.no_grad():
    output = model(image.to(device))

# Interpret the output to determine the recognized class
_, predicted = torch.max(output, 1)
predicted_label = displayed_classes[predicted.item()]
print(f"Predicted Label: {predicted_label}")
```

## Dependencies

- PyTorch
- torchvision

## Author

Cihan Yalçın

