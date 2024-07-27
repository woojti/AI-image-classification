import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Constants
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CLASSES = 3

# Define Image Preprocessing for Prediction
transform_val = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the Model Architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_HEIGHT // 16) * (IMG_WIDTH // 16), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load the Trained Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(NUM_CLASSES).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Define Class Names
class_names = ['dogs', 'horses', 'lions']

# Function to Predict Class of New Image
def predict_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform_val(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted_idx = torch.max(output, 1)

    predicted_class = class_names[predicted_idx.item()]

    return predicted_class

# Predict classes for images from 2d-test1.png to 2d-test6.png
if __name__ == "__main__":
    current_dir = os.getcwd()
    path="datasets/image/"

    for i in range(1, 11):
        image_name = f"animal_{i}.png"
        image_path = os.path.join(current_dir, path, image_name)

        predicted_class = predict_image(image_path)
        print(f'Image {image_name}: Predicted class is {predicted_class}')
