import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import requests

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

MODEL_URL = "https://huggingface.co/muhammadawaiskhan94725/brain-tumor-model/resolve/main/brain_tumor_model_complete.pth"
MODEL_PATH = "brain_tumor_model_complete.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (50MB)..."):
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

download_model()

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*14*14, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')

    # --- FIX FOR KeyError ---
    # Default classes if not saved in model
    default_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

    if isinstance(checkpoint, dict):
        # Get class names if available
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
        elif 'classes' in checkpoint:
            class_names = checkpoint['classes']
        else:
            class_names = default_classes

        # Get state dict if available
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        # checkpoint is directly state_dict
        state_dict = checkpoint
        class_names = default_classes

    model = BrainTumorCNN(num_classes=len(class_names))
    model.load_state_dict(state_dict)
    model.eval()
    return model, class_names

model, class_names = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

st.title("🧠 Brain Tumor Detection")
st.write(f"Model loaded. Classes: {', '.join(class_names)}")

uploaded_file = st.file_uploader("Choose MRI Image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI', use_column_width=True)
    if st.button('Predict Tumor', type="primary"):
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        st.success(f"**Prediction: {class_names[predicted.item()]}**")
        st.info(f"Confidence: {probs[0][predicted].item()*100:.2f}%")
