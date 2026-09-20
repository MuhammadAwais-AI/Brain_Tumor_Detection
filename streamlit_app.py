import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import gdown

# --- PAGE CONFIG ---
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

# --- CONFIG ---
MODEL_ID = "1RzmEhsIWG_iczHIeOiB6evLrhcfTfL6p"
MODEL_PATH = "brain_tumor_model_complete.pth"

# --- DOWNLOAD MODEL IF NOT EXISTS ---
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model for first time... please wait (50MB)..."):
        # This fuzzy=True fixes the FileURLRetrievalError
        gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False, fuzzy=True)

# --- MODEL ARCHITECTURE (Must be same as training) ---
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
            nn.Linear(256*14*14, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    class_names = checkpoint['class_names']
    model = BrainTumorCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names

# Load model
try:
    model, class_names = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- IMAGE TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- UI ---
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI scan to classify: Glioma, Meningioma, Pituitary, or No Tumor")

uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI', use_column_width=True)

    if st.button('Predict Tumor', type="primary"):
        with st.spinner("Analyzing..."):
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence = probs[0][predicted].item() * 100
                predicted_class = class_names[predicted.item()]

            st.success(f"**Prediction: {predicted_class}**")
            st.info(f"Confidence: {confidence:.2f}%")

            # Show all probabilities
            st.write("**All Class Probabilities:**")
            for i, class_name in enumerate(class_names):
                st.write(f"- {class_name}: {probs[0][i].item()*100:.2f}%")
else:
    st.info("Please upload an image to get started.")
