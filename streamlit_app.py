import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ============================================================
# Brain Tumor CNN - architecture used during training
# ============================================================

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 224x224 -> 112 -> 56 -> 28 -> 14
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# Put the .pth file in the same folder as app.py
MODEL_PATH = "brain_tumor_model_complete (1).pth"


# ============================================================
# Image preprocessing - same as training
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# Model loading
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n\n"
            "Put the .pth file in the same folder as app.py."
        )

    model = BrainTumorCNN(num_classes=4)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu"),
        weights_only=False
    )

    # Support the common checkpoint formats:
    # 1. {'model_state_dict': ...}
    # 2. {'state_dict': ...}
    # 3. direct state_dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError(
            "Unsupported checkpoint format. Expected a PyTorch state_dict/checkpoint."
        )

    # Remove prefixes sometimes added by DataParallel or wrappers.
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        cleaned_state_dict[new_key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()

    return model


# ============================================================
# Prediction
# ============================================================

def predict(image, model):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_class = CLASS_NAMES[predicted_index.item()]
    confidence_value = confidence.item()

    return predicted_class, confidence_value, probabilities[0]


# ============================================================
# Streamlit UI
# ============================================================

st.title("🧠 Brain Tumor Detection")
st.write(
    "Upload a brain MRI image and the trained CNN will classify it "
    "into one of four classes."
)

st.info(
    "Classes: Glioma • Meningioma • No Tumor • Pituitary"
)

try:
    model = load_model()
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error("❌ Model could not be loaded.")
    st.code(str(e))
    st.stop()


uploaded_file = st.file_uploader(
    "Upload MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI",
        use_container_width=True
    )

    if st.button("🔍 Predict", type="primary"):
        with st.spinner("Analyzing image..."):
            predicted_class, confidence, probabilities = predict(
                image, model
            )

        st.subheader("Prediction")
        st.success(f"**{predicted_class}**")

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.subheader("Class probabilities")

        for class_name, probability in zip(
            CLASS_NAMES, probabilities.tolist()
        ):
            st.write(
                f"**{class_name}:** {probability * 100:.2f}%"
            )
            st.progress(float(probability))

        st.warning(
            "This application is for educational/research purposes only "
            "and is not a medical diagnosis. MRI results should be "
            "interpreted by a qualified medical professional."
        )
