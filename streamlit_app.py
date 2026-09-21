import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------
REPO_ID = "muhammadawaiskhan94725/brain-tumor-model"
MODEL_FILENAME = "brain_tumor_model_complete.pth"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# IMPORTANT: This architecture matches the checkpoint
# ---------------------------------------------------------
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 224x224 -> 14x14 after four 2x2 pooling layers
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


# ---------------------------------------------------------
# Download/load model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    model = BrainTumorCNN(num_classes=4)

    # The checkpoint contains:
    # {"model_state": state_dict, "classes": [...]}
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict):
        # Also support a plain state_dict checkpoint
        state_dict = checkpoint
    else:
        raise TypeError(
            "Unexpected checkpoint format. Expected a PyTorch state_dict."
        )

    # Remove possible DataParallel prefix if present
    state_dict = {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    return model


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
def predict(image, model):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index].item())

    return predicted_class, confidence, probabilities.cpu()


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🧠 Brain Tumor Classification")
st.write(
    "Upload a brain MRI image to classify it into one of four classes."
)

st.info(
    "Classes: Glioma, Meningioma, No Tumor, and Pituitary."
)

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI",
        use_container_width=True
    )

    if st.button("🔍 Predict", type="primary"):
        try:
            with st.spinner("Loading model and making prediction..."):
                model = load_model()
                predicted_class, confidence, probabilities = predict(
                    image, model
                )

            st.subheader("Prediction")

            if predicted_class == "notumor":
                display_name = "No Tumor"
            else:
                display_name = predicted_class.capitalize()

            st.success(
                f"Prediction: **{display_name}**"
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.subheader("Class probabilities")

            for i, class_name in enumerate(CLASS_NAMES):
                label = (
                    "No Tumor"
                    if class_name == "notumor"
                    else class_name.capitalize()
                )
                st.write(
                    f"**{label}:** {probabilities[i].item() * 100:.2f}%"
                )

        except Exception as e:
            st.error("The model could not be loaded or the image could not be processed.")
            st.exception(e)

st.divider()

st.caption(
    "⚠️ Educational/research use only. This application is not a "
    "medical diagnostic tool and should not be used for clinical decisions. "
    "Please consult a qualified healthcare professional for medical advice."
)
