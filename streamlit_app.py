import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import requests

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# ============================================================
# MODEL URL
# ============================================================

MODEL_URL = "https://huggingface.co/muhammadawaiskhan94725/brain-tumor-model/resolve/main/brain_tumor_model_complete.pth"

MODEL_PATH = "brain_tumor_model_complete.pth"


# ============================================================
# DOWNLOAD MODEL
# ============================================================

def download_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading model..."):

            response = requests.get(
                MODEL_URL,
                stream=True,
                timeout=120
            )

            response.raise_for_status()

            with open(MODEL_PATH, "wb") as f:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:
                        f.write(chunk)


download_model()


# ============================================================
# MODEL ARCHITECTURE
# MUST MATCH TRAINING MODEL
# ============================================================

class BrainTumorCNN(nn.Module):

    def _init_(self, num_classes=4):

        super()._init_()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256 * 14 * 14,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    # Load checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    # Your training code has 4 classes
    class_names = [
        "glioma",
        "meningioma",
        "notumor",
        "pituitary"
    ]

    # --------------------------------------------------------
    # Extract state_dict
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            # Your training code:
            # torch.save(model.state_dict(), ...)
            state_dict = checkpoint

    else:

        raise RuntimeError(
            "Invalid PyTorch model file."
        )

    # --------------------------------------------------------
    # FIX: Remove prefixes from saved weights
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        # Remove "model." prefix
        if key.startswith("model."):
            key = key[len("model."):]

        # Also handle DataParallel models
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    state_dict = cleaned_state_dict

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = BrainTumorCNN(num_classes=4)

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict,
        strict=True
    )

    # Evaluation mode
    model.eval()

    return model, class_names


# Load model
model, class_names = load_model()


# ============================================================
# IMAGE TRANSFORMATION
# MUST MATCH TRAINING PREPROCESSING
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
# STREAMLIT UI
# ============================================================

st.title("🧠 Brain Tumor Detection")

st.write(
    "Upload an MRI image to classify it into one of the four categories."
)

st.info(
    "Classes: " + ", ".join(class_names)
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Choose MRI Image...",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI",
        use_container_width=True
    )

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    if st.button(
        "Predict Tumor",
        type="primary"
    ):

        # Transform image
        img_tensor = transform(
            image
        ).unsqueeze(0)

        # Prediction
        with torch.no_grad():

            outputs = model(
                img_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = probabilities[
                0,
                predicted_class
            ].item()

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        prediction = class_names[
            predicted_class
        ]

        st.success(
            f"Prediction: *{prediction}*"
        )

        st.info(
            f"Confidence: *{confidence * 100:.2f}%*"
        )
