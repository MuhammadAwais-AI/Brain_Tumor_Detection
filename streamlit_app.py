import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

@st.cache_resource
def load_model():
    checkpoint = torch.load('brain_tumor_model_complete.pth', map_location='cpu')
    model = BrainTumorCNN(len(checkpoint['classes']))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint['classes']

model, class_names = load_model()

st.set_page_config(page_title="Brain Tumor AI", page_icon="🧠", layout="centered")
st.markdown("""
<style>
.big {font-size:20px; font-weight:600}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Brain Tumor Classification")
st.caption("AI-powered MRI analysis | For educational purpose only")

col1, col2 = st.columns([1,1])
with col1:
    st.info("**Classes:** glioma, meningioma, pituitary, notumor")
with col2:
    st.warning("⚠️ Not a medical diagnosis")

uploaded = st.file_uploader("Upload MRI Image", type=["jpg","jpeg","png"])
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption="Input MRI", use_column_width=True)
    with st.spinner("Analyzing..."):
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            prob = torch.softmax(model(tensor),1)[0]
            pred = torch.argmax(prob).item()

    st.divider()
    st.markdown(f"### Result: **{class_names[pred].upper()}**")
    st.metric("Confidence", f"{prob[pred]*100:.2f}%")

    st.write("Confidence per class:")
    for i, c in enumerate(class_names):
        st.progress(float(prob[i]), text=f"{c}: {prob[i]*100:.1f}%")