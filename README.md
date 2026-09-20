🧠 Brain Tumor Classification using CNN from Scratch

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-green)
![Accuracy](https://img.shields.io/badge/Accuracy-~92%25-brightgreen)

An end-to-end Deep Learning project that classifies brain MRI scans into 4 categories: **Glioma, Meningioma, Pituitary, and No Tumor**. Built with a custom CNN from scratch (no transfer learning) and deployed as an interactive web app.

> 🚀 **Live Demo:** [Click here to try the app](https://your-streamlit-link.streamlit.app)  
> 📦 **Dataset:** [Brain Tumor MRI Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

📸 Demo Screenshots

| Glioma | Meningioma | Pituitary | No Tumor |
| :---: | :---: | :---: | :---: |
| ![glioma](screenshots/glioma.png) | ![meningioma](screenshots/meningioma.png) | ![pituitary](screenshots/pituitary.png) | ![notumor](screenshots/notumor.png) |

*Upload any MRI scan and get instant prediction with confidence score.*

🧩 Problem Statement
Brain tumors are life-threatening and early detection is critical. This project aims to automate the classification of brain tumors from MRI images using Computer Vision to assist in early screening (for educational purposes only).

📂 Dataset
- Total Images: ~7023 MRI scans
- Classes: 4
    - `glioma`
    - `meningioma`
    - `pituitary`
    - `notumor`
- Split: 80% Training, 20% Testing
- Image Size: 224x224

🏗️ Model Architecture (Built from Scratch)
I did **NOT use any pretrained model** like ResNet or VGG. Custom CNN designed in PyTorch:

```python
Features:
- Conv2d(3, 32) + ReLU + MaxPool
- Conv2d(32, 64) + ReLU + MaxPool
- Conv2d(64, 128) + ReLU + MaxPool
- Conv2d(128, 256) + ReLU + MaxPool

Classifier:
- Flatten
- Linear(256*14*14, 256) + ReLU + Dropout(0.25)
- Linear(256, 4)
*Training Details:*
- Framework: PyTorch
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Epochs: 20-30
- Normalization: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
- Augmentation: Resize, ToTensor

📊 Results
- Training Accuracy: ∼95%
- Validation Accuracy: ∼89-92%
- The model correctly distinguishes between tumor types with high confidence.

💻 Installation & Usage (Local)

*1. Clone the repo*
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
*2. Install dependencies*
pip install -r requirements.txt
*3. Run the Streamlit App*
streamlit run app.py
*requirements.txt*
streamlit
torch
torchvision
Pillow
📁 Project Structure
brain-tumor-app/
├── app.py                              # Streamlit web app
├── brain_tumor_model_complete.pth      # Trained model (state_dict + classes)
├── requirements.txt
├── README.md
└── screenshots/                        # 4 demo screenshots for LinkedIn
    ├── glioma.png
    ├── meningioma.png
    ├── pituitary.png
    └── notumor.png
🌐 Deployment on Streamlit Cloud
1. Push this repo to GitHub
2. Go to http://share.streamlit.io
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Deploy!

🛠️ Tech Stack
- *Language:* Python
- *Deep Learning:* PyTorch
- *Web App:* Streamlit
- *Image Processing:* PIL, torchvision
- *Platform:* Google Colab (Training), Streamlit Cloud (Deployment)

⚠️ Disclaimer
This project is for *educational and learning purposes only*. It is NOT a medical diagnosis tool. Always consult a qualified doctor for medical advice.

🔮 Future Improvements
- [ ] Add Grad-CAM for explainability (show where tumor is)
- [ ] Use EfficientNet / ResNet50 for higher accuracy
- [ ] Add PDF report generation
- [ ] Deploy with Docker + Hugging Face Spaces

👨‍💻 Author
*Muhammad Awais khan*
- LinkedIn: https://linkedin.com/in/yourprofile
- GitHub: https://github.com/yourusername

If you like this project, please ⭐ star the repo!

---