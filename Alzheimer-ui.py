import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gc

st.set_page_config(page_title="Alzheimer's Prediction", page_icon="🧠")
st.title("Alzheimer's Prediction App")

CONFIDENCE_THRESHOLD = 0.5
MIN_IMAGE_SIZE = 32

class AlzheimerCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(AlzheimerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu((self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu((self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(F.relu((self.conv5(x))))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(F.relu(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def validate_image(image):
    if image.size[0] < MIN_IMAGE_SIZE or image.size[1] < MIN_IMAGE_SIZE:
        return False
    return True

@st.cache_resource
def load_model():
    model = AlzheimerCNN()
    try:
        state_dict = torch.load('best_model.pth', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

if model is None:
    st.error("Failed to load the model. Please check if 'best_model.pth' exists.")
    st.stop()

classes = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

st.sidebar.title("Navigation")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if uploaded_file is not None:
    try:
        with st.spinner('Processing image...'):
            image = Image.open(uploaded_file)

            if not validate_image(image):
                st.error(f"Image is too small. Please upload an image with dimensions of at least {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE} pixels.")
                st.stop()

            if image.mode != 'RGB':
                image = image.convert('RGB')

            input_img = transformer(image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_img)
                probabilities = torch.nn.functional.softmax(output[0], dim=0).numpy()

            st.image(image, caption="Uploaded Image", use_container_width=True)

            predicted_class = classes[probabilities.argmax()]
            confidence = probabilities.max()

            st.subheader("Prediction Results")
            if confidence >= CONFIDENCE_THRESHOLD:
                st.success(f"Prediction: {predicted_class} (Confidence: {confidence:.2%})")
            else:
                st.warning(f"Low confidence prediction: {predicted_class} (Confidence: {confidence:.2%})")

            st.subheader("Prediction Probabilities")
            st.bar_chart({classes[i]: float(probabilities[i]) for i in range(len(classes))})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del input_img
            del output
            gc.collect()

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

else:
    st.sidebar.warning("Please upload an image to get predictions.")

    with st.sidebar.expander("Help"):
        st.write("""
        - Upload an image of a brain scan.
        - The app will predict the likelihood of Alzheimer's disease.
        - Ensure the image is in JPG, JPEG, or PNG format.
        - Ensure the image is atleast 32 X 32 pixels
        """)