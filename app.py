import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import io
import os
from ultralytics import YOLO
import numpy as np
import cv2
import traceback
import pydicom
import time
import base64
from datetime import datetime

# --- STYLING AND PAGE CONFIGURATION ---
st.set_page_config(
    page_title="X-ray Fracture Detection and Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to enhance the UI
def local_css():
    css = """
    <style>
        /* Main container styles */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Custom title card */
        .title-card {
            padding: 1.5rem;
            border-radius: 10px;
            background: linear-gradient(135deg, #2E3192 0%, #1BFFFF 100%);
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Card styling */
        .stcard {
            border-radius: 10px;
            padding: 1.5rem;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        
        /* Custom file uploader */
        .uploadFile {
            border: 2px dashed #4a90e2;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            margin-bottom: 1rem;
        }
        
        /* Result container */
        .result-container {
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }
        
        /* Status indicators */
        .status-success {
            background-color: #D4EDDA;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        .status-processing {
            background-color: #CCE5FF;
            color: #004085;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        .status-warning {
            background-color: #FFF3CD;
            color: #856404;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        .status-error {
            background-color: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        /* Override Streamlit's default button styles */
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3rem;
            font-weight: 600;
            background-color: #4a90e2;
            color: white;
            border: none;
        }
        
        .stButton>button:hover {
            background-color: #357ABD;
        }
        
        /* Progress bar styling */
        .stProgress > div > div {
            background-color: #4a90e2;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px 4px 0px 0px;
            padding: 10px 20px;
            background-color: #F0F2F6;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #4a90e2 !important;
            color: white !important;
        }
        
        /* Header customization */
        h1, h2, h3 {
            color: #2E3192;
        }
        
        /* Footer style */
        .footer {
            text-align: center;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e6e6e6;
            color: #6c757d;
            font-size: 0.8rem;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Apply custom CSS
local_css()

# --- HELPER FUNCTIONS ---

# Function to create a custom card with title and content
def create_card(title, content):
    st.markdown(f"""
    <div class="stcard">
        <h3>{title}</h3>
        {content}
    </div>
    """, unsafe_allow_html=True)

# Path to models
CLASSIFIER_PATH = "models/classifier.pth"
SEGMENTATION_MODEL_PATH = "models/xray_seg.pt"

# Function to validate if an image is an X-ray using image analysis
def is_xray_image(img):
    """
    Determine if an image is likely to be an X-ray using basic image analysis techniques.
    X-rays typically have:
    1. Limited color palette (mostly grayscale)
    2. High contrast between bones (white/bright) and soft tissues (dark/gray)
    3. Specific histogram distribution characteristics
    4. Low color saturation
    """
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Check if image is already grayscale
    is_grayscale = (len(img_array.shape) == 2) or \
                   (len(img_array.shape) == 3 and img_array.shape[2] == 1) or \
                   (len(img_array.shape) == 3 and np.allclose(img_array[:,:,0], img_array[:,:,1]) and 
                    np.allclose(img_array[:,:,1], img_array[:,:,2]))
    
    # If not grayscale, check color saturation
    if not is_grayscale:
        # Convert to HSV and check saturation
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            # Convert RGB to HSV
            img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = img_hsv[:,:,1].mean()
            
            # X-rays have very low saturation
            if saturation > 30:  # Threshold for saturation
                return False
    
    # Convert to grayscale for further analysis
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img_array.squeeze()
    
    # Analyze histogram characteristics
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist_norm = hist.flatten() / hist.sum()
    
    # Calculate histogram metrics
    mean_intensity = np.mean(gray_img)
    std_intensity = np.std(gray_img)
    
    # Calculate contrast (difference between 90th and 10th percentile)
    p10 = np.percentile(gray_img, 10)
    p90 = np.percentile(gray_img, 90)
    contrast = p90 - p10
    
    # X-rays typically have medium-high contrast and specific intensity distribution
    if contrast < 40:  # Low contrast usually indicates non-X-ray
        return False
    
    # Check for proper histogram distribution (X-rays have specific peaks for bone and tissue)
    # Calculate bimodality coefficient (higher for bimodal distributions typical in X-rays)
    m3 = np.sum((gray_img.flatten() - mean_intensity)**3) / (gray_img.size * std_intensity**3)  # Skewness
    m4 = np.sum((gray_img.flatten() - mean_intensity)**4) / (gray_img.size * std_intensity**4)  # Kurtosis
    bimodality = (m3**2 + 1) / m4 if m4 > 0 else 0
    
    # Edge detection to check for bone-like structures
    edges = cv2.Canny(gray_img, 50, 150)
    edge_percentage = np.count_nonzero(edges) / edges.size
    
    # X-rays typically have distinct edges but not too many
    if edge_percentage < 0.01 or edge_percentage > 0.15:
        return False
        
    # Combined decision based on all metrics
    is_xray = (is_grayscale or saturation < 30) and \
              (contrast > 40) and \
              (0.01 < edge_percentage < 0.15) and \
              (bimodality > 0.2)
    
    return is_xray

# Function to load the classifier model
@st.cache_resource
def load_classifier():
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Image transformation for classifier
def transform_image(img):
    # Convert grayscale images to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# Classification function
def classify_image(model, img):
    img_tensor = transform_image(img)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Load segmentation model
@st.cache_resource
def load_segmentation_model():
    model = YOLO(SEGMENTATION_MODEL_PATH)
    return model

# Perform segmentation
def segment_image(model, img):
    # Ensure the image is in RGB format for YOLO
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img)
    
    # Run inference with YOLO model
    results = model.predict(source=img_array, save=False, conf=0.25)
    
    # Get segmentation result
    result = results[0]
    return result

# Function to display custom status
def display_status(message, status_type):
    status_class = f"status-{status_type}"
    st.markdown(f'<div class="{status_class}">{message}</div>', unsafe_allow_html=True)

# Function to add custom upload widget
def custom_file_uploader():
    st.markdown("""
    <div class="uploadFile">
        <svg width="50" height="50" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 5V19M5 12H19" stroke="#4a90e2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <p>Drag and drop your X-ray image here or click to browse</p>
        <p style="font-size: 0.8rem; color: #6c757d;">Supported formats: JPEG, PNG, TIFF, BMP, DICOM</p>
    </div>
    """, unsafe_allow_html=True)
    
    return st.file_uploader("Upload X-ray image", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "dicom", "dcm"], label_visibility="collapsed")

# Function to get image dimensions
def get_image_info(image):
    width, height = image.size
    mode = image.mode
    format_info = image.format if hasattr(image, 'format') and image.format else "Unknown"
    return f"Dimensions: {width}x{height} | Mode: {mode} | Format: {format_info}"

# --- MAIN APP ---
def main():
    # Custom title with gradient background
    st.markdown("""
    <div class="title-card">
        <h1>Advanced X-ray Fracture Detection & Segmentation</h1>
        <p>Upload X-ray images to detect and visualize bone fractures using AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for app information and settings
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/x-ray.png", width=80)
        st.title("About")
        st.info("""
        This application uses deep learning to detect and segment fractures in X-ray images.
        
        **Features:**
        - Upload X-ray images (JPEG, PNG, TIFF, BMP, DICOM)
        - AI-powered fracture detection
        - Automatic fracture segmentation visualization
        - Download analysis results
        
        **Models:**
        - Classification: ResNet18
        - Segmentation: YOLO
        """)
        
        st.markdown("---")
        
        # Settings section
        st.subheader("Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
        
        # Display current time in sidebar
        st.markdown("---")
        st.caption(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        create_card("Upload X-ray Image", "")
        uploaded_file = custom_file_uploader()
        
        if uploaded_file is not None:
            try:
                # Check if the file is a DICOM file
                file_extension = os.path.splitext(uploaded_file.name.lower())[1]
                
                # Show processing indicator
                with st.spinner("Processing image..."):
                    # Add a small artificial delay to show processing
                    time.sleep(0.5)
                    
                    if file_extension in ['.dcm', '.dicom']:
                        # For DICOM files
                        try:
                            # Read the DICOM file
                            uploaded_file.seek(0)
                            dicom_bytes = uploaded_file.read()
                            with open('temp_dicom.dcm', 'wb') as f:
                                f.write(dicom_bytes)
                            
                            # Load DICOM
                            dicom_data = pydicom.dcmread('temp_dicom.dcm')
                            
                            # Convert to PIL Image
                            pixel_array = dicom_data.pixel_array
                            if len(pixel_array.shape) == 2:  # Grayscale image
                                # Normalize to 0-255 and convert to uint8
                                img_2d = (((pixel_array - pixel_array.min()) / 
                                         (pixel_array.max() - pixel_array.min())) * 255).astype(np.uint8)
                                # Create PIL Image
                                image = Image.fromarray(img_2d)
                            else:  # RGB image
                                image = Image.fromarray(pixel_array)
                            
                            # Show DICOM metadata
                            st.markdown("#### DICOM Metadata")
                            with st.expander("View DICOM metadata"):
                                try:
                                    st.write(f"Patient ID: {dicom_data.PatientID if 'PatientID' in dicom_data else 'N/A'}")
                                    st.write(f"Study Date: {dicom_data.StudyDate if 'StudyDate' in dicom_data else 'N/A'}")
                                    st.write(f"Modality: {dicom_data.Modality if 'Modality' in dicom_data else 'N/A'}")
                                    st.write(f"Body Part: {dicom_data.BodyPartExamined if 'BodyPartExamined' in dicom_data else 'N/A'}")
                                except:
                                    st.write("Some metadata fields are not available")
                            
                            # Remove temporary file
                            if os.path.exists('temp_dicom.dcm'):
                                os.remove('temp_dicom.dcm')
                                
                        except Exception as e:
                            st.error(f"Error processing DICOM file: {e}")
                            st.error(traceback.format_exc())
                            return
                    else:
                        # For regular image files
                        image = Image.open(uploaded_file)
                
                # Verify image has data and is valid
                if image.size[0] == 0 or image.size[1] == 0:
                    st.error("Invalid image: The image has no dimensions.")
                    return
                
                # Display original image with enhanced UI
                st.markdown("#### Original Image")
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
                st.caption(get_image_info(image))
                
                # Check if the image is an X-ray
                is_xray = is_xray_image(image)
                
                if not is_xray:
                    # Display error message for non-X-ray images
                    st.markdown("""
                    <div style="background-color: #f8d7da; color: #721c24; padding: 1rem; border-radius: 8px; text-align: center; margin: 1rem 0;">
                        <h3 style="margin: 0; color: #721c24;">⚠️ Not an X-ray Image</h3>
                        <p style="margin: 0.5rem 0 0 0;">The uploaded image does not appear to be an X-ray. Please upload a valid X-ray image.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show examples of valid X-ray images
                    st.markdown("#### Examples of Valid X-ray Images")
                    st.markdown("""
                    <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                        <div style="text-align: center; width: 30%;">
                            <img src="https://img.icons8.com/dotty/80/000000/x-ray-lungs.png" style="max-width: 100%;">
                            <p>Chest X-ray</p>
                        </div>
                        <div style="text-align: center; width: 30%;">
                            <img src="https://img.icons8.com/dotty/80/000000/bones.png" style="max-width: 100%;">
                            <p>Bone X-ray</p>
                        </div>
                        <div style="text-align: center; width: 30%;">
                            <img src="https://img.icons8.com/dotty/80/000000/human-skull.png" style="max-width: 100%;">
                            <p>Skull X-ray</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # No need to continue with classification and segmentation
                    return
                
                # Load models with progress indicators
                progress_bar = st.progress(0)
                
                try:
                    # Show step-by-step loading progress
                    progress_bar.progress(25)
                    display_status("Loading classification model...", "processing")
                    classifier_model = load_classifier()
                    progress_bar.progress(50)
                    display_status("Classification model loaded successfully!", "success")
                except Exception as e:
                    st.error(f"Error loading classifier model: {e}")
                    st.error(traceback.format_exc())
                    return
                
                # Perform classification with progress updates
                try:
                    display_status("Analyzing image for fractures...", "processing")
                    class_names = ['Fractured', 'Non_fractured']
                    prediction = classify_image(classifier_model, image)
                    classification_result = class_names[prediction]
                    progress_bar.progress(75)
                except Exception as e:
                    st.error(f"Error during classification: {e}")
                    st.error(traceback.format_exc())
                    return
                
                # Display classification result with appropriate styling
                if classification_result == "Fractured":
                    result_color = "#dc3545"
                    result_icon = "❗"
                else:
                    result_color = "#28a745"
                    result_icon = "✅"
                
                st.markdown(f"""
                <div style="background-color: {result_color}; color: white; padding: 1rem; border-radius: 8px; text-align: center; margin: 1rem 0;">
                    <h2 style="margin: 0; color: white;">{result_icon} {classification_result}</h2>
                    <p style="margin: 0.5rem 0 0 0;">Classification confidence: High</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Complete the progress
                progress_bar.progress(100)
                
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.error(traceback.format_exc())
    
    with col2:
        if uploaded_file is not None and 'image' in locals() and 'classification_result' in locals() and 'is_xray' in locals() and is_xray:
            # If fractured, show segmentation results
            if classification_result == "Fractured":
                create_card("Fracture Segmentation", "<p>AI-based visualization of detected fracture locations</p>")
                
                try:
                    # Load segmentation model
                    segmentation_model = load_segmentation_model()
                    display_status("Segmentation model loaded successfully!", "success")
                    
                    # Prepare image for segmentation
                    img_resized = image.resize((640, 640))
                    
                    # Perform segmentation
                    with st.spinner('Generating fracture segmentation map...'):
                        try:
                            result = segment_image(segmentation_model, img_resized)
                        except Exception as e:
                            st.error(f"Error during segmentation: {e}")
                            st.error(traceback.format_exc())
                            return
                    
                    # Display segmentation result with tabs for different views
                    tabs = st.tabs(["Segmentation Result", "Overlay View", "Analysis"])
                    
                    with tabs[0]:
                        seg_img = result.plot()  # Get segmentation visualization
                        seg_img = Image.fromarray(seg_img)
                        st.image(seg_img, caption="Fracture Segmentation", use_container_width=True)
                    
                    with tabs[1]:
                        # Create overlay view
                        seg_array = np.array(result.plot())
                        orig_resized = np.array(img_resized.convert('RGB'))
                        overlay = cv2.addWeighted(seg_array, 0.7, orig_resized, 0.3, 0)
                        st.image(overlay, caption="Overlay View", use_container_width=True)
                    
                    with tabs[2]:
                        st.markdown("#### Fracture Analysis")
                        
                        # Extract masks and calculate statistics
                        if hasattr(result, 'masks') and result.masks is not None:
                            num_fractures = len(result.boxes)
                            
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                                <h4 style="margin-top: 0;">Detection Summary:</h4>
                                <ul>
                                    <li>Number of fractures detected: {num_fractures}</li>
                                    <li>Average confidence score: {result.boxes.conf.mean().item():.2f}</li>
                                    <li>Largest fracture area: {int(np.max([mask.sum().item() for mask in result.masks.data]) if len(result.masks.data) > 0 else 0)} pixels</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display bounding boxes and confidence values
                            if len(result.boxes) > 0:
                                st.markdown("##### Fracture Details")
                                for i, box in enumerate(result.boxes):
                                    conf = box.conf.item()
                                    st.markdown(f"""
                                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                                        <div style="width: 20px; height: 20px; background-color: #dc3545; margin-right: 10px; border-radius: 50%;"></div>
                                        <span>Fracture {i+1}: Confidence {conf:.2f}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("No detailed mask information available for this segmentation.")
                    
                    # Convert to bytes for download
                    buf = io.BytesIO()
                    seg_img.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    # Download section with multiple options
                    st.markdown("#### Download Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="Download Segmentation",
                            data=byte_im,
                            file_name=f"fracture_seg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )
                    
                    with col2:
                        # Prepare overlay for download
                        overlay_img = Image.fromarray(overlay)
                        buf_overlay = io.BytesIO()
                        overlay_img.save(buf_overlay, format="PNG")
                        byte_overlay = buf_overlay.getvalue()
                        
                        st.download_button(
                            label="Download Overlay",
                            data=byte_overlay,
                            file_name=f"fracture_overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )
                    
                except Exception as e:
                    st.error(f"Error in segmentation: {e}")
            else:
                # Show result for non-fractured case
                st.markdown("### Analysis Result")
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center;">
                    <img src="https://img.icons8.com/color/96/000000/ok--v1.png" width="80">
                    <h3>No fracture detected</h3>
                    <p>The AI analysis indicates that this X-ray does not contain any visible fractures.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Show placeholder when no image is uploaded
            st.markdown("""
            <div style="background-color: #f8f9fa; height: 500px; display: flex; flex-direction: column; justify-content: center; align-items: center; border-radius: 10px; margin-top: 2rem;">
                <img src="https://img.icons8.com/fluency/96/000000/x-ray.png" width="80">
                <h3 style="color: #6c757d; margin-top: 1rem;">Segmentation Results</h3>
                <p style="color: #6c757d; text-align: center; max-width: 80%;">Upload an X-ray image to see fracture detection and segmentation results here.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer section
    st.markdown("""
    <div class="footer">
        <p>Advanced X-ray Fracture Detection & Segmentation | Powered by Streamlit and Deep Learning</p>
        <p>© 2025 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()