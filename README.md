# X-ray Fracture Detection and Segmentation App

This Streamlit application performs two-step analysis on X-ray images:
1. Classification of X-ray images as "Fractured" or "Non-fractured"
2. Segmentation of fractured areas for images classified as "Fractured"

## File Structure
```
fracture-detection-app/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── models/                 # Directory for model files
│   ├── classifier.pth      # ResNet18 classification model
│   └── xray-seg.pt         # YOLO segmentation model
```

## Setup Instructions

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Place your trained models in the `models` directory:
   - Place the classification model (`classifier.pth`) in the `models` directory
   - Place the segmentation model (`xray-seg.pt`) in the `models` directory

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Open the application in your web browser (typically at http://localhost:8501)
2. Upload an X-ray image using the file uploader
3. The application will first classify the image as "Fractured" or "Non-fractured"
4. If the image is classified as "Fractured", the application will perform segmentation to highlight the fractured area
5. The segmentation result can be downloaded using the provided button

## Model Information

- **Classification Model**: ResNet18 fine-tuned to detect fractures in X-ray images
- **Segmentation Model**: YOLO-based model for segmenting fractured areas in X-ray images