# 🌱 Plant Village - AI Plant Disease Diagnosis

Welcome to **Plant Village**, a web-based project for detecting and analyzing plant diseases using AI. This project provides detailed reports, recommendations, and suggestions for supplements to help maintain plant health.

---

## 🚀 Features

- **AI Disease Detection**: Upload plant images and get accurate disease predictions using a trained model.  
- **Confidence Score**: Shows the confidence level of the AI prediction.  
- **Detailed PDF Reports**: Generates downloadable PDF reports containing:
  - Disease name  
  - Confidence score  
  - AI-generated recommendations  
  - Suggested supplements for the detected disease  
  - Plant image for reference  

- **Gemini API Integration (optional)**: Generates detailed textual reports for each disease (currently disabled to save quota).  
- **Demo Available**: Users can try the system via **FastAPI** or **Streamlit** interface.  
- **Market Page**: Displays relevant plant supplements with images and purchase links.  

---

## 🖥️ Technologies Used

- **Backend**: FastAPI  
- **Frontend**: HTML, CSS, Jinja2 Templates  
- **PDF Generation**: FPDF  
- **AI Model**: Pretrained PyTorch model (ResNet50)  
- **Image Processing**: PIL (Python Imaging Library)  
- **Markdown Rendering**: `markdown` Python library  
- **Streamlit Demo**: Simple web interface for uploading plant images and seeing results  


## 📷 Demo Screenshots

### Home page
![Home Page](https://github.com/HendRamadan1/Plant-Village/blob/main/Screenshot%20(111).png)

### Upload & Analyze a Plant
![Analysis Demo](https://github.com/HendRamadan1/Plant-Village/blob/main/Screenshot%20(112).png)

### Report / Generated PDF Report
![Report Demo](http://github.com/HendRamadan1/Plant-Village/blob/main/Screenshot%20(115).png)

---

## ⚡ How It Works

1. **Upload an Image**  
   Users upload a photo of a plant leaf or fruit.

2. **AI Prediction**  
   - The uploaded image is preprocessed and passed to the AI model (`ResNet50`).  
   - Model outputs the predicted disease class and confidence score.

3. **Generate Report**  
   - PDF report is automatically generated including:
     - Disease name  
     - Confidence  
     - Recommendations and suggested supplements  
     - Plant image  
   - Users can download the report.

4. **Market Suggestions**  
   - Related supplements for the detected disease are displayed with links to buy.

---

## 🛠️ Running the Project

### FastAPI Demo
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
---

pip install -r requirements.txt
uvicorn main:app --reload
