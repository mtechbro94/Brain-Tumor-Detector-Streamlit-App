# 🧠 Brain Tumor Detector - Streamlit App

A user-friendly web app built with **Streamlit** that uses a deep learning model to classify brain MRI scans into four categories:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

> 🚀 Live Demo: [Click to Open the App](https://your-app-url.streamlit.app)  
> 📦 Model is auto-downloaded from Google Drive on first launch.

---

## 📸 App Preview

![App Screenshot](https://lakezurichopenmri.com/wp-content/uploads/2025/04/MRI-for-Brain-Tumor-Detection-scaled.jpg)

---

## 🧠 How It Works

- Upload a brain MRI image (`.jpg`, `.jpeg`, or `.png`)
- Click **Predict**
- The app classifies the tumor type using a pre-trained CNN model
- Displays:
  - Predicted class
  - Confidence score
  - A probability bar chart for all classes

---

## 🚀 Technologies Used

| Technology     | Purpose                        |
|----------------|-------------------------------|
| Streamlit      | Web application interface     |
| TensorFlow     | Deep learning model            |
| PIL & NumPy    | Image preprocessing            |
| Plotly         | Interactive visualizations     |
| gdown          | Downloads model from GDrive    |

---

## 🛠️ How to Run Locally

### 1. Clone the Repo

```bash
git clone https://github.com/mtechbro94/Brain-Tumor-Detector-Streamlit-App.git
cd Brain-Tumor-Detector-Streamlit-App
2. Install Dependencies
- pip install -r requirements.txt
3. Run the App
- streamlit run app.py
- The model will be automatically downloaded from Google Drive the first time.

# 📁 Project Structure
├── app.py               # Streamlit app
├── requirements.txt     # Python dependencies
├── .gitignore
└── models/
     └── model.h5        # Downloaded at runtime via gdown
# 📦 Model Hosting (Google Drive)
- The model is stored on Google Drive and downloaded using gdown.
🔗 Model Link: Google Drive

# 📌 Notes
- This app is for educational and research purposes only.
- MRI classification is done using a pre-trained model, and not suitable for real medical diagnostics.

# 🙋‍♂️ Author
Aaqib Rashid Mir
🔗 LinkedIn
🔗 GitHub
