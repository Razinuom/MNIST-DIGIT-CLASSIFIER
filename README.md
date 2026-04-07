# 🧠 Handwritten Digit Classifier

A machine learning web app that predicts handwritten digits (0–9) using a neural network trained on the MNIST dataset.

---

## 🚀 Features

- Upload your own digit image  
- Real-time prediction using a trained model  
- Automatically handles both black-on-white and white-on-black images  
- Simple and clean web interface built with Streamlit  

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- Streamlit  
- NumPy  
- Pillow  

---

## 🖼️ Demo

(Add a screenshot of your app here)

---

## 📦 Requirements

Install all dependencies with:

pip install -r requirements.txt

---

## ▶️ How to Run

1. Clone the repository:
git clone https://github.com/razinuom/MNIST-DIGIT-CLASSIFIER.git  
cd digit-classifier  

2. Install dependencies:
pip install -r requirements.txt  

3. Run the app:
streamlit run app.py  

---

## 🧠 How It Works

- The model is trained on the MNIST dataset of handwritten digits  
- Images are resized to 28×28 and normalised  
- The app preprocesses uploaded images to match the training format  
- A neural network predicts the digit using softmax probabilities  

---

## 📌 Notes

- The model performs best with clear, centered digits  
- Works with both black and white backgrounds (auto-adjusted)

---

## 📈 Future Improvements

- Add drawing canvas input  
- Show prediction confidence  
- Improve model accuracy  
- Deploy the app online  

---

## 🙌 Acknowledgements

- MNIST dataset  
- TensorFlow & Keras  
- Streamlit  
