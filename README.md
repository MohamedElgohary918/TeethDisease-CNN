# 🦷 Teeth Disease Classification using CNN & Transfer Learning

This project applies **Deep Learning** to classify different types of teeth diseases using image data.  
We built and evaluated two approaches:  
1. A **CNN model from scratch**  
2. A **Pretrained DenseNet121 model (transfer learning)**  

Finally, we deployed the CNN model using the **Streamlit library** to provide an interactive demo.  

---

## 📂 Project Structure
```
Teeth_diseases_Classification/
│── README.md
│── Confusion_Matrix.png
│── Dental_AI_App.png   # Streamlit demo screenshot (ignored in repo)
│
├── Deployment/
│   ├── app.py
│   └── model_Teeth.keras
│
├── Model_from_Scratch/
│   ├── Build_CNN_Model.ipynb
│   ├── TeethDisease_Model_Evaluation.ipynb
│   └── Model/
│       └── model_Teeth.keras
│
├── Pretrained_Model/
│   ├── Transfer_Learning.ipynb
│   └── model.dense121.h5
```

---

## 🧠 CNN Model (from Scratch)

The CNN was built using **TensorFlow/Keras**.  
It includes convolutional layers, batch normalization, max pooling, and fully connected layers with dropout for regularization.  
We replaced the Flatten layer with **GlobalAveragePooling2D** to reduce overfitting.  

### Architecture:
- **Conv2D → BatchNormalization → MaxPooling2D** (Block 1)  
- **Conv2D → BatchNormalization → MaxPooling2D** (Block 2)  
- **Conv2D → BatchNormalization → MaxPooling2D** (Block 3)  
- **Flatten / GlobalAveragePooling2D**  
- **Dense (ReLU) + Dropout**  
- **Dense (Softmax)**  

---

## 📊 Results

### CNN (from Scratch)
- **Test Accuracy:** `97%`  
- **Classification Report:** Precision, Recall, F1-score ≈ `0.97`  
- **Confusion Matrix:**  

![Confusion Matrix](Confusion_Matrix.png)

### Pretrained DenseNet121
- **Test Accuracy:** `88%`  
- Transfer learning improved training time and stability but did not outperform the scratch CNN in this dataset.

---

## 🚀 Deployment with Streamlit

We deployed the **CNN model from scratch** as a web app using **Streamlit**.  
The app allows users to upload dental images and predicts one of **7 classes**:  

```
CaS | CoS | Gum | MC | OC | OLP | OT
```

### Run the app
```bash
cd Deployment
streamlit run app.py
```

### Demo Screenshot
*(File ignored in GitHub, but here’s the result preview)*  

![Dental AI App](Dental_AI_App.png)

---

## ✅ Conclusion

- **Scratch CNN model** achieved **97% accuracy** and showed the best results.  
- **DenseNet121 pretrained model** reached **88% accuracy** with decent performance.  
- **Streamlit deployment** demonstrates real-time predictions, making this approach usable in clinical workflows.  

**Future Work:**  
- Expand dataset with more samples.  
- Experiment with other pretrained models (e.g., EfficientNet, ResNet).  
- Integrate explainability (Grad-CAM) for better clinical trust.  
