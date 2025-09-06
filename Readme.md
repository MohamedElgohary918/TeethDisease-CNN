# Teeth Disease Classification using CNN

This project applies **Convolutional Neural Networks (CNNs)** to classify different types of teeth diseases using image data.  
We trained and evaluated a deep learning model on a custom dataset and achieved **97% accuracy** on the test set.

---

## 🧠 CNN Model Structure

The CNN was built using **TensorFlow/Keras**.  
It includes convolutional layers, batch normalization, max pooling, and fully connected layers with dropout for regularization.  
We replaced the Flatten layer with **GlobalAveragePooling2D** to reduce overfitting.  

### Architecture:
- **Conv2D → BatchNormalization → MaxPooling2D** (Block 1)  
- **Conv2D → BatchNormalization → MaxPooling2D** (Block 2)  
- **Conv2D → BatchNormalization → MaxPooling2D** (Block 3)  
- **Flatten**  
- **Dense (ReLU) + Dropout**  
- **Dense (Softmax)**  

---

## 📊 Results

- **Test Accuracy:** `97%`  
- **Classification Report:**  
  - Precision, Recall, F1-score all around `0.97`  
- **Confusion Matrix:**  

![Confusion Matrix](Confusion_Matrix.png)

---

## 🚀 How to Use

1. Open `Build_CNN_Model.ipynb` to train the CNN model.  
2. The trained model is saved as `model_Teeth.keras`.  
3. Use `TeethDisease_Model_Evaluation.ipynb` to evaluate the model on test data.  
4. Replace images in `Testing/` with new ones to test different cases.  

---

## ✅ Conclusion

This CNN-based model provides a reliable approach to detect teeth diseases from images, achieving **state-of-the-art accuracy of 97%**. Future improvements can include transfer learning and larger datasets to improve generalization.

