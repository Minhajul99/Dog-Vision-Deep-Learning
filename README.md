# 🐶 Dog Breed Classification Using Deep Learning

An end-to-end multi-class image classification project that identifies dog breeds from photos using TensorFlow 2.x and Transfer Learning via TensorFlow Hub.

---

## Problem Statement

> Given a photo of a dog, can we identify its breed?

With 120 possible breeds to classify, this is a challenging fine-grained image recognition problem — solved using transfer learning on a pretrained MobileNetV2 model.

---

## Dataset

- **Source:** [Kaggle — Dog Breed Identification Competition](https://www.kaggle.com/c/dog-breed-identification/data)
- **Training set:** 10,000+ labeled images across 120 dog breeds
- **Test set:** 10,000+ unlabeled images (predictions submitted to Kaggle)

---

## Project Workflow

1. **Data Preparation** — Load labels, build image file paths, encode breed labels as boolean arrays
2. **Image Preprocessing** — Resize images to 224×224, normalize pixel values (0–1), convert to Tensors
3. **Batching** — Create TensorFlow data pipelines with shuffled training batches (batch size = 32)
4. **Model Building** — Transfer learning with MobileNetV2 + custom Dense output layer (120 classes, softmax)
5. **Training** — Train on a 1,000-image subset first, then scale to the full dataset
6. **Evaluation** — Visualize predictions, confidence scores, and top-10 breed probabilities
7. **Prediction** — Generate predictions on test data and custom photos

---

## Model Architecture

| Layer | Details |
|---|---|
| Input | Images of shape (224, 224, 3) |
| Base Model | MobileNetV2 (pretrained on ImageNet via TensorFlow Hub) |
| Output Layer | Dense(120, activation='softmax') |
| Loss | Categorical Crossentropy |
| Optimizer | Adam |

Transfer learning is used — the MobileNetV2 weights are frozen, and only the output Dense layer is trained on the dog breed data.

---

## Training Details

- **Epochs:** Up to 100 (with Early Stopping, patience=3)
- **Callbacks:** TensorBoard (training visualization) + EarlyStopping (prevents overfitting)
- **Subset training:** Model first validated on 1,000 images (800 train / 200 val), then trained on full 10,000+ image dataset
- **Hardware:** GPU (Google Colab)

---

## Results

- Achieved **~65% validation accuracy** on the 1,000-image subset after ~26 epochs
- Accuracy improves significantly when trained on the full dataset
- Predictions exported as probability scores across all 120 breeds for Kaggle submission

---

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow 2.x, TensorFlow Hub, Keras
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Platform:** Google Colab (GPU)

---

## Repository Structure

```
Dog-Vision-Deep-Learning/
├── Dog vision.ipynb                              # Main notebook
├── labels.csv                                    # Training image labels
├── preds_array.csv                               # Saved test predictions
├── full_model_predictions_submission_1_...csv    # Kaggle submission file
├── models/                                       # Saved model files (.h5)
├── logs/                                         # TensorBoard training logs
└── my-dog-photos/                                # Custom images for prediction
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Minhajul99/Dog-Vision-Deep-Learning.git
   ```

2. Download the dataset from [Kaggle](https://www.kaggle.com/c/dog-breed-identification/data) and place it in your Google Drive under `Dog Vision/`.

3. Open `Dog vision.ipynb` in Google Colab and connect to a GPU runtime:
   `Runtime → Change runtime type → GPU`

4. Run all cells in order.

---

## Key Features

- **Transfer Learning** with MobileNetV2 — avoids training from scratch on limited data
- **TensorBoard integration** for real-time training monitoring
- **Early Stopping** to prevent overfitting automatically
- **Custom image predictions** — test the model on your own dog photos
- **Kaggle submission** — outputs properly formatted prediction CSV
