# model-training

**Overview**
This repository contains the training pipeline for the Restaurant Sentiment Analysis model. It preprocesses data, trains the model, and stores the trained model in a versioned manner for use in the `model-service`. Once the model is trained, upload the model through GitHub Release.

#### **Features**

- Preprocesses restaurant review data using shared logic from `lib-ml`.
- Trains a sentiment analysis model.
- Stores trained models with versioning for traceability.

#### **Setup**

1. Clone the repository:

   ```
   git clone https://github.com/remla2025-team16/model-training.git
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the training pipeline:

   ```
   python train.py
   ```

   The trained model will be saved to `models/` with a version tag (e.g., `model_v1.0.0.pkl`).

#### **Dependencies**

- Requires `lib-ml` for preprocessing (installed via package manager).
