# model-training

This repository implements the training pipeline for a sentiment classification model using restaurant review data. It uses a reusable preprocessing library (`lib-ml`) and saves a versioned model for later use in model-service and orchestration.

## Overview

The training pipeline loads labeled review data, applies text preprocessing, trains a logistic regression classifier, and saves the trained model. Preprocessing is factored out to the separate `lib-ml` repository for consistency and reuse across components.

The repository includes a GitHub Actions workflow that automatically trains and releases the model when a version tag is pushed.

## Repository Structure

- `src/train.py`: main training script
- `data/`: contains the training dataset
- `models/`: output directory for the trained model
- `requirements.txt`: Python dependencies
- `.github/workflows/model-training.yml`: CI workflow that builds and releases the model


