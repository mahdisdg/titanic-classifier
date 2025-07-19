# Titanic Classifier

## Project Overview
A Random Forest model to predict survival of Titanic passengers using a cleaned subset of features.

## Project Structure

titanic-classifier/
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
│ └── titanic_exploration.ipynb
├── src/
│ ├── data_preparation.py
│ └── train_model.py
├── models/
│ └── model.pkl
├── .gitignore
├── requirements.txt
└── README.md


## Installation

```bash
pip install -r requirements.txt

## Usage

1. Prepare data

```bash
python src/data_preparation.py

2. Train model

```bash
python src/train_model.py

3. Explore data
Open ``notebooks/titanic_exploration.ipynb`` in Jupyter.

4. Results
Test Accuracy: 0.762