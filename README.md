# EMNIST Digit Classifier

This project is an exploration into machine learning with a focus on classifying handwritten digits from the Extended Modified National Institute of Standards and Technology (EMNIST) dataset. Utilizing the RandomForestClassifier algorithm, this Python-based project demonstrates the process of loading and preparing data, selecting, training, and fine-tuning a model, and finally evaluating its performance.

## Getting Started

These instructions will guide you through setting up and running the project on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following software installed on your system:

- Python 3.x
- Pip (Python package installer)

### Installation

First, clone the repository to your local machine:

```bash
git clone https://github.com/<your_username>/<project_name>.git
cd <project_name>
```

Then, install the required Python packages:
```bash
pip install pandas numpy scikit-learn
```

Running the Project
Execute the following command to run the project:
```bash
python digit_classifier.py
```

This script will:

* Load and prepare the EMNIST dataset.
* Train a RandomForestClassifier model.
* Fine-tune the model using GridSearchCV.
* Evaluate the model's accuracy on the test dataset.


# Project Structure
* emnist-digits-train.csv: Training data file.
* emnist-digits-test.csv: Test data file.
* digit_classifier.py: Main Python script for training and evaluating the model.

# Built With

* Pandas - For data manipulation and analysis.
* NumPy - For handling large, multi-dimensional arrays and matrices.
* Scikit-learn - For machine learning.

