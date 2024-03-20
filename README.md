# EMNIST Digit Classifier

This project is an exploration into machine learning with a focus on classifying handwritten digits from the Extended Modified National Institute of Standards and Technology (EMNIST) dataset. Utilizing both the RandomForestClassifier algorithm and a Neural Network model, this Python-based project demonstrates the process of loading and preparing data, selecting, training, and finally evaluating their performance.

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
pip install pandas numpy scikit-learn matplotlib tensorflow
```

#### Running the Project
To run the RandomForest model, execute the following command:

```bash
python RandomForest_Model.py
```

To run the Neural Network model, execute the following command:
```bash
python NeuralNetwork_Model.py
```

This script will:

* Load and prepare the EMNIST dataset.
* Train the specified model (RandomForestClassifier or Neural Network).
* Evaluate the model's accuracy on the test dataset.

## Data Download

The datasets for this project can be downloaded from the following Google Drive links:

- Train set: [Download Train Set](https://drive.google.com/file/d/11sxm4A06o9qsFqBG_03_zWQM0sKW6STA/view)
- Test set: [Download Test Set](https://drive.google.com/file/d/1HbRM6M6IH7vBmlAQMv5P1F8bXfypRMKP/view?usp=share_link)

Please download the datasets and place them in the appropriate directory before running the project scripts.

# Project Structure
* emnist-digits-train.csv: Training data file.
* emnist-digits-test.csv: Test data file.
* RandomForest_Model: Main Python script for training and evaluating the **random forest model**
* NeuralNetwork_Model: Main Python script for training and evaluating the **neural network model**

# Built With

* Pandas - For data manipulation and analysis.
* NumPy - For handling large, multi-dimensional arrays and matrices.
* Scikit-learn - For machine learning.
* Matplotlib - For creating static, interactive, and animated visualizations in Python.
* TensorFlow and Keras - For building and training the neural network model.
