# Results of Random Forest & Neural Network Models

## Random Forest Model Results

The evaluation with `max_depth: 40` and `n_estimators: 500` yielded an accuracy of 98.25%, with the process completed in 1.5 minutes.

### Fine-Tuning (GridSearchCV)

GRIDSEARCHCV Results:

Best parameters found: {'max_depth': 70, 'n_estimators': 1000}
Best accuracy achieved: 98.08%

RandomForestClassifier configuration:
- max_depth=70
- n_estimators=1000
- random_state=42
- verbose=2

Model accuracy: 98.29%

## Neural Network Model Results

### Training Process

Trained over 50 epochs, the neural network model showed significant improvements in accuracy and loss reduction across both training and validation datasets.

### Final Epoch Results (Epoch 50)

#### Training Results
- **Accuracy:** 99.19%
- **Loss:** 0.0273

#### Validation Results
- **Accuracy:** 99.02%
- **Loss:** 0.0370

### Test Dataset Evaluation

The model demonstrated a strong generalization capability on the test dataset, achieving an accuracy of 99.12% and a loss of 0.0312.

### Prediction Accuracy

Predictions on a small sample from the test dataset matched the true labels exactly, showcasing its precision:
- **True Labels:** [9 7 9]
- **Predictions:** [9 7 9]

## Comparison and Conclusion

Both models demonstrated high accuracy and effective classification of handwritten digits from the EMNIST dataset. The Random Forest model achieved an impressive accuracy of 98.25% in just 1.5 minutes, making it a robust and time-efficient choice for this classification task.

The Neural Network model, on the other hand, reached a slightly higher accuracy of 99.12% on the test dataset, showing exceptional generalization capabilities and precision. Although it requires a longer training time over 50 epochs, the investment in time is justified by its performance and accuracy.

Given the results, the preference between the two models depends on the specific requirements of the task. For tasks where time efficiency is critical, the Random Forest model is preferable due to its shorter training time. However, for applications where the highest possible accuracy is paramount, the Neural Network model, with its superior generalization and precision, would be the preferred choice.
