# The Performance metrics used in Machine Learning

## Classification Metrics:

- **Accuracy:**
  - Formula: $Accuracy = \frac{Number\ of\ Correct\ Predictions}{Total\ Number\ of\ Predictions}$
  - Useful for balanced datasets but can be misleading for imbalanced ones.
  
- **Precision:**
  - Formula: $Precision = \frac{True\ Positives}{True\ Positives\ +\ False\ Positives}$
  - Measures the accuracy of positive predictions and is sensitive to false positives.
  
- **Recall (Sensitivity):**
  - Formula: $Recall = \frac{True\ Positives}{True\ Positives\ +\ False\ Negatives}$
  - Measures the ability of the model to find all relevant cases within a dataset.
  
- **F1 Score:**
  - Formula: $F1\ Score = 2 \times \frac{Precision \times Recall}{Precision + Recall}$
  - Harmonic mean of precision and recall, balances both metrics.

- **ROC Curve and AUC:**
  - ROC Curve plots True Positive Rate (Recall) against False Positive Rate at various thresholds.
  - Area Under the Curve (AUC) represents the model's ability to distinguish between positive and negative classes.
  
- **Confusion Matrix:**
  - Provides a breakdown of model predictions, highlighting true positives, true negatives, false positives, and false negatives.


## Regression Metrics:

- **Mean Absolute Error (MAE):**
  - Formula: $MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
  - Represents the average absolute difference between predicted and actual values.
  
- **Mean Squared Error (MSE):**
  - Formula: $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
  - Squares the errors, giving more weight to large errors.
  
- **Root Mean Squared Error (RMSE):**
  - Formula: $RMSE = \sqrt{MSE}$
  - Provides interpretable scale and is useful for comparing with the target variable's range.
  
- **R-squared (Coefficient of Determination):**
  - Formula: $R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$
  - Measures the proportion of variance explained by the model. A higher value indicates a better fit.


## Clustering Metrics:

- **Silhouette Score:**
  - Measures the compactness and separation of clusters, ranges from -1 to 1.
  
- **Davies-Bouldin Index:**
  - Computes the average similarity measure between each cluster and its most similar one, lower values indicate better clustering.
  
- **Adjusted Rand Index (ARI):**
  - Measures the similarity between two clusterings, adjusting for chance.

## Cross-Validation:

- **K-Fold Cross-Validation:**
  - Divides the dataset into k subsets, trains the model on k-1 folds, and validates on the remaining fold, repeating the process k times.
  
- **Leave-One-Out Cross-Validation:**
  - Special case of k-fold where k equals the number of samples, leaving one sample for validation.

## Hyperparameter Tuning Metrics:

- **Grid Search:**
  - Exhaustively searches through a specified parameter grid to find the optimal combination.
  
- **Random Search:**
  - Randomly samples from a specified parameter distribution, useful for a large hyperparameter space.

In conclusion, understanding and applying appropriate metrics are essential for evaluating machine learning models effectively and making informed decisions in model selection, optimization, and deployment.
