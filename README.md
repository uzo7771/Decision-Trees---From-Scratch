# Decision Tree Model
This project implements a Decision Tree algorithm in Python, focusing on classification tasks. The tree is built using recursive binary splits based on information gain.

## Features:
- Recursive construction of a decision tree
- Information gain-based node splits
- Feature importance calculation
- Hyperparameter tuning (`max_depth`, `min_samples_split`)

## How it works:
1. The algorithm calculates the entropy of the target variable `y`.
2. It splits the dataset based on different thresholds for each feature.
3. The split that results in the maximum information gain is chosen.
4. This process is recursively applied to subtrees, limited by parameters like `max_depth` and `min_samples_split`.
