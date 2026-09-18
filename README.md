# MLP Project – Comment Category Prediction

A multi-class text classification project that predicts the correct **label** of an online comment using its text content, categorical features, numeric features, and time-based features.

## 📌 Objective

The goal of this project is to predict the correct **label** of a comment using:

- Comment text
- Categorical features
- Numeric features
- Time-based features

This is a **multi-class classification problem**.

## 🔁 Workflow

1. Load and understand the dataset
2. Perform basic EDA
3. Preprocess text and other features
4. Create text, numeric, and categorical features
5. Train **3 models**
6. Compare the models
7. Select the best model
8. Generate final submission

## 🎯 Final Aim

The purpose of this notebook is not only to train models, but also to **compare multiple approaches and choose the best one** for final prediction.

---

## 📂 Dataset

The competition files used are:

| File | Description |
|---|---|
| `train.csv` | Contains features and target label (198,000 rows × 15 columns) |
| `test.csv` | Contains features only (102,000 rows × 14 columns) |
| `sample_submission.csv` | Shows the required output format (102,000 rows × 2 columns) |

## 🧩 Features Used

**Text Feature**
- `comment`

**Categorical Features**
- `race`
- `religion`
- `gender`

**Numeric Features**
- `emoticon_1`
- `emoticon_2`
- `emoticon_3`
- `upvote`
- `downvote`
- `if_1`
- `if_2`
- `disability`

**Date Feature**
- `created_date`

**Target**
- `label`

---

## 🛠️ Data Preprocessing

- Fill missing text values
- Fill missing categorical values
- Fill missing numeric values
- Process date column
- Create time-based features (hour, day of week)

## 📊 Exploratory Data Analysis (EDA)

- **Target Distribution** – checked class balance to avoid biased models
- **Comment Length Distribution** – examined the spread of comment lengths
- **Numeric Feature Summary** – inspected numeric feature statistics
- **Average Comment Length by Label** – examined whether comment length varies by class

## 🔤 Feature Engineering

- **One-Hot Encoding** for categorical features (`race`, `religion`, `gender`)
- **Numeric Features** prepared: emoticon counts, upvotes/downvotes, indicator variables, disability, hour, day of week, comment length
- **Word-Level TF-IDF** – captures important words, useful word pairs, and word importance across documents
- **Character-Level TF-IDF** – captures spelling variation, repeated letters, short noisy patterns, and modified abusive words
- **Final Feature Matrix** – combines word TF-IDF, character TF-IDF, numeric features, and encoded categorical features into one matrix (giving the model both text and metadata information)

## ✂️ Train / Validation Split

- 85% training / 15% validation
- Stratified split to preserve class distribution in both sets
- Training shape: `(168300, 45033)`
- Validation shape: `(29700, 45033)`

---

## 🤖 Models Trained

### 1. Multinomial Naive Bayes
A simple and strong baseline for text classification.
- Fast to train
- Works well with sparse text features
- Good baseline for comparison

### 2. Linear SVC
A fast and effective model for text classification.
- Fast to train
- Works well with TF-IDF features
- Strong baseline for text data
- Good for comparison with other models

### 3. LightGBM
Used as the advanced model.
- Efficient and fast
- Good for large feature spaces
- Supports tuning and early stopping
- Can learn more complex patterns

**LightGBM configuration:**
```python
LGBMClassifier(
    objective="multiclass",
    num_class=num_classes,
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42
)
```

---

## 📈 Model Comparison

| Model | Validation Accuracy |
|---|---|
| MultinomialNB | 0.5193 |
| LinearSVC | 0.8012 |
| **LightGBM** | **0.9173** |

Comparing models on validation accuracy helps select the best model instead of choosing one without evidence.

## 🏆 Best Model Selection

**Best Model:** LightGBM
**Best Validation Accuracy:** 0.9173

### Why LightGBM Was Selected

The final model is selected based on validation performance because:

- Different models behave differently on text data
- A model should be selected based on result, not only on assumption
- Comparison makes the notebook stronger and more complete

### Validation Classification Report (LightGBM)

```
              precision    recall  f1-score   support

           0       0.98      0.95      0.96     17126
           1       0.73      0.85      0.79      2388
           2       0.88      0.90      0.89      9366
           3       0.66      0.72      0.69       820

    accuracy                           0.92     29700
   macro avg       0.81      0.86      0.83     29700
weighted avg       0.92      0.92      0.92     29700
```

---

## 🚀 Final Prediction & Submission

- The selected LightGBM model generates class probabilities on the test set
- A power transform (`p^1.1`) is applied to the probabilities and re-normalized before taking the final class via `argmax`
- Final predictions are written to `submission.csv` in the format required by `sample_submission.csv`

```python
test_proba = best_model.predict_proba(X_full_test)
test_proba = np.power(test_proba, 1.1)
test_proba = test_proba / test_proba.sum(axis=1, keepdims=True)
test_pred = np.argmax(test_proba, axis=1)
```

---

## 🧰 Tech Stack

- **Python 3**
- `numpy`, `pandas` – data handling
- `matplotlib` – visualization
- `scikit-learn` – TF-IDF, train/test split, Naive Bayes, Linear SVC, metrics
- `lightgbm` – gradient boosting classifier
- `scipy` – sparse matrix operations

## 📁 Repository Structure

```
.
├── notebook.ipynb        # Full analysis, preprocessing, modeling, and submission pipeline
├── submission.csv        # Final predictions (generated by the notebook)
└── README.md             # Project documentation
```

## ▶️ How to Run

1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scipy scikit-learn lightgbm
   ```
2. Place `train.csv`, `test.csv`, and `sample_submission.csv` in the expected input directory.
3. Run the notebook cells in order from top to bottom.
4. The final predictions will be saved to `submission.csv`.

## 📝 Conclusion

By combining text features (word & character TF-IDF), numeric features, categorical encodings, and time-based features, and by systematically comparing three different models, **LightGBM** was identified as the best-performing model with a **validation accuracy of ~91.7%**, and was used to generate the final submission.
