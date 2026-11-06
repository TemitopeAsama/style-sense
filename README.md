# StyleSense - Customer Recommendation Prediction

A machine learning pipeline that predicts whether a customer recommends a clothing product using numerical, categorical, and review text features. This Udacity Data Scientist Nanodegree project combines feature preprocessing and logistic regression in a scikit-learn pipeline, with model evaluation, cross-validation, and hyperparameter tuning.

## Getting Started

The analysis and model are implemented in [starter/starter.ipynb](starter/starter.ipynb). The supplied dataset, [starter/data/reviews.csv](starter/data/reviews.csv), contains 18,442 anonymized reviews with no missing values.

The target column, `Recommended IND`, is `1` when a customer recommends the product and `0` otherwise. The eight input features are:

| Type | Features |
| --- | --- |
| Numerical | `Age`, `Positive Feedback Count` |
| Categorical | `Clothing ID`, `Division Name`, `Department Name`, `Class Name` |
| Text | `Title`, `Review Text` |

### Dependencies

Use Python 3 and the packages listed in [requirements.txt](requirements.txt):

- scikit-learn
- pandas
- seaborn
- Matplotlib
- spaCy
- Jupyter Notebook

The notebook also requires spaCy's `en_core_web_sm` English language model, which is downloaded separately during installation. Package versions are not pinned, so results may vary between environments.

### Installation

1. Download or clone this repository and open a terminal in its root directory.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, use `.venv\Scripts\activate` to activate the environment.

3. Install the dependencies and spaCy model:

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. Launch the notebook from the `starter/` directory:

```bash
cd starter
jupyter notebook starter.ipynb
```

Select the virtual environment's Python kernel and run the cells in order. The notebook expects its working directory to be `starter/` because it loads `data/reviews.csv` using a relative path.

## Testing

This project uses notebook-based model evaluation and does not include an automated test suite. To reproduce the analysis, restart the notebook kernel and run all cells in order after completing installation.

Cross-validation and grid search repeat the text preprocessing for each fit and may take time to finish.

### Break Down Tests

- **Data inspection:** Checks feature types, distributions, missing values, and target class balance.
- **Holdout evaluation:** Reserves 10% of the data for testing using `random_state=27`. A classification report measures precision, recall, and F1 score, while a confusion matrix shows correct predictions and errors for each class.
- **Cross-validation:** Uses five folds on the training data to measure accuracy across different splits. Preprocessing stays inside the pipeline so each fold fits its transformations on its own training subset.
- **Hyperparameter search:** Uses five-fold `GridSearchCV` to compare logistic regression's `C` values of `0.1`, `1.0`, and `10.0`, selecting the best estimator by accuracy.

The saved notebook outputs report approximately **90% baseline test accuracy**, **90.27% mean cross-validation accuracy**, and a best parameter of **`classifier__C = 1.0`**. Test-set F1 scores are **0.71** for non-recommendations and **0.94** for recommendations. About 82% of reviews recommend the product, making per-class metrics useful alongside accuracy.

These figures come from the saved outputs of a previous run. The notebook selects the tuned estimator but does not include a separate test-set evaluation of that estimator.

## Project Instructions

Complete the workflow in [starter/starter.ipynb](starter/starter.ipynb):

1. Load the reviews and separate the eight input features from `Recommended IND`.
2. Explore the data, including numerical summaries, categorical values, missing values, and class balance.
3. Build preprocessing pipelines for each feature type:
   - Scale numerical features with `StandardScaler`.
   - Encode categorical features with `OneHotEncoder(handle_unknown='ignore')`.
   - Process titles and review bodies separately with TF-IDF, capped at 1,000 features per text column. Use a custom spaCy tokenizer to lemmatize and lowercase tokens and remove stop words, punctuation, and whitespace.
4. Combine preprocessing with `ColumnTransformer` and add a `LogisticRegression` classifier in a single `Pipeline`.
5. Train the pipeline on the training split and evaluate its predictions on the test split.
6. Run five-fold cross-validation and tune the classifier with `GridSearchCV`.

## Built With

- Python — Programming language used for the analysis and model.
- pandas — Data loading and exploration.
- scikit-learn — Preprocessing, TF-IDF features, classification, and model selection.
- spaCy — English text tokenization and lemmatization.
- Matplotlib and seaborn — Evaluation plots, including the confusion matrix.
- Jupyter Notebook — Interactive analysis and documentation.

## License

See [LICENSE.txt](LICENSE.txt) for the Udacity educational content license and usage terms.
