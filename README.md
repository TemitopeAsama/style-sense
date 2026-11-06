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

### Repository files

| File | Purpose |
| --- | --- |
| `starter/starter.ipynb` | Data exploration, preprocessing, training, tuning, and final evaluation. |
| `starter/data/reviews.csv` | Customer reviews used for training and evaluation. |
| `starter/README.md` | Original instructions describing the starter folder. |
| `requirements.txt` | Python packages required to run the notebook. |
| `.gitignore` | Patterns for local files excluded from version control. |
| `CODEOWNERS` | Repository code ownership configuration. |
| `LICENSE.txt` | License and usage terms for the educational content. |

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
- **Holdout evaluation:** Reserves 10% of the data for testing using `random_state=27`. After tuning finishes, evaluates `grid_search.best_estimator_` on this held-out set. Reports accuracy and per-class precision, recall, and F1 score, plus a confusion matrix.
- **Cross-validation:** Uses five folds on the training data to measure accuracy across different splits. Preprocessing stays inside the pipeline so each fold fits its transformations on its own training subset.
- **Hyperparameter search:** Uses five-fold `GridSearchCV` to compare logistic regression's `C` values of `0.1`, `1.0`, and `10.0`, selecting the best estimator by accuracy.

- **Missing-value prediction check:** Predicts on synthetic reviews containing `NaN`, `None`, and `pandas.NA` across feature types, an entirely missing row, and an unseen category. Checks that the fitted pipeline returns valid labels without errors.

About 82% of reviews recommend the product, making per-class metrics useful alongside accuracy. The final test metrics are reported in the notebook's **Final Test Evaluation** section, after model selection is complete.

## Project Instructions

Complete the workflow in [starter/starter.ipynb](starter/starter.ipynb):

1. Load the reviews and separate the eight input features from `Recommended IND`.
2. Explore the data, including numerical summaries, categorical values, missing values, and class balance.
3. Build preprocessing pipelines for each feature type:
   - Normalize null markers, impute numerical features with training-set medians, and scale with `StandardScaler`.
   - Normalize null markers, impute categorical features with training-set modes, and encode with `OneHotEncoder(handle_unknown='ignore')`. Keep `Clothing ID` categorical.
   - Fill missing titles and review bodies with empty strings inside the text pipeline before TF-IDF, capped at 1,000 features per text column. Use a custom spaCy tokenizer to lemmatize and lowercase tokens and remove stop words, punctuation, and whitespace.
4. Combine preprocessing with `ColumnTransformer` and add a `LogisticRegression` classifier in a single `Pipeline`.
5. Train the pipeline and run five-fold cross-validation using only the training split.
6. Tune the classifier with `GridSearchCV` on the training data, then evaluate the best estimator on the held-out test set.
7. Verify that the fitted pipeline can predict on synthetic reviews with missing features and an unseen category.

## Built With

- Python — Programming language used for the analysis and model.
- pandas — Data loading and exploration.
- scikit-learn — Preprocessing, TF-IDF features, classification, and model selection.
- spaCy — English text tokenization and lemmatization.
- Matplotlib and seaborn — Evaluation plots, including the confusion matrix.
- Jupyter Notebook — Interactive analysis and documentation.

## License

See [LICENSE.txt](LICENSE.txt) for the Udacity educational content license and usage terms.
