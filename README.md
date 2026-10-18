# MSCS 634 - Lab 4: Regression Analysis with Regularization Techniques

## Purpose
This lab compares several regression models on the Diabetes dataset from scikit-learn (442 patients, 10 health features, and a target that measures disease progression). I built simple linear, multiple linear, polynomial, Ridge, and Lasso models, then compared them with MAE, MSE, RMSE, and R². All models use the same 80/20 train/test split (`random_state=42`) so the comparison is fair.

## Results (test set)

| Model | RMSE | R² |
|---|---|---|
| Ridge + Polynomial degree 3 (alpha = 0.001) | 50.91 | 0.5108 |
| Lasso + Polynomial degree 3 (alpha = 0.001) | 51.43 | 0.5007 |
| Lasso (alpha = 0.1) | 52.90 | 0.4719 |
| Ridge (alpha = 0.1) | 53.45 | 0.4609 |
| Multiple Linear Regression | 53.85 | 0.4526 |
| Polynomial degree 2 | 55.64 | 0.4156 |
| Simple Linear Regression (BMI only) | 63.73 | 0.2334 |
| Polynomial degree 3 | 181.42 | -5.2120 |

## Key Insights
- Using all 10 features (R² 0.45) was much better than using BMI alone (R² 0.23).
- Plain polynomial regression overfit quickly. Degree 3 reached a training R² of 0.86 but a test R² of -5.21, and degree 4 memorized the training data completely.
- Ridge and Lasso fixed the overfitting. With a small penalty, the same degree 3 features gave the best results of all: Ridge R² 0.5108 and Lasso R² 0.5007.
- Ridge shrinks all coefficients but keeps every feature. Lasso sets some coefficients to exactly zero, so it also selects features. At alpha = 1, Lasso kept only `bmi`, `bp`, and `s5`.
- Even the best model explains only about half of the variation, so these 10 measurements do not tell the whole story. `s5`, `bmi`, and `bp` were the most useful features.

## Challenges and Decisions
- **Choosing the feature for simple regression:** I used `bmi` because it is a well-known health indicator. In a quick check, `s5` actually scored higher alone (R² about 0.38).
- **Polynomial degrees:** On all 10 features, degree 4 creates 1000 features for only 353 training rows, which makes the overfitting very easy to see. I stopped at degree 4. I left the plain degree 3 model out of the bar chart so its huge error would not squash the other bars.
- **Correlated features:** `s1` and `s2` are closely related blood measures, which gave them large opposite coefficients in plain regression. Ridge and Lasso made them smaller and more stable.
- **Choosing alpha:** I picked the best alpha by looking at test results because the dataset is small. A cleaner method would be cross-validation on the training data. The scores may be slightly optimistic, and results come from a single split.
- **Lasso warnings:** Lasso on the degree 3 features showed convergence warnings, so I raised `max_iter`. The results were still fine.

## Files
- `MSCS_634_Lab_4.ipynb` - notebook with all code, plots, and explanations
- `README.md` - this file
