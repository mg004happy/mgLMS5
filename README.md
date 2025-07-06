# House Price Prediction

This project uses machine learning models (Random Forest and XGBoost) to predict house prices based on the Ames Housing dataset.
1. **Load Training Data**
   - `train.csv` is loaded using Pandas.
   - `SalePrice` is the target variable.
   - The `Id` column is dropped.

2. **Preprocessing**
   - Categorical variables are encoded via `pd.get_dummies()`.
   - All numeric features are standardized using `StandardScaler`.

3. **Train-Validation Split**
   - The dataset is split into 60% train and 40% validation.

4. **Model Training & Evaluation**
   - **Random Forest Regressor**:
     - `n_estimators=123`
     - Evaluated using Root Mean Squared Error (RMSE).
   - **XGBoost Regressor**:
     - `n_estimators=100`
     - `learning_rate=0.1`
     - Evaluated using RMSE.
   - Both models are compared to choose the better-performing one.

5. **Test Set Prediction**
   - `test.csv` is loaded.
   - The same preprocessing and scaling are applied.
   - XGBoost is retrained on the **full training data** (train + validation) for final predictions.
   - Predicted house prices are saved in `submission.csv`.

---

## 🛠️ Requirements

- Python 3.x
- numpy
- pandas
- scikit-learn
- xgboost

Install dependencies via pip:

```bash
pip install numpy pandas scikit-learn xgboost
