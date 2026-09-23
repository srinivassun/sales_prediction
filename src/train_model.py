import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

# 1. Load data
df = pd.read_csv("../data/sales_data_new.csv")

# 2. Select features
X = df[
    [
        "marketing_spend",
        "customers",
        "orders",
        "holiday",
        "price"
    ]
]

# 3. Select target
y = df["sales"]

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. Create algorithm/estimator
model = LinearRegression()

# 6. Train
model.fit(X_train, y_train)

# 7. Predict
predictions = model.predict(X_test)

# 8. Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)