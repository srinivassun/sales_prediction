import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)
new_data = {
    "marketing_spend": [
        5500, 600, 7000, 8000, 9000,
        10000, 1100, 12000, 13000, 14000
    ],

    "customers": [
        300, 320, 35, 370, 400,
        420, 450, 470, 500, 520
    ],

    "orders": [
        75, 80, 99, 95, 105,
        110, 120, 125, 135, 140
    ],

    "holiday": [
        0, 0, 0, 0, 1,
        0, 0, 1, 0, 0
    ],

    "price": [
        5000, 5050, 5000, 5000, 5000,
        5000, 5000, 5000, 5000, 5000
    ],

    "sales": [
        375000, 400000, 400000, 475000, 525000,
        550000, 600000, 625000, 685000, 750000
    ]
}
df = pd.DataFrame(new_data)

print(df)

#Identify Features and Target
x = df[
    [
        "marketing_spend",
        "customers",
        "orders",
        "holiday",
        "price"
    ]
]

y = df["sales"]

# x -> features
# y -> target

#Implement Train/Test Split
#Scikit-learn provides:
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# Now Load the Saved Model
model = joblib.load(
    "../model/sales_model.pkl"
)

predictions = model.predict(x_test)

# 8. Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)