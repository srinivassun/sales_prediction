import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

# 8. Save the model
joblib.dump(
    model,
    "../model/sales_model.pkl"
)