import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error

data = {
    "marketing_spend": [
        5000, 6000, 7000, 8000, 9000,
        10000, 11000, 12000, 13000, 14000
    ],

    "customers": [
        300, 320, 350, 370, 400,
        420, 450, 470, 500, 520
    ],

    "orders": [
        75, 80, 90, 95, 105,
        110, 120, 125, 135, 140
    ],

    "holiday": [
        0, 0, 0, 0, 1,
        0, 0, 1, 0, 0
    ],

    "price": [
        5000, 5000, 5000, 5000, 5000,
        5000, 5000, 5000, 5000, 5000
    ],

    "sales": [
        375000, 400000, 450000, 475000, 525000,
        550000, 600000, 625000, 675000, 700000
    ]
}

df = pd.DataFrame(data)

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
#Meaning:
#80% → Training
#20% → Testing
#random_state=42 makes the split reproducible.

#Now We Choose the Algorithm
#Let's use:
#Linear Regression
model = LinearRegression()

#Train the Model
model.fit(x_train, y_train)

#Now Make a Prediction
predictions = model.predict(x_test)

print(predictions)

#Evaluate the Model
#MAE-Mean Absolute Error.
mae = mean_absolute_error(
    y_test,
    predictions
)

print(mae)


#RMSE
rmse = root_mean_squared_error(
    y_test,
    predictions
)

print(rmse)

#Compare Actual vs Predicted
results = pd.DataFrame({
    "actual": y_test,
    "predicted": predictions
})

print(results)