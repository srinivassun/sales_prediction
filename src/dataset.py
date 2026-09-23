import pandas as pd

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
#Shows the first 5 rows.
print(df.head())
#Shows the first 3 rows.
print(df.head(3))
#Shows the last 5 rows.
print(df.tail())
#Shows the last 2 rows.
print(df.tail(2))
print(df.shape)

#Inspect the Data
print(df.info())
print(df.describe())

