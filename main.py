import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


# Дані
data = {
    "Days": [1, 2, 3, 4, 5],
    "Сandies": [2, 4, 6, 8, 10]
}

#data = {
#    "Days": [15, 18, 21, 24, 27, 30, 33],
#    "Сandies": [20, 30, 50, 65, 80, 100, 115]
#}

df = pd.DataFrame(data)
print(df)

# Побудова графіка
plt.scatter(df["Days"], df["Сandies"], color = "blue")
plt.title("Графік залежності")
plt.xlabel("Days")
plt.ylabel("Сandies")
plt.show()


#Навчання моделі
X = df[["Days"]]
Y =  df[["Сandies"]]

model = LinearRegression()
model.fit(X, Y)

print("Вага (W): ", model.coef_)
print("Зміщення (b):", model.intercept_)

# Прогнозування та побудува лінійної регресії:
Y_pred = model.predict(X)

plt.scatter(X, Y, color = "blue")
plt.plot(X, Y_pred, color = "red")
plt.title("Лінійна регресія")
plt.xlabel("Days")
plt.ylabel("Сandies")
plt.show()

# Прогноз на десятий день
new_value = [[10]]
predicted_candies = model.predict(new_value)

print(f"Прогнозована кількість цукерок на {new_value[0][0]} день: ", predicted_candies[0])