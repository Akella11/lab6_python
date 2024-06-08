import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Завантаження даних з CSV файлу
data = pd.read_csv('power_consumption.csv', parse_dates=['date'], index_col='date')

# Виведення перших рядків
print(data.head())

# Ресемплінг даних на щотижневий період для зменшення шуму
weekly_data = data.resample('W').sum()

# Візуалізація даних

plt.figure(figsize=(10,6))
plt.plot(weekly_data, label='Weekly Power Consumption')
plt.title('Weekly Power Consumption')
plt.xlabel('Date')
plt.ylabel('Power Consumption (kilowatts)')
plt.legend()
plt.show()

# Побудова моделі ARIMA
model = ARIMA(weekly_data, order=(10,1,0))  # Параметри (p,d,q) можна налаштувати
model_fit = model.fit()

# Прогнозування на 30 тижнів вперед
forecast = model_fit.forecast(steps=15)

# Візуалізація прогнозу
plt.figure(figsize=(10,6))
plt.plot(weekly_data, label='Observed')
plt.plot(forecast, label='Forecast', color='red')
plt.title('Forecast of Weekly Power Consumption')
plt.xlabel('Date')
plt.ylabel('Power Consumption (kilowatts)')
plt.legend()
plt.show()
# Розділення даних на тренувальну та тестову вибірки
train_size = int(len(weekly_data) * 0.8)
train, test = weekly_data[0:train_size], weekly_data[train_size:]

# Побудова моделі на тренувальних даних
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

# Прогнозування на тестових даних
forecast_test = model_fit.forecast(steps=len(test))

# Візуалізація результатів
plt.figure(figsize=(10,6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(forecast_test, label='Forecast', color='red')
plt.title('Train, Test and Forecast of Weekly Power Consumption')
plt.xlabel('Date')
plt.ylabel('Power Consumption (kilowatts)')
plt.legend()
plt.show()

# Обчислення метрик точності
mse = mean_squared_error(test, forecast_test)
print(f'Mean Squared Error: {mse}')
