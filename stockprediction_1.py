import yfinance as yf
import pandas as pd
import ta
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

model = XGBClassifier(
    n_estimators=200,      
    max_depth=3,         
    learning_rate=0.05,   
    subsample=0.8,       
    colsample_bytree=0.8,  
    random_state=42
)

df = yf.download("AAPL", start="2025-01-01", end="2026-03-01")
df.columns = df.columns.get_level_values(0)

df["Target"] = (df["Close"].shift(-1) >df["Close"]).astype(int)
df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

df["MA_20"] = df["Close"].rolling(20).mean()
df["MA_diff"] = (df["Close"] - df["MA_20"]) / df["MA_20"] * 100

df["Daily_Return"] = df["Close"].pct_change() * 100

df["Volatility"] = df["Daily_Return"].rolling(10).std()

macd = ta.trend.MACD(df["Close"])
df["MACD"] = macd.macd()
df["MACD_signal"] = macd.macd_signal()
df["MACD_diff"] = macd.macd_diff()  

bb = ta.volatility.BollingerBands(df["Close"])
df["BB_upper"] = bb.bollinger_hband()
df["BB_lower"] = bb.bollinger_lband()
df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["MA_20"]  
df["BB_position"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

df.dropna(inplace=True)

features = ["RSI", "MA_diff", "Daily_Return", "Volatility"]
X= df[features]
y = df["Target"]

split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\n", classification_report(y_test, y_pred))

son_gun = df[features].iloc[-1].values.reshape(1, -1)

tahmin = model.predict(son_gun)[0]
olasilik = model.predict_proba(son_gun)[0]

print("\n📈 YARIN TAHMİNİ:")
print(f"Yön: {'⬆️ YÜKSELECEK' if tahmin == 1 else '⬇️ DÜŞECEK'}")
print(f"Yükselme olasılığı: %{olasilik[1]*100:.1f}")
print(f"Düşme olasılığı:   %{olasilik[0]*100:.1f}")

importance = model.feature_importances_
plt.bar(features, importance)
plt.title("Feature Importance")
plt.show(block=False)


test_prices = df["Close"].iloc[split:].values

actual = y_test.values
predicted = y_pred

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,8), sharex=True)

ax1.plot(test_prices, color="blue", linewidth=1.5, label="Gerçek Fiyat")

ax1.set_title("AAPL - Tahmin vs Gerçek Yön")
ax1.set_ylabel("Fiyat (USD)")
ax1.legend()

for i in range(len(predicted)):
    if predicted[i] == actual[i]:
        ax1.axvspan(i, i+1, alpha=0.1, color="green")  
    else:
        ax1.axvspan(i, i+1, alpha=0.1, color="red")

ax2.plot(actual, label="Gerçek Yön", color="blue", linewidth=1)
ax2.plot(predicted, label="Tahmin", color="orange", linewidth=1, linestyle="--")
ax2.set_ylabel("Yön (1=↑, 0=↓)")
ax2.set_xlabel("Gün")
ax2.legend()

plt.tight_layout()
plt.show()

