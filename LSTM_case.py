##############################################################
###### LSTM ile Hisse Senedi Fiyati Tahminlemesi
##############################################################

import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt

#### Model degerlendirme ve scale edebilmemiz icin kullanilacaklar
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#### Model icin kullanilacak kutuphaneler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import warnings
warnings.filterwarnings("ignore")

####Tensorflow warning engelleme
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

df = pd.read_csv("/Users/yunusemreturkoglu/Desktop/BootCamp/turkcell/TSLA.csv")
df.head()

df["Date"] = pd.to_datetime(df["Date"])
df.info()

tesla_df = df[["Date", "Close"]]

print("Minumum tarih:", tesla_df["Date"].min())
# Minumum tarih: 2010-06-29 00:00:00
print("Maximum tarih:", tesla_df["Date"].max())
# Maximum tarih: 2020-02-03 00:00:00

tesla_df.index = tesla_df["Date"]
tesla_df.drop("Date", axis=1, inplace=True)

result_df = tesla_df.copy()

tesla_df.plot(figsize=(15,5), title="Tesla Stock Price")

plt.figure(figsize=(12,6))
plt.plot(tesla_df["Close"], color="blue")
plt.ylabel("Stock Prize")
plt.xlabel("Time")
plt.title("Tesla Stock Priza")
plt.show()

tesla_df = tesla_df.values

tesla_df = tesla_df.astype("float32")
tesla_df[0:5]
### Veri Hazirlama Islemleri(Data Preparation)

# train-test ayriminin yapilmasi

def split_data(dataframe, test_size):
    pos = int(round(len(dataframe)*(1-test_size)))
    train = dataframe[:pos]
    test = dataframe[pos:]
    return train, test, pos
train, test, pos = split_data(tesla_df, 0.20)
train.shape

scaler_train = MinMaxScaler(feature_range=(0,1))
train = scaler_train.fit_transform(train)

scaler_test = MinMaxScaler(feature_range=(0,1))
test = scaler_test.fit_transform(test)

def create_features(data, lookback):
    X, Y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i,0])
        Y.append(data[i,0])
    return np.array(X), np.array(Y)

lookback = 20 # Kullanmamizin sebebi aylik tahmin yapacagiz icin borsalar sadece hafta ici acik oldugu icin

#### Train Veri Seti
X_train, y_train = create_features(train, lookback)

#### Test Veri Seti

X_test, y_test = create_features(test, lookback)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train[0:5]

### LSTM formatina uygun hale getirecegiz

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#### Modellemee (Modeling)

model = Sequential()

model.add(LSTM(units=50, activation="relu", input_shape=(X_train.shape[1], lookback)))
model.add(Dropout(0.2)) # asiri ogrenmeyi engellemek amacli
model.add(Dense(1)) # cikti katmani  normalde sinif sayisi kadar olmasi gerekiyor. Ancak regression problemi oldugu icin 1

model.summary()


# Optimizasyon ve degerlendirme metricleri ayarlanmasi

model.compile(loss="mean_squared_error", optimizer="adam")

# Asiri ogrenmeye karsi tedbir olarak ve modelin kaydedilmesini saglayacak bolum

callbacks = [EarlyStopping(monitor="val_loss", patience=3, verbose=1, mode="min"),
             ModelCheckpoint(filepath="mymodel.h5", monitor="val_loss", mode="min",
                             save_best_only=True, save_weights_only=False, verbose=1)]

history = model.fit(x=X_train, y=y_train, epochs=100, batch_size=20,
                    validation_data=(X_test, y_test), callbacks=callbacks, shuffle=False)

plt.figure(figsize=(20,5))
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend(loc="upper right")
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.ylim([0, max(plt.ylim())])
plt.title("Training and Validation Loss", fontsize=3)
plt.show()

#### Degerlendirme (Evaluation)

loss = model.evaluate(X_test, y_test, batch_size=1)
print("\nTest Loss: %.1f%%" % (100.0*loss))
# Test Loss: 2.2%

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Tahmin edilen degerlere yapilan 0 1 arasindan kurtarma islemi
train_predict = scaler_train.inverse_transform(train_predict)
test_predict = scaler_test.inverse_transform(test_predict)

# Gercek degerlere yapilan 0 1 arasindan kurtarma islemi
y_train = scaler_train.inverse_transform(y_train)
y_test = scaler_test.inverse_transform(y_test)

# Train veri setine ait RMSE degeri
train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))

# Test veri setine ait RMSE degero
test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))

print(f"Train RMSE: {train_rmse}")
# Train RMSE: 12.784770965576172
print(f"Test RMSE: {test_rmse}")
# Test RMSE: 28.969621658325195

train_prediction_df = result_df[lookback:pos]
train_prediction_df["Predicted"] = train_predict
train_prediction_df.head()

test_prediction_df = result_df[pos+lookback:]
test_prediction_df["Predicted"] = test_predict
test_prediction_df.head()

# Train ve Test Tahminlerinin Ayri Ayri gorsellestirilmesi

plt.figure(figsize=(15,5))
plt.plot(result_df, label="Real Number of Passengers")
plt.plot(train_prediction_df["Predicted"], color="blue", label="Train Predicted Number of Passengers")
plt.plot(test_prediction_df["Predicted"], color="red", label="Test Predicted Number of Passengers")
plt.title("Number of Passengers Prediction")
plt.xlabel("Time")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()



