import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf

# Verificación de GPU
print("GPU disponible:", tf.config.list_physical_devices('GPU'))

# === Datos ===
texto = "hola"
caracteres = sorted(list(set(texto)))  # ['a', 'h', 'l', 'o']
char_to_index = {c: i for i, c in enumerate(caracteres)}
index_to_char = {i: c for c, i in char_to_index.items()}

# One-hot encoding de entrada y salida
X = []
y = []

for i in range(len(texto) - 1):
    entrada = texto[i]
    salida = texto[i + 1]

    # One-hot para entrada y salida
    X.append(to_categorical(char_to_index[entrada], num_classes=len(caracteres)))
    y.append(to_categorical(char_to_index[salida], num_classes=len(caracteres)))

# Convertir a arrays y ajustar forma
X = np.array(X)
y = np.array(y)

# RNN espera [batch, timesteps, features], usamos timesteps=1
X = X.reshape((X.shape[0], 1, X.shape[1]))

# === Modelo ===
model = Sequential([
    Input(shape=(1, len(caracteres))),  # Primera capa con tamaño de entrada explícito
    SimpleRNN(8, activation='tanh'),
    Dense(len(caracteres), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#  EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1, restore_best_weights=True)

#  Entrenamiento 
history = model.fit(X, y, epochs=500, verbose=1, callbacks=[early_stopping])

#  Graficar precisión y pérdida 
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.title('Precisión y Pérdida durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Valor')
plt.legend()
plt.show()

# Prueba del modelo
def predecir_siguiente(letra):
    if letra not in char_to_index:
        return "Letra desconocida"

    x_input = to_categorical(char_to_index[letra], num_classes=len(caracteres)).reshape((1, 1, len(caracteres)))
    pred = model.predict(x_input, verbose=0)
    pred_index = np.argmax(pred)
    return index_to_char[pred_index]

# Ejemplo:
for letra in "hl":
    print(f"Entrada: '{letra}' → Predicción: '{predecir_siguiente(letra)}'")
