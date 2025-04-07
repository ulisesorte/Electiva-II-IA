import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title("Gráfica de datos con Streamlit")

# Slider para seleccionar el número de datos
num_datos = st.slider("Selecciona el número de datos", min_value=5, max_value=50, value=10)

# Generar datos aleatorios
tiempo = np.sort(np.random.uniform(0, 10, num_datos))  # tiempos entre 0 y 10 segundos, ordenados
voltaje = np.random.uniform(0, 5, num_datos)           # voltajes entre 0 y 5 V (puedes ajustar este rango)

# Crear DataFrame
df = pd.DataFrame({"Tiempo (s)": tiempo, "Voltaje (V)": voltaje})

# Mostrar datos en una tabla
st.write("Datos de medición:")
st.dataframe(df)

# Graficar los datos con mejoras estéticas
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df["Tiempo (s)"], df["Voltaje (V)"], marker="o", linestyle="-", color="royalblue")
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Voltaje (V)")
ax.set_title("Voltaje vs. Tiempo")
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
st.pyplot(fig)
