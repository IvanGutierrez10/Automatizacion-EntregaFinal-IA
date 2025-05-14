import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 1000
time = np.arange(n_samples)

# caudal en L/s (0.8 a 1.0 L/s)
caudal = np.random.normal(loc=0.9, scale=0.05, size=n_samples)
caudal = np.clip(caudal, 0.8, 1.0)

# Level en cm (0 a 10 cm), donde:
# 8.3cm = 1.0m & 9.1cm = 1.1m
level = np.zeros(n_samples)
level[0] = 8.7  # nivel inicial

# Parámetros para mantener el nivel en el rango deseado
target_level = 8.7  # nivel objetivo
level_std = 0.1    # desviación estándar más pequeña para variaciones más controladas

for i in range(1, n_samples):
    # Generamos un cambio más controlado
    change = np.random.normal(loc=0, scale=level_std)
    
    # Calculamos la desviación del nivel objetivo
    deviation = level[i-1] - target_level
    
    # Ajustamos el nivel considerando la desviación y el caudal
    level[i] = level[i-1] + change - 0.2 * deviation + 0.1 * (caudal[i] - 0.9)
    
    # Aseguramos que el nivel se mantenga en el rango deseado
    if level[i] < 8.3 or level[i] > 9.1:
        level[i] = np.clip(level[i], 8.3, 9.1)

# Inicializamos el array de ruido
noise = np.zeros(n_samples)

# Tipos de perturbaciones posibles
def generate_perturbation(level, caudal, i):
    # Seleccionamos aleatoriamente el tipo de perturbación
    perturbation_type = np.random.choice(['spike', 'drift', 'noise', 'caudal_related'])
    
    if perturbation_type == 'spike':
        # Perturbación tipo pico (cambio brusco)
        return np.random.choice([-1.5, 1.5]) * np.random.rand()
    
    elif perturbation_type == 'drift':
        # Perturbación tipo deriva (cambio gradual)
        return np.random.uniform(-0.8, 0.8)
    
    elif perturbation_type == 'noise':
        # Ruido aleatorio
        return np.random.normal(0, 0.5)
    
    else:  # caudal_related
        # Perturbación relacionada con el caudal
        if caudal[i] > 0.95:  # Caudal alto
            return np.random.uniform(0.3, 1.0)
        elif caudal[i] < 0.85:  # Caudal bajo
            return np.random.uniform(-1.0, -0.3)
        else:
            return np.random.normal(0, 0.3)

# Generamos perturbaciones
perturbation_prob = 0.15  # Aumentamos la probabilidad al 15%
min_duration = 2  # Duración mínima de perturbaciones
max_duration = 5  # Duración máxima de perturbaciones

i = 1
while i < n_samples:
    if np.random.random() < perturbation_prob:
        # Generamos una perturbación
        duration = np.random.randint(min_duration, max_duration + 1)  # Duración entre 2 y 5 muestras
        
        for j in range(duration):
            if i + j < n_samples:
                noise[i + j] = 1
                perturbation = generate_perturbation(level, caudal, i + j)
                level[i + j] += perturbation
                level[i + j] = np.clip(level[i + j], 0, 10)
        
        i += duration
    else:
        i += 1

df = pd.DataFrame({
    'time': time,
    'caudal': caudal,
    'level': level,
    'noise': noise
})

# Imprimir estadísticas de las perturbaciones
print("\nEstadísticas de las perturbaciones generadas:")
print(f"Total de muestras: {len(df)}")
print(f"Muestras con perturbación: {df['noise'].sum()}")
print(f"Porcentaje de perturbaciones: {(df['noise'].sum()/len(df))*100:.2f}%")

# Visualización de los datos
plt.figure(figsize=(12, 5))
plt.plot(df['time'], df['level'], label='Nivel (cm)', color='blue')
plt.plot(df['time'], df['caudal'], label='Caudal (L/s)', color='green', alpha=0.7)
plt.scatter(df['time'][df['noise'] == 1], df['level'][df['noise'] == 1],
            color='red', label='Perturbación', zorder=5)
plt.axhline(8.3, color='gray', linestyle='--', label='1.0 m (8.3 cm)')
plt.axhline(9.1, color='gray', linestyle='--', label='1.1 m (9.1 cm)')
plt.title('Simulación de Nivel y Caudal - Piscina de Zinc')
plt.xlabel('Tiempo')
plt.ylabel('Mediciones')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Guardar los datos
df.to_csv('data/synthetic_data.csv', index=False)