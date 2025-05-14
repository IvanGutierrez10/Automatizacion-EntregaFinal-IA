import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cargar los datos
df = pd.read_csv('data/synthetic_data.csv')

# Preparar las características (features)
def create_features(df, window_size=5):
    features = pd.DataFrame()
    
    # Características del nivel
    features['level'] = df['level']
    features['level_diff'] = df['level'].diff()
    features['level_ma'] = df['level'].rolling(window=window_size).mean()
    features['level_std'] = df['level'].rolling(window=window_size).std()
    
    # Características del caudal
    features['caudal'] = df['caudal']
    features['caudal_diff'] = df['caudal'].diff()
    features['caudal_ma'] = df['caudal'].rolling(window=window_size).mean()
    features['caudal_std'] = df['caudal'].rolling(window=window_size).std()
    
    # Características de interacción
    features['level_caudal_ratio'] = df['level'] / df['caudal']
    features['level_caudal_diff'] = df['level'].diff() / df['caudal'].diff()
    
    features = features.dropna()
    
    return features

# Crear características
X = create_features(df)
y = df['noise'].iloc[4:]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Definir los modelos a probar con configuraciones por defecto
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(random_state=42)
}

results = {}

# Entrenar y evaluar cada modelo
for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'precision': precision,
        'f1_score': f1,
        'y_pred': y_pred
    }

# Seleccionar el mejor modelo basado en F1-Score
best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
best_model = results[best_model_name]['model']

# Mostrar métricas del mejor modelo en la consola
print(f"\nMejor modelo: {best_model_name}")
print(f"F1-Score: {results[best_model_name]['f1_score']:.3f}")
print(f"Precision: {results[best_model_name]['precision']:.3f}")
print("\nReporte de clasificación detallado:")
print(classification_report(y_test, results[best_model_name]['y_pred']))

# Visualizar comparación de modelos
metrics_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'F1-Score': [results[m]['f1_score'] for m in results],
    'Precision': [results[m]['precision'] for m in results]
})

plt.figure(figsize=(12, 6))
metrics_df.set_index('Modelo').plot(kind='bar')
plt.title('Comparación de Métricas por Modelo')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/metricas_comparativas.png', dpi=300, bbox_inches='tight')
plt.close()

# Matriz de confusión del mejor modelo
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matriz de Confusión - {best_model_name}')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predicho')
plt.savefig('images/matriz_confusion.png', dpi=300, bbox_inches='tight')
plt.close()

# Guardar el mejor modelo
joblib.dump(best_model, 'models/perturbation_detector.pkl')
print(f"\nMejor modelo ({best_model_name}) guardado en 'models/perturbation_detector.pkl'")
print("Gráficas guardadas en la carpeta 'images':")
print("- metricas_comparativas.png")
print("- matriz_confusion.png") 