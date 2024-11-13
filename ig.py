# Cálculo de Ganancia de Información

import numpy as np
import matplotlib.pyplot as plt

# Entropía de Dispersión
def dispersion_entropy(series, m_dim, time_lag, symbols_count):
    n_points = series.shape[0]
    n_embeddings = n_points - (m_dim - 1) * time_lag
    embedded_data = np.array([series[i: i + m_dim * time_lag: time_lag] for i in range(n_embeddings)])
    symbol_patterns = np.round((embedded_data + 0.5) * (symbols_count - 1)).astype(int)
    unique_patterns, pattern_counts = np.unique(symbol_patterns, axis=0, return_counts=True)
    probabilities = pattern_counts / n_embeddings
    return -np.sum(probabilities * np.log2(probabilities))

# Normalización sigmoidal de los datos
def sigmoid_normalisation(data):
    avg = data.mean(axis=0)
    std_dev = data.std(axis=0)
    return 1 / (1 + np.exp(-(data - avg) / (std_dev + 1e-8)))

# Cálculo de Ganancia de Información
def info_gain(features, target, m_dim, time_lag, symbols_count):
    base_entropy = dispersion_entropy(target, m_dim, time_lag, symbols_count)
    info_gains = []
    for col in range(features.shape[1]):
        cond_entropy = 0
        for unique_value in np.unique(features[:, col]):
            subset_target = target[features[:, col] == unique_value]
            if len(subset_target) > 0:
                subset_entropy = dispersion_entropy(subset_target, m_dim, time_lag, symbols_count)
                cond_entropy += (len(subset_target) / len(target)) * subset_entropy
        info_gains.append(base_entropy - cond_entropy)
    return np.array(info_gains)

# Cargar DataClass
def load_and_preprocess_data():
    params = np.loadtxt("data/config.csv", delimiter=",")
    m_dim, time_lag, symbols_count, top_k_vars = int(params[0]), int(params[1]), int(params[2]), int(params[3])
    dataset = np.genfromtxt("outputData/DataClass.csv", delimiter=",", dtype=float)
    features, target = dataset[:, :-1], dataset[:, -1]
    features = sigmoid_normalisation(features)
    return features, target, m_dim, time_lag, symbols_count, top_k_vars

# Guardar resultados de la ganancia de información
def store_info_gain_results(info_gains, features, top_k_vars):
    # Índices de variables en orden descendente de ganancia de información
    sorted_indices = np.argsort(info_gains)[::-1]
    
    # Guardar los índices en Idx_variable.csv
    np.savetxt("outputData/Idx_variable.csv", sorted_indices, fmt="%d", delimiter=",")
    
    # Seleccionar las Top-K variables
    best_indices = sorted_indices[:top_k_vars]
    top_features = features[:, best_indices]
    
    # Guardar los datos seleccionados en DataIG.csv
    np.savetxt("outputData/DataIG.csv", top_features, delimiter=",")

# Proceso principal
def main():
    features, target, m_dim, time_lag, symbols_count, top_k_vars = load_and_preprocess_data()
    info_gains = info_gain(features, target, m_dim, time_lag, symbols_count)
    store_info_gain_results(info_gains, features, top_k_vars)
    print("Ganancia de información calculada y guardada.")
 
    # Gráfico de la ganancia de información por variable
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(info_gains) + 1), info_gains, marker='o', linestyle='-')
    plt.title("Ganancia de Información vs Número de Variable")
    plt.xlabel("Número de Variable")
    plt.ylabel("Ganancia de Información")
    plt.grid()
    plt.show()
    
    # Gráfico del Top 7 en ganancia de información
    top_7_indices = np.argsort(info_gains)[-7:][::-1]
    top_7_values = info_gains[top_7_indices]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 8), top_7_values, tick_label=[f'Var {i+1}' for i in top_7_indices])
    plt.title("Top 7 Variables por Ganancia de Información")
    plt.xlabel("Variables")
    plt.ylabel("Ganancia de Información")
    plt.grid(axis='y')
    plt.show()

if __name__ == '__main__':
    main()
