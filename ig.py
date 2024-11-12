import numpy as np

# Entropía de Dispersión
def entropy_disp(data, m, tau, c):
    """
    Calcula la entropía de dispersión para los datos dados.
    
    Parámetros:
    data -- Array de datos de entrada.
    m    -- Dimensión del vector embedding.
    tau  -- Distancia entre elementos consecutivos del vector embedding.
    c    -- Número de símbolos (patrones) de dispersión.
    
    Retorna:
    Entropía de dispersión.
    """
    # Normalización de datos entre 0 y 1
    data_min = np.min(data)
    data_max = np.max(data)
    if data_min == data_max:
        data_norm = np.zeros_like(data)
    else:
        data_norm = (data - data_min) / (data_max - data_min)
    
    # Creación de patrones de dispersión
    patterns = []
    for i in range(len(data_norm) - (m - 1) * tau):
        pattern = tuple((data_norm[i + j * tau] * c).astype(int) for j in range(m))
        patterns.append(pattern)
    
    # Contar frecuencias de cada patrón
    unique_patterns, counts = np.unique(patterns, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    
    # Calcular la entropía de dispersión
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Normalización Sigmoidal
def norm_data_sigmoidal(data):
    """
    Normaliza los datos usando la función sigmoidal.
    
    Parámetros:
    data -- Array de datos de entrada.
    
    Retorna:
    Datos normalizados.
    """
    return 1 / (1 + np.exp(-data))

# Ganancia de Información
def inform_gain(data, labels, m, tau, c):
    """
    Calcula la ganancia de información de cada variable en los datos.
    
    Parámetros:
    data   -- Matriz de datos de entrada.
    labels -- Etiquetas de las clases (última columna de `data`).
    m      -- Dimensión del vector embedding.
    tau    -- Distancia entre elementos consecutivos del vector embedding.
    c      -- Número de símbolos (patrones) de dispersión.
    
    Retorna:
    Array con la ganancia de información para cada variable.
    """
    # Calcular la entropía de dispersión para las etiquetas
    H_y = entropy_disp(labels, m, tau, c)
    
    # Calcular la ganancia de información de cada variable
    info_gain_values = []
    for i in range(data.shape[1]):
        # Entropía condicional de la variable dada la clase
        H_y_given_x = entropy_disp(data[:, i], m, tau, c)
        info_gain = H_y - H_y_given_x
        info_gain_values.append(info_gain)
    
    return np.array(info_gain_values)

def main():
    # Cargar los datos desde la salida de la ETL
    data = np.genfromtxt('outputData/DataClass.csv', delimiter=',')
    
    # Separar las etiquetas de las clases
    labels = data[:, -1]
    data = data[:, :-1]
    
    # Normalizar las variables usando la función sigmoidal
    data_normalized = norm_data_sigmoidal(data)
    
    # Parámetros de Entropía Dispersión
    m = 3  # Dimensión embedding
    tau = 2  # Factor de retardo
    c = 3  # Número de símbolos
    
    # Calcular la ganancia de información
    info_gain_values = inform_gain(data_normalized, labels, m, tau, c)
    
    # Parámetros Top-K
    K = 25  # Número de variables relevantes seleccionadas
    
    # Obtener los índices de las K variables más relevantes
    top_k_indices = np.argsort(info_gain_values)[-K:][::-1]
    
    # Crear archivos de salida
    np.savetxt('outputData/Idx_variable.csv', top_k_indices, delimiter=',', fmt='%d')
    print("Archivo Idx_variable.csv creado exitosamente.")
    
    # Crear DataIG.csv con las variables más relevantes
    data_ig = data_normalized[:, top_k_indices]
    data_ig_with_labels = np.hstack((data_ig, labels.reshape(-1, 1)))
    np.savetxt('outputData/DataIG.csv', data_ig_with_labels, delimiter=',', fmt='%f')
    print("Archivo DataIG.csv creado exitosamente.")

if __name__ == "__main__":
    main()
