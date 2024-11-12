import numpy as np

# Gaussian Kernel
def kernel_gauss(X, sigma):
    """
    Calcula el kernel gaussiano de la matriz de datos X.
    
    Parámetros:
    X     -- Matriz de datos de entrada.
    sigma -- Parámetro de ancho del kernel.
    
    Retorna:
    Matriz del kernel K.
    """
    # Calcular las distancias cuadradas
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    K = np.exp(-sq_dists / (2 * sigma**2))
    return K

# Kernel-PCA usando Kernel Gaussiano
def kpca_gauss(X, sigma, top_k):
    """
    Aplica Kernel PCA con un kernel gaussiano en los datos X.
    
    Parámetros:
    X      -- Matriz de datos de entrada.
    sigma  -- Parámetro de ancho del kernel.
    top_k  -- Número de componentes principales a seleccionar.
    
    Retorna:
    X_kpca -- Datos proyectados en los top_k componentes principales.
    """
    # Calcular el kernel gaussiano
    K = kernel_gauss(X, sigma)
    
    # Centrar el kernel
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    
    # Eigen descomposición
    eigvals, eigvecs = np.linalg.eigh(K_centered)
    
    # Seleccionar los top_k componentes principales
    X_kpca = eigvecs[:, -top_k:] @ np.diag(np.sqrt(eigvals[-top_k:]))
    
    return X_kpca

def main():
    # Cargar los datos desde la salida de la etapa IG.py
    data = np.genfromtxt('outputData/DataIG.csv', delimiter=',')
    
    # Seleccionar las primeras 3000 muestras
    data_3000 = data[:3000, :]
    
    # Guardar las primeras 3000 muestras en un nuevo archivo Data.csv
    np.savetxt('outputData/Data.csv', data_3000, delimiter=',', fmt='%f')
    print("Archivo Data.csv creado exitosamente.")
    
    # Separar las etiquetas de las clases
    labels = data_3000[:, -1]
    data_3000 = data_3000[:, :-1]
    
    # Parámetros de KPCA
    sigma = 6.5  # Ancho del Kernel
    top_k = 10   # Número de variables menos redundantes
    
    # Aplicar KPCA
    data_kpca = kpca_gauss(data_3000, sigma, top_k)
    
    # Combinar los datos proyectados con las etiquetas
    data_kpca_with_labels = np.hstack((data_kpca, labels.reshape(-1, 1)))
    
    # Crear archivo de salida DataKpca.csv
    np.savetxt('outputData/DataKpca.csv', data_kpca_with_labels, delimiter=',', fmt='%f')
    print("Archivo DataKpca.csv creado exitosamente.")

if __name__ == "__main__":
    main()
