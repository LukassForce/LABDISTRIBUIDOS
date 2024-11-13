import numpy as np

# Radial Basis Function (RBF) Kernel
def rbf_kernel(data_matrix, kernel_width):
    squared_distances = np.sum(data_matrix**2, axis=1).reshape(-1, 1) + \
                        np.sum(data_matrix**2, axis=1) - 2 * np.dot(data_matrix, data_matrix.T)
    kernel_matrix = np.exp(-squared_distances / (2 * kernel_width ** 2))
    return kernel_matrix

# Centralize the Kernel Matrix
def centralize_kernel(kernel_matrix):
    num_samples = kernel_matrix.shape[0]
    one_matrix = np.ones((num_samples, num_samples)) / num_samples
    kernel_centered = kernel_matrix - one_matrix @ kernel_matrix - kernel_matrix @ one_matrix + one_matrix @ kernel_matrix @ one_matrix
    return kernel_centered

# Principal Component Extraction with Kernel PCA
def extract_principal_components(input_data, kernel_width, num_components):
    kernel_matrix = rbf_kernel(input_data, kernel_width)
    kernel_centered = centralize_kernel(kernel_matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(kernel_centered)
    main_components = eigenvectors[:, -num_components:] @ np.diag(np.sqrt(eigenvalues[-num_components:]))
    return main_components

# Load and Process Data for Kernel PCA
def load_and_process_kpca_data():
    # Leer configuración de KPCA desde el archivo config.csv
    config = np.loadtxt("data/config.csv", delimiter=",")
    kernel_width = config[4]    # Ancho del kernel RBF
    top_components = int(config[5])   # Número de componentes principales

    # Cargar el archivo DataIG.csv y seleccionar las primeras 3000 muestras
    raw_data = np.genfromtxt('outputData/DataIG.csv', delimiter=',')
    subset_data = raw_data[:3000, :]
    np.savetxt('outputData/Data.csv', subset_data, delimiter=',', fmt='%f')
    print("Archivo Data.csv creado exitosamente.")

    # Separar características y etiquetas
    feature_matrix = subset_data[:, :-1]
    class_labels = subset_data[:, -1]

    # Aplicar Kernel PCA
    transformed_data = extract_principal_components(feature_matrix, kernel_width, top_components)

    # Combinar los datos transformados con las etiquetas
    final_data_with_labels = np.hstack((transformed_data, class_labels.reshape(-1, 1)))
    
    # Guardar el resultado en el archivo DataKpca.csv
    np.savetxt('outputData/DataKpca.csv', final_data_with_labels, delimiter=',', fmt='%f')
    print("Archivo DataKpca.csv creado exitosamente.")

# Run the Complete KPCA Process
def execute_kpca():
    load_and_process_kpca_data()

if __name__ == '__main__':
    execute_kpca()
