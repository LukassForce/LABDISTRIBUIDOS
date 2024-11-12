import numpy as np

# Cargar parámetros desde una configuración interna
def config():
    """
    Devuelve los parámetros de configuración desde el propio código.
    
    Retorna:
    Diccionario con parámetros de configuración.
    """
    config_params = {
        'm': 3,        # Dimensión embedding
        'tau': 2,      # Factor de retardo
        'c': 3,        # Número de símbolos
        'top_k': 25,    # Número de variables relevantes
        'sigma': 6.5,  # Ancho del kernel para KPCA
        'top_k_kpca': 10  # Número de variables para KPCA
    }
    return config_params

def load_and_process_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', dtype=str)
    categorical_columns = [1, 2, 3]
    mappings = {}
    for col in categorical_columns:
        unique_values = np.unique(data[:, col])
        mappings[col] = {value: idx for idx, value in enumerate(unique_values)}
    for col in categorical_columns:
        data[:, col] = np.vectorize(mappings[col].get)(data[:, col])
    class_mapping = {
        'normal': 1,
        'neptune': 2, 'teardrop': 2, 'smurf': 2, 'pod': 2, 'back': 2, 'land': 2, 'apache2': 2,
        'processtable': 2, 'mailbomb': 2, 'udpstorm': 2,
        'ipsweep': 3, 'portsweep': 3, 'nmap': 3, 'satan': 3, 'saint': 3, 'mscan': 3
    }
    # Aplicar el mapeo de clases con un valor predeterminado para valores no encontrados
    data[:, 41] = np.vectorize(lambda x: class_mapping.get(x, 0))(data[:, 41])
    return data.astype(float)

def save_data(data, file_path):
    np.savetxt(file_path, data, delimiter=',', fmt='%f')
    print(f"Archivo {file_path} creado exitosamente.")

def create_class_files(data):
    data_class1 = data[data[:, 41] == 1]
    data_class2 = data[data[:, 41] == 2]
    data_class3 = data[data[:, 41] == 3]
    save_data(data_class1, 'outputData/class1.csv')
    save_data(data_class2, 'outputData/class2.csv')
    save_data(data_class3, 'outputData/class3.csv')

def select_random_samples(file_path, sample_size):
    data = np.genfromtxt(file_path, delimiter=',')
    indices = np.random.choice(data.shape[0], sample_size, replace=False)
    return data[indices]

def create_combined_file():
    samples_class1 = select_random_samples('outputData/class1.csv', 1000)
    samples_class2 = select_random_samples('outputData/class2.csv', 1000)
    samples_class3 = select_random_samples('outputData/class3.csv', 1000)
    combined_data = np.vstack([samples_class1, samples_class2, samples_class3])
    save_data(combined_data, 'outputData/DataClass.csv')

def main():
    data = load_and_process_data('data/KDDTrain.txt')
    save_data(data, 'outputData/Data.csv')
    create_class_files(data)
    create_combined_file()

if __name__ == "__main__":
    main()