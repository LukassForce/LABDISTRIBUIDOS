import numpy as np

def process_data():
    # Definición de diccionarios de transformación para columnas categóricas
    protocol_map = {'tcp': 0, 'udp': 1, 'icmp': 2}
    service_map = {service: idx for idx, service in enumerate([
        'urh_i', 'private', 'shell', 'finger', 'domain_u', 'ecr_i', 'rje', 'uucp', 'imap4', 'vmnet', 
        'ctf', 'daytime', 'pop_2', 'red_i', 'X11', 'http_8001', 'bgp', 'tim_i', 'domain', 'exec', 
        'urp_i', 'remote_job', 'auth', 'uucp_path', 'login', 'http_443', 'ldap', 'gopher', 'sunrpc', 
        'kshell', 'sql_net', 'discard', 'klogin', 'netbios_ns', 'http', 'mtp', 'eco_i', 'IRC', 'efs', 
        'ntp_u', 'time', 'csnet_ns', 'courier', 'echo', 'telnet', 'whois', 'printer', 'name', 'smtp', 
        'ftp_data', 'supdup', 'iso_tsap', 'netstat', 'pm_dump', 'ftp', 'Z39_50', 'other', 'nnsp', 
        'netbios_dgm', 'systat', 'ssh', 'netbios_ssn', 'link', 'pop_3', 'nntp', 'hostnames'
    ])}
    state_map = {'S3': 0, 'S2': 1, 'SF': 2, 'RSTO': 3, 'REJ': 4, 'RSTOS0': 5, 'S0': 6, 
                 'OTH': 7, 'S1': 8, 'RSTR': 9, 'SH': 10}

    # Primera etapa: convertir columnas categóricas y guardar en outputData/Data.csv
    with open('data/KDDTrain.txt', 'r') as source_file, open('outputData/Data.csv', 'w') as target_file:
        for line in source_file:
            fields = line.strip().split(',')
            fields_converted = fields[:-2]
            
            # Transformación de valores categóricos en columnas específicas
            fields_converted[1] = str(protocol_map.get(fields[1], -1))  # Columna protocolo
            fields_converted[2] = str(service_map.get(fields[2], -1))   # Columna servicio
            fields_converted[3] = str(state_map.get(fields[3], -1))     # Columna estado de conexión

            # Escribir la línea convertida en outputData/Data.csv
            target_file.write(','.join(fields_converted) + '\n')

    # Segunda etapa: Clasificación en archivos específicos según el tipo de ataque
    dos_attacks = {'neptune', 'teardrop', 'smurf', 'pod', 'back', 'land', 'apache2', 'processtable', 'mailbomb', 'udpstorm'}
    probe_attacks = {'ipsweep', 'portsweep', 'nmap', 'satan', 'saint', 'mscan'}
    
    with open('data/KDDTrain.txt', 'r') as source_file, \
         open('outputData/class1.csv', 'w') as normal_file, \
         open('outputData/class2.csv', 'w') as dos_file, \
         open('outputData/class3.csv', 'w') as probe_file:
        for line in source_file:
            fields = line.strip().split(',')
            attack_label = fields[-2]  # Columna de tipo de ataque
            
            # Convertir valores categóricos de protocolo, servicio y estado
            fields_converted = fields[:-2]
            fields_converted[1] = str(protocol_map.get(fields[1], -1))  # Protocolo
            fields_converted[2] = str(service_map.get(fields[2], -1))   # Servicio
            fields_converted[3] = str(state_map.get(fields[3], -1))     # Estado de conexión

            # Guardar la línea en el archivo correspondiente según el tipo de ataque
            if attack_label == 'normal':
                normal_file.write(','.join(fields_converted) + '\n')
            elif attack_label in dos_attacks:
                dos_file.write(','.join(fields_converted) + '\n')
            elif attack_label in probe_attacks:
                probe_file.write(','.join(fields_converted) + '\n')

    # Tercera etapa: Selección de muestras desde índices y creación de outputData/DataClass.csv
    def load_sample_indices(filename, sample_size):
        # Cargar índices de muestra desde archivo
        indices = np.genfromtxt(filename, dtype=int)
        # Seleccionar los primeros `sample_size` índices
        return indices[:sample_size]

    sample_size = 5000
    indices_normal = load_sample_indices('data/idx_class1.csv', sample_size)
    indices_dos = load_sample_indices('data/idx_class2.csv', sample_size)
    indices_probe = load_sample_indices('data/idx_class3.csv', sample_size)

    # Leer todas las líneas y seleccionar muestras según índices
    with open('data/KDDTrain.txt', 'r') as source_file:
        all_lines = source_file.readlines()

    collected_samples = []
    for idx in indices_normal:
        fields = all_lines[idx-1].strip().split(',')[:-2]
        fields[1] = str(protocol_map.get(fields[1], -1))
        fields[2] = str(service_map.get(fields[2], -1))
        fields[3] = str(state_map.get(fields[3], -1))
        collected_samples.append(','.join(fields) + ',1\n')  # Etiqueta clase 1

    for idx in indices_dos:
        fields = all_lines[idx-1].strip().split(',')[:-2]
        fields[1] = str(protocol_map.get(fields[1], -1))
        fields[2] = str(service_map.get(fields[2], -1))
        fields[3] = str(state_map.get(fields[3], -1))
        collected_samples.append(','.join(fields) + ',2\n')  # Etiqueta clase 2

    for idx in indices_probe:
        fields = all_lines[idx-1].strip().split(',')[:-2]
        fields[1] = str(protocol_map.get(fields[1], -1))
        fields[2] = str(service_map.get(fields[2], -1))
        fields[3] = str(state_map.get(fields[3], -1))
        collected_samples.append(','.join(fields) + ',3\n')  # Etiqueta clase 3

    # Guardar todas las muestras seleccionadas en outputData/DataClass.csv
    with open('outputData/DataClass.csv', 'w') as output_samples:
        output_samples.writelines(collected_samples)

if __name__ == '__main__':
    process_data()
