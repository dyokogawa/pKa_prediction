FILE_ENCODING = 'UTF-8'

def write_file_from_list(p_filename, p_mode, p_data):
    file = open(p_filename, p_mode, encoding=FILE_ENCODING)
    for i in range(len(p_data)):
        file.write(str(p_data[i]) + '\n')
    file.close()

def read_file_to_list(p_filename):
    file = open(p_filename, 'r', encoding=FILE_ENCODING)
    file_data = [data.strip() for data in file.readlines()]
    file.close()
    return file_data

