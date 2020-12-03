# 存数据
def save_data(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as b_file:
        print(data, file=b_file)

# 读数据
def load_data(file_name):
    file = open(file_name,'rb').read()
    file = file.decode(encoding='UTF-8',errors='strict')
    return eval(file)