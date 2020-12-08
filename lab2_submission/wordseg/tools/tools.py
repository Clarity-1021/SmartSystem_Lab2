# 存数据
def save_data(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as b_file:
        print(data, file=b_file)


# 读数据
def load_data(file_name):
    file = open(file_name,'rb').read()
    file = file.decode(encoding='UTF-8',errors='strict')
    return eval(file)


# 读模型
def load_template(gram, template_list):
    for i in range(2, len(gram)):
        cur_tp = gram[i].split(':')[1].split('/')
        tp_list = []
        for j in range(len(cur_tp)):
            tp = cur_tp[j].split(',')
            tp_list.append((int(tp[0][3:]), int(tp[1][:-1])))
        template_list.append(tp_list)