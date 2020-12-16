import matplotlib.pyplot as plt
from lab2_submission.wordseg.tools.tools import *

ROOT_PATH = 'dataset_30/U_'

develop_right_rates = []

develop = [5, 9, 10]
# develop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i in develop:
    file_name = ROOT_PATH + '%d/1/develop_right_rate.py' % (i)
    develop_right_rates.append(load_data(file_name))


plt.figure()
for i, develop_right_rate in enumerate(develop_right_rates):
    plt.plot(develop_right_rate, label='template_' + str(develop[i]))
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('right rate')
plt.title('compare different template by develop right rate')
plt.show()

train_right_rates = []

for i in develop:
    file_name = ROOT_PATH + '%d/1/train_right_rate.py' % (i)
    train_right_rates.append(load_data(file_name))


plt.figure()
for i, train_right_rate in enumerate(train_right_rates):
    plt.plot(train_right_rate, label='template_' + str(develop[i]))
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('right rate')
plt.title('compare different template by train right rate')
plt.show()