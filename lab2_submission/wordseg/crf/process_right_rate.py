from lab2_submission.wordseg.tools.tools import *

PATH_1 = 'dataset_30/U_7/1/develop_right_rate.py'
# PATH_1 = 'dataset_30/U_2/1/train_right_rate.py'
# PATH_1 = 'dataset_30/U_2/1/develop_right_rate.py'
# PATH_2 = 'dataset_30/U_6/2/develop_right_rate.py'
# PATH_2 = 'dataset_30/U_2/2/develop_right_rate.py'

a = load_data(PATH_1)
b = a[0:90]
# b = a[0:52] + load_data(PATH_2)
# a += load_data(PATH_2)

# PATH_3 = 'dataset_30/U_2/train_right_rate.py'
PATH_3 = 'dataset_30/U_7/develop_right_rate.py'
# PATH_3 = 'dataset_30/U_2/develop_right_rate.py'
save_data(PATH_3, b)

pass
