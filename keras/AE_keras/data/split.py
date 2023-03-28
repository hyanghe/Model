import numpy as np

cons = np.load('x_train_total.npy')
sols = np.load('y_train_total.npy')

total_num = cons.shape[0]

x_train = cons[:int(total_num*0.8)]
x_test = cons[int(total_num*0.8):]

y_train = sols[:int(total_num*0.8)]
y_test = sols[int(total_num*0.8):]

np.save('x_train.npy', x_train)
np.save('x_test.npy', x_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
