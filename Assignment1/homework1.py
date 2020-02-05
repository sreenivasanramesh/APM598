import numpy as np
import matplotlib.pyplot as plt


############################################## 
		## Exercise 1 (Python) ## 
##############################################
data = np.loadtxt('data_HW1_ex1.csv',delimiter=',') 
x,y = data[:,0],data[:,1]
x_train, y_train = data[0:int(len(data)*0.8), 0], data[0:int(len(data)*0.8), 1]
x_test, y_test = data[int(len(data)*0.8):, 0], data[int(len(data)*0.8):, 1]

k = 12

def loss(degree, x, y):
	return np.sum((np.polyval(np.polyfit(x, y, degree), x) - y)**2)


def ex1a():
	losses = list()
	for degree in range(0, k):
		losses.append(loss(degree, x, y))

	#plot loss against k (k = degree of polynomial)
	plt.figure('Ex. 1a - Loss function vs Poly Degree')
	plt.plot(range(0, k), losses, 'b-')
	plt.xlabel('Degree K')
	plt.ylabel('Loss')
	plt.show()


def ex1bc():
	train_error = list()
	test_error = list()

	for degree in range(0, k+1):
		train_error.append(loss(degree, x_train, y_train))
		#poly1d makes an object of the fitted polynomial 
		#and can be used to predict 
		#another way of predicting as compared to polyval
		poly_fit = np.poly1d(np.polyfit(x_train, y_train, degree))
		predict = poly_fit(x_test)
		test_error.append(np.sum((predict-y_test)**2))
	print('Degree   TrainError   TestError')
	for degree, (train_err, test_err) in enumerate(zip(train_error, test_error)):
		print('%-9i%-13.2f%.2f'%(degree, train_err, test_err))
	k_star = test_error.index(min(test_error))
	print('\nOrder k* = {} since it has the least test error\n'.format(k_star))
	coeffs = np.polyfit(x, y, k_star)
	print('\nCoeffs for k*(2): ', coeffs)

	#display test vs train loss
	fig = plt.figure('Ex. 1b - Test/Train Error vs Poly Degree', figsize=(12,5))
	ax1 = fig.add_subplot(1, 2, 1)
	train_line, = ax1.plot(range(0, k+1), train_error, c='b')
	test_line, = ax1.plot(range(0, k+1), test_error, c='r')
	ax1.set(xlabel="Degree K", ylabel="Error")
	ax1.legend([train_line, test_line], ["Train Loss", "Test Loss"])

	#display polyfit
	ax2 = fig.add_subplot(1, 2, 2)
	predict = np.poly1d(np.polyfit(x, y, k_star))
	ax2.plot(x, y, 'bo')
	x.sort()
	p_fit, = ax2.plot(x, predict(x), 'r-')
	ax2.set(xlabel="x", ylabel="y")
	ax2.legend([p_fit], ["K* = {}".format(k_star)])
	plt.show()

print('\n******* Exercise 1 ********\n')
ex1a()
ex1bc()




############################################## 
		## Exercise 2 (Python) ## 
##############################################





############################################## 
		## Exercise 3 (Python) ## 
##############################################





