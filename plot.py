import matplotlib.pyplot as plt
replicas = [1, 2, 3]
one = [120.706, 126.673, 124.63586]
two = [81.629, 70.9276, 75.217243]
three = [78.3605, 55.00378, 46.532]
four = [80.41742, 42.4678, 34.74959]
five = [85.204, 45.31792, 33.694775]
#six = []
one[:] = [x / 128 for x in one]
two[:] = [x / 128 for x in two]
three[:] = [x / 128 for x in three]
four[:] = [x / 128 for x in four]
five[:] = [x / 128 for x in five]
#six[:] = [x / 128 for x in six]
plt.plot(replicas, one, color='green', label='1 Thread')
plt.plot(replicas, two, color='orange', label='2 Threads')
plt.plot(replicas, three, color='red', label='4 Threads')
plt.plot(replicas, four, color='blue', label='8 Threads')
plt.plot(replicas, five, color='purple', label='16 Threads')
#plt.plot(replicas, six, color='black', label='6 Threads')
plt.legend(loc="lower left")
plt.xlabel('Number of Pods')
plt.xticks([1,2,3])
plt.ylabel('Average Response time')
plt.title('Pod Performance')
plt.show()
