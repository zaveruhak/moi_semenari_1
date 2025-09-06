#1. Import the numpy package under the name np (★☆☆)
import numpy as np


#12. Create a 3x3x3 array with random values (★☆☆
# a = np.empty((3, 3, 3)) #он создаёт рандомные, честно честно, я в доках читал
# print(a)


#24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)
# A = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9],
#               [10, 11, 12],
#               [13, 14, 15]])
# B = np.array([[1, 2],
#               [3, 4],
#               [5, 6]])
# C = np.dot(A, B)
# print(C)

#30. How to find common values between two arrays? (★☆☆)
# A = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9],
#               [10, 11, 12],
#               [13, 14, 15]])
# B = np.array([[1, 2],
#               [3, 4],
#               [5, 20]])
# C = np.intersect1d(A, B)
# print(C)


#40. Create a random vector of size 10 and sort it (★★☆)
# rg = np.random.default_rng(1)
# a = np.floor(10 * rg.random((1, 10)))
# b = np.sort(a)
# print(a)
# print(b)


# 59. How to sort an array by the nth column? (★★☆)
# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15],
#              [5, 5, 5], [1, 1, 1], [10, 11, 10], [2, 2, 2], [3, 3, 3]])
# b = sorted(a, key=lambda x: x[1])
# c = np.vstack(b)
# print(c)


# 61. Find the nearest value from a given value in an array (★★☆)
# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15],
#              [5, 5, 5], [1, 1, 1], [10, 11, 10], [2, 2, 2], [3, 3, 3]])
# z = 5.4
# d = np.inf
# ans = 0
# for i in a.flat:
#     if abs(i - z) < d:
#         d = abs(i - z)
#         ans = i
# print(ans)



#72. How to swap two rows of an array? (★★★)
# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15],
#              [5, 5, 5], [1, 1, 1], [10, 11, 10], [2, 2, 2], [3, 3, 3]])
# a[3], a[5] = a[5], a[3]
# print(a)

#83. How to find the most frequent value in an array?
# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15],
#              [5, 5, 5], [1, 1, 1], [10, 11, 10], [2, 2, 2], [3, 3, 3]])
# d = dict()
# for i in a.flat:
#     if d.get(i):
#         d[i] += 1
#     else:
#         d[i] = 1
# m = max(d.values())
# for j in d.keys():
#     if d[j] == m:
#         print(j)


#94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)
# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15],
#              [5, 5, 5], [1, 1, 1], [10, 11, 10], [2, 2, 2], [3, 3, 3]])
# for i in a:
#     if len(set(i)) != 1:
#         print(i)




