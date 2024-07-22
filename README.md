# ImageProcessing

- Tính tích vô hướng hai cách:
+ Điều kiện: hai ma trận/vector cùng kích cỡ a, b
+ Cách 1: np.sum(a*b)
+ Cách 2:
'''
a = a.flatten()
b = b.flatten()
c = a @ b
'''
