# 需要转换为png图片运行！！
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 9))
acc = plt.imread('outputs/accuracy.png')
plt.imshow(acc)
plt.show()
plt.figure(figsize=(12, 9))
loss = plt.imread('outputs/loss.png')
plt.imshow(loss)
plt.show()
