import numpy as np
import matplotlib.pyplot as plt

# 定义每个分段的px和pz函数
def vx1(t):
    return 5.41682 * (t/2)**3 - 0.36654 * (t/2)**2 + 0.0312 * (t/2) - 0.0005  # 返回 t 对应的零数组

def vz1(t):
    return 1.18999 * (t/2)**3 + 30.88172 * (t/2)**2 - 12.73049 * (t/2) + 0.05268

def vx2(t):
    return -80.54291 * ((t/2) - 0.41)**3 + 12.09768 * ((t/2) - 0.41)**2 + 3.90848 * ((t/2) - 0.41) + 0.32456

def vz2(t):
    return -250.37119 * ((t/2) - 0.41)**3 - 20.73337 * ((t/2) - 0.41)**2 + 23.1536 * ((t/2) - 0.41) + 0.01772

def vx3(t):
    return np.full_like((t/2), 0.946588)

def vz3(t):
    return -9.81 * ((t/2) - 0.6) + 1.94436

def vx4(t):
    return 76.30915 * ((t/2) - 1)**3 - 26.22602 * ((t/2) - 1)**2 - 0.33936 * ((t/2) - 1) + 0.94734

def vz4(t):
    return -206.50273 * ((t/2) - 1)**3 + 143.54032 * ((t/2) - 1)**2 - 8.8353 * ((t/2) - 1) - 1.97791

def vx5(t):
    return 0.385 * np.exp(-13.056 * ((t/2) - 1.19))

def vz5(t):
    return 0.142 + 0.633 * np.exp(-20.000 * (((t/2)-1.19) - 0.113) ** 2)
# 时间区间
t1 = np.linspace(0, 0.8, 100)
t2 = np.linspace(0.81, 1.19, 100)
t3 = np.linspace(1.2, 2.0, 100)
t4 = np.linspace(2.01, 2.36, 100)
t5 = np.linspace(2.37, 3.94, 100)

# 计算每个分段的px和pz值
vx_values = np.concatenate([vx1(t1), vx2(t2), vx3(t3), vx4(t4), vx5(t5)])
vz_values = np.concatenate([vz1(t1), vz2(t2), vz3(t3), vz4(t4), vz5(t5)])
time_values = np.concatenate([t1, t2, t3, t4, t5])

# 绘制px随时间变化的图
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)  # 创建一个2行1列的子图，选择第一个
plt.plot(time_values, vx_values, label="vx(t)", color='b')
plt.xlabel("time (t)")
plt.ylabel("px")
plt.title("px over time")
plt.grid(True)
plt.legend()

# 绘制pz随时间变化的图
plt.subplot(2, 1, 2)  # 选择第二个子图
plt.plot(time_values, vz_values, label="vz(t)", color='r')
plt.xlabel("time (t)")
plt.ylabel("pz")
plt.title("pz over time")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()