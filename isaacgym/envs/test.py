import torch
import matplotlib.pyplot as plt

class Test:
    def __init__(self):
        self.phase = 0
    def compute_ref_state(self, phase):
        self.phase = torch.tensor([phase])
        sin_pos = torch.sin(2 * torch.pi * self.phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        x = torch.tensor(10, dtype=torch.float64)  # 64位浮点型
        ref_dof_pos = torch.zeros((1, 10), dtype=torch.float64)
        scale_1 = 0.12
        scale_2 = 2 * scale_1

        sin_pos_l[sin_pos_l > 0] = 0
        sin_pos_l[torch.abs(sin_pos) < 0.1] = 0
        ref_dof_pos[0, 2] = -sin_pos_l * scale_1 + 0.4
        ref_dof_pos[0, 3] =  sin_pos_l * scale_2 - 0.8
        ref_dof_pos[0, 4] =  sin_pos_l * scale_1 - 0.4

        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        sin_pos_r[torch.abs(sin_pos) < 0.1] = 0
        ref_dof_pos[0, 7] =  sin_pos_r * scale_1 + 0.4
        ref_dof_pos[0, 8] = -sin_pos_r * scale_2 - 0.8
        ref_dof_pos[0, 9] = -sin_pos_r * scale_1 - 0.4
        return ref_dof_pos

start = 0.0  # 起始值
end = 1.0  # 结束值
step = 0.001  # 步长
test1 = Test()
# 数据存储
phases = []
data_2 = []
data_3 = []
data_4 = []
data_7 = []
data_8 = []
data_9 = []

for i in range(int((end - start) / step) + 1):  # 计算循环次数，并确保包含结束值
    current_value = start + i * step
    ref_dof_pos = test1.compute_ref_state(current_value)

    phases.append(current_value)
    data_2.append(ref_dof_pos[0, 2].item())
    data_3.append(ref_dof_pos[0, 3].item())
    data_4.append(ref_dof_pos[0, 4].item())
    data_7.append(ref_dof_pos[0, 7].item())
    data_8.append(ref_dof_pos[0, 8].item())
    data_9.append(ref_dof_pos[0, 9].item())

# 绘制图形
plt.figure(figsize=(14, 8))
plt.plot(phases, data_2, label='ref_dof_pos[:, 2]')
plt.plot(phases, data_3, label='ref_dof_pos[:, 3]')
plt.plot(phases, data_4, label='ref_dof_pos[:, 4]')
plt.plot(phases, data_7, label='ref_dof_pos[:, 7]')
plt.plot(phases, data_8, label='ref_dof_pos[:, 8]')
plt.plot(phases, data_9, label='ref_dof_pos[:, 9]')

plt.xlabel('Phase')
plt.ylabel('ref_dof_pos values')
plt.legend()
plt.title('Reference DOF Position Values over Phases')
plt.grid(True)
plt.show()