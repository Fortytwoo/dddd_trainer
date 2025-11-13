# 安装 cuda
（参考：https://blog.csdn.net/AI_dataloads/article/details/133043869）

命令行查看当前的 GPU cuda 版本

```powershell
PS C:\Users\12460> nvidia-smi
Thu Nov 13 11:11:43 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 572.83                 Driver Version: 572.83         CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3050 ...  WDDM  |   00000000:01:00.0  On |                  N/A |
| N/A   68C    P5              7W /   60W |     708MiB /   4096MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

这里是 12.8
所以在这个页面 https://developer.nvidia.com/cuda-toolkit-archive
安装 12.8.0 版本即可

# 安装 pytorch
(参考 https://pytorch.org/get-started/locally/ )

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

# 安装 cudnn(不过一般安装 pytorch 时会自动安装 cudnn)

```powershell
uv add nvidia-cudnn
```
