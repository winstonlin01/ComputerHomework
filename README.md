# 模拟退火算法优化程序

基于模拟退火算法的结构优化程序，具有图形界面和JSON配置功能。

## 安装步骤

### 1. 确保已安装 Python

确保您的系统已安装 Python 3.7 或更高版本。在命令行中运行：

```bash
python --version
```

或

```bash
python3 --version
```

### 2. 安装依赖包

在项目目录下，运行以下命令安装所需的依赖包：

**Windows (PowerShell):**
```powershell
pip install -r requirements.txt
```

**或者使用 python -m pip:**
```powershell
python -m pip install -r requirements.txt
```

**如果使用 Python 3:**
```powershell
python3 -m pip install -r requirements.txt
```

### 3. 运行程序

安装完成后，运行主程序：

```bash
python main.py
```

或

```bash
python3 main.py
```

## 依赖包说明

程序需要以下Python包：
- **matplotlib** (>=3.5.0): 用于图形可视化
- **numpy** (>=1.21.0): 用于数值计算

这些包会在运行 `pip install -r requirements.txt` 时自动安装。

## 配置文件

程序使用 `config.json` 文件进行配置。您可以修改此文件来调整：
- 算法参数（温度、冷却率等）
- 优化问题设置（城市数量、搜索空间等）
- 显示选项（更新间隔、窗口大小等）

## 使用说明

1. 启动程序后，界面会显示随机生成的城市分布
2. 点击"开始优化"按钮开始运行模拟退火算法
3. 实时查看最优路径和成本变化曲线
4. 可以随时点击"停止"暂停优化
5. 点击"重置"可以重新生成新的问题实例

## 故障排除

如果遇到安装问题：

1. **pip 未找到**: 确保 Python 已正确安装并添加到系统路径
2. **权限错误**: 尝试使用 `pip install --user -r requirements.txt`
3. **网络问题**: 如果下载缓慢，可以使用国内镜像源：
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```


