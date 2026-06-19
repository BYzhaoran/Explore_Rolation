# 旋转等变触觉方向实验

这个项目研究的是：当触觉图像或合成椭圆旋转时，模型能不能稳定预测接触方向。核心对比对象是普通 CNN 和旋转等变模型，实验重点不是单次准确率，而是模型在不同旋转角度下是否保持稳定。

## 一句话概览

输入是一张 `64x64` 灰度触觉图像，输出是一个二维方向向量 `[cos(2θ), sin(2θ)]`。这里使用 `2θ` 是为了处理方向的 180° 对称性：同一个长轴方向转半圈后，本质上还是同一个方向。

项目里有三类关键流程：

- 生成训练集和测试集
- 训练 baseline 和 equivariant 两种模型，并按 mode 区分数据集
- 评估单个 checkpoint，以及把四个 checkpoint 放在同一张表和同一张图里对比

## 目录结构

主要内容都在这些路径下：

- `Datasets/Dims2/`：mode 1 使用的数据集
- `Datasets/Dims2_Mul/`：mode 2 使用的数据集
- `Scripts/Data_generate_D2.py`：生成 `Dims2`
- `Scripts/Data_generate_D2_mul.py`：生成 `Dims2_Mul`
- `Scripts/generate_tactile_test_sets.py`：重建测试集角度划分
- `Scripts/tactile_rotation/train_baseline.py`：训练 baseline
- `Scripts/tactile_rotation/train_equivariant.py`：训练 equivariant
- `Scripts/tactile_rotation/evaluate.py`：单模型评估
- `Scripts/tactile_rotation/compare_checkpoints.py`：四个 checkpoint 综合对比

## 数据集说明

### `Datasets/Dims2`

这个数据集适合看模型是否能从少量固定朝向中学到旋转规律。默认训练集和测试集布局如下：

- `images/`：训练图像
- `labels/`：训练标签
- `test/<rotation>/`：测试图像，按旋转角度分文件夹
- `test_labels/<rotation>/`：对应测试标签

### `Datasets/Dims2_Mul`

这个数据集结构相同，只是训练样本本身就更偏向多角度场景，适合 mode 2。

## 标签编码

标签不是一个角度值，而是一个二维单位向量：

- `x = cos(2θ)`
- `y = sin(2θ)`

这样做的原因是：方向在几何上有 180° 周期性，直接回归角度会遇到 `θ` 和 `θ + 180°` 等价的问题。把目标放在单位圆上，模型更容易学习，而且误差也更容易统一成角度差来解释。

## 环境依赖

项目使用 Python、PyTorch、OpenCV、NumPy、tqdm、pandas，绘图还需要 `matplotlib`。

建议先激活你的 `learn` 环境，再安装依赖：

```bash
conda activate learn
pip install -r requirements.txt
```

如果你是从别的环境跑命令，记得保证运行目录在仓库根目录，否则相对路径会找不到数据集和 checkpoint。

## 生成数据

### 生成 `Dims2`

```bash
python Scripts/Data_generate_D2.py
```

默认会生成：

- 训练集到 `Datasets/Dims2/images/` 和 `Datasets/Dims2/labels/`
- 测试集到 `Datasets/Dims2/test/<rotation>/` 和 `Datasets/Dims2/test_labels/<rotation>/`

默认测试角度是 `0° / 90° / 180° / 270°`。

如果你要自己指定训练旋转和测试旋转，也可以传参：

```bash
python Scripts/Data_generate_D2.py --train-rotations 0 90 180 270 --test-rotations 0 45 90 135
```

常用参数：

- `--train-samples`：训练样本数，默认 `10000`
- `--test-samples-per-rotation`：每个测试角度的样本数，默认 `1000`
- `--train-rotations`：训练集允许的角度
- `--test-rotations`：测试集角度
- `--no-preview`：不弹出预览图

### 生成 `Dims2_Mul`

```bash
python Scripts/Data_generate_D2_mul.py
```

它会生成同样结构的数据，只是训练分布不同。默认测试角度是 `0° / 45° / 90° / 135°`，更适合看多角度鲁棒性。

### 重新生成测试集

如果要把 `Dims2` 和 `Dims2_Mul` 的测试集重新清空并生成一套新的 45° 步进测试角度，可以用：

```bash
python Scripts/generate_tactile_test_sets.py --dataset both --rotations 0 45 90 135 180 225 270 315
```

这个脚本会先删除原有的 `test/` 和 `test_labels/`，再重新创建同名目录并写入新测试集。它默认每个角度生成 `1000` 个样本。

## 训练

### mode 1 的 baseline

```bash
python Scripts/tactile_rotation/train_baseline.py --mode 1
```

默认读取 `Datasets/Dims2/`，输出 checkpoint 到：

- `Scripts/tactile_rotation/baseline_mode1.pt`

### mode 2 的 baseline

```bash
python Scripts/tactile_rotation/train_baseline.py --mode 2
```

默认读取 `Datasets/Dims2_Mul/`，输出 checkpoint 到：

- `Scripts/tactile_rotation/baseline_mode2.pt`

### mode 1 的 equivariant

```bash
python Scripts/tactile_rotation/train_equivariant.py --mode 1
```

输出 checkpoint 到：

- `Scripts/tactile_rotation/equivariant_mode1.pt`

### mode 2 的 equivariant

```bash
python Scripts/tactile_rotation/train_equivariant.py --mode 2
```

输出 checkpoint 到：

- `Scripts/tactile_rotation/equivariant_mode2.pt`

### 训练脚本的常用参数

这两个训练脚本都支持：

- `--mode`：选择 `Dims2` 或 `Dims2_Mul`
- `--data-root`：手动覆盖数据集路径
- `--epochs`：训练轮数
- `--batch-size`：批大小
- `--lr`：学习率
- `--device`：指定设备
- `--checkpoint`：手动指定 checkpoint 路径
- `--max-samples`：只用前多少个样本做快速实验

## 单模型评估

评估脚本会按测试角度逐个输出平均角度误差，并把结果图保存到对应目录。

### 评估 baseline mode 1

```bash
python Scripts/tactile_rotation/evaluate.py --mode 1 --model baseline --checkpoint Scripts/tactile_rotation/baseline_mode1.pt
```

### 评估 baseline mode 2

```bash
python Scripts/tactile_rotation/evaluate.py --mode 2 --model baseline --checkpoint Scripts/tactile_rotation/baseline_mode2.pt
```

### 评估 equivariant mode 1

```bash
python Scripts/tactile_rotation/evaluate.py --mode 1 --model equivariant --checkpoint Scripts/tactile_rotation/equivariant_mode1.pt
```

### 评估 equivariant mode 2

```bash
python Scripts/tactile_rotation/evaluate.py --mode 2 --model equivariant --checkpoint Scripts/tactile_rotation/equivariant_mode2.pt
```

### 输出位置

评估结果会分别输出到：

- `Scripts/tactile_rotation/baseline_mode1/`
- `Scripts/tactile_rotation/baseline_mode2/`
- `Scripts/tactile_rotation/equivariant_mode1/`
- `Scripts/tactile_rotation/equivariant_mode2/`

每次运行会在目录里保存一张误差曲线图，文件名是 `error_comparison.png`。

### 评估脚本行为

它会遍历 `Datasets/.../test/` 下实际存在的角度文件夹，所以测试角度不是写死的。某个角度如果目录为空，会跳过，不会直接报错。

## 四模型综合对比

如果你想把四个 checkpoint 放在一起看，可以运行：

```bash
python Scripts/tactile_rotation/compare_checkpoints.py
```

这个脚本会做三件事：

- 在同一组 `Dims2` 测试角度上评估四个 checkpoint
- 输出一张按旋转角度组织的结果表
- 画出四条误差曲线，横轴是旋转角度，纵轴是方向预测误差

### 结果文件

脚本会输出到：

- `Scripts/tactile_rotation/comparison_results/comparison_table.csv`
- `Scripts/tactile_rotation/comparison_results/comparison_error_plot.png`

### 结果解读

表格里每一行对应一个旋转角度，每一列对应一个模型。图里四条线分别代表：

- `baseline` + mode 1
- `equivariant` + mode 1
- `baseline` + mode 2
- `equivariant` + mode 2

这样你可以直接比较：哪种模型整体误差更低，哪种模型在某些旋转角度上更容易退化。

## 为什么这件事重要

对于机器人抓取来说，模型面对的是从没见过的物体，而且物体在桌面上的姿态通常并不固定。一个好的触觉模型不能只会记住“这件东西在训练时长什么样”，而要能处理“换个角度后它还是同一个物体”。这就是“转一转还能猜对”的价值。

如果模型对旋转很敏感，机械臂稍微换个方向就可能判断失真，导致夹爪对不准、抓空、滑落或者过度用力。相反，如果模型学到了旋转一致性，它就更关注物体本身的结构和接触方向关系，而不是表面姿态。这会直接提升抓取稳定性、泛化能力和对未知物体的适应性。

## 常见问题

### checkpoint 找不到

确认你传给 `--checkpoint` 的路径是存在的，而且从仓库根目录运行时相对路径是正确的。

### 评估结果里某个角度被跳过

说明那个旋转文件夹是空的。先检查测试集是否已经重新生成。

### 画图报错

说明环境里缺 `matplotlib`。先安装依赖，再重新运行对比脚本。

## 文件索引

- `Scripts/Data_generate_D2.py`：生成 `Dims2`
- `Scripts/Data_generate_D2_mul.py`：生成 `Dims2_Mul`
- `Scripts/generate_tactile_test_sets.py`：批量重建测试集
- `Scripts/tactile_rotation/dataset.py`：数据读取
- `Scripts/tactile_rotation/models.py`：baseline 网络
- `Scripts/tactile_rotation/equivariant.py`：equivariant 网络
- `Scripts/tactile_rotation/train_baseline.py`：baseline 训练
- `Scripts/tactile_rotation/train_equivariant.py`：equivariant 训练
- `Scripts/tactile_rotation/evaluate.py`：单模型评估
- `Scripts/tactile_rotation/compare_checkpoints.py`：四模型对比
