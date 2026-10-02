# Structured Optical Prior

论文 **Interference-enabled structured optical priors in dual-branch photonic networks** 的配套研究代码。

- 作者：Hongbo Chen
- 期刊：Journal of the Optical Society of America B，2026
- 论文：[10.1364/JOSAB.607960](https://doi.org/10.1364/JOSAB.607960)
- 许可：[MIT](LICENSE)
- 完整命令：[English README](README.md)

本仓库整理自作者提供的 Neuroptica 代码包，保留原有模型、训练配置和实验逻辑。包含单/双分支模型、相干/非相干读出、基底与输入表示对照、相位噪声和器件非理想性扫描，以及绘图脚本。仓库提供源代码，不附带训练权重、数值结果表或出版社校样。

## Windows 安装

使用 Python 3.9 至 3.11，在仓库根目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_paper.py --help
```

## 运行

两轮训练的运行检查，仍使用完整 MNIST 数据集：

```powershell
.\.venv\Scripts\python.exe run_paper.py --exp smoke --epochs 2 --seeds 0
```

运行原代码中的主要实验：

```powershell
.\.venv\Scripts\python.exe run_paper.py --exp all_core --epochs 300 --seeds 0,1,2
```

首次训练通过 Keras 下载 MNIST，之后读取本地缓存；无需 GPU。结果默认写入 `results/`，该目录中的生成文件不会被 Git 自动提交。不同配置建议使用不同的 `--out_dir`。

此版本中的 `best_acc` 为训练过程中最高的测试准确率，`final_acc` 为最后一轮的测试准确率。绘图文件沿用原代码包编号，具体内容以脚本的实验与面板说明为准。

相位噪声、器件扫描和绘图命令见 [README](README.md#run-experiments)。引用格式见 [Citation](README.md#citation)。
