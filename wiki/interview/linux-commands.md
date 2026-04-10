---
title: Linux 常用命令速查手册
type: interview
tags: [linux, 命令行, GPU, 进程管理, 面试]
created: 2026-04-10
updated: 2026-04-10
sources: []
status: active
---

# Linux 常用命令速查手册

> 面向深度学习研究 & 算法岗面试，重点覆盖：GPU 监控、内存/CPU 管理、进程管理、文件操作、网络、磁盘、环境管理。

---

## 1. GPU / 显卡状态查看 🔴高频

### 1.1 `nvidia-smi` — GPU 核心监控工具

```bash
# 基础用法：查看所有 GPU 状态
nvidia-smi

# 输出示例：
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2     |
# +-----------------------------------------------------------------------------+
# | GPU  Name        Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
# |   0  NVIDIA A100-SXM4-80GB  On | 00000000:07:00.0 Off |                    0 |
# | N/A   35C    P0    62W / 400W  |  12345MiB / 81920MiB |      45%      Default|
# +-----------------------------------------------------------------------------+
```

**关键字段解读：**
| 字段 | 含义 |
|------|------|
| `GPU-Util` | GPU 计算核心利用率（0% 表示空闲，100% 满载） |
| `Memory-Usage` | 显存使用量 / 总显存（如 `12345MiB / 81920MiB`） |
| `Temp` | GPU 温度（> 85°C 需要注意散热） |
| `Pwr:Usage/Cap` | 功耗 / 功耗上限 |
| `Perf` | 性能状态（P0 最高，P8 最低） |
| `Persistence-M` | 持久化模式（On = 驱动常驻，减少启动延迟） |
| `Compute M.` | 计算模式（Default / Exclusive_Process） |

```bash
# 持续监控（每 1 秒刷新）
nvidia-smi -l 1
# 或
watch -n 1 nvidia-smi

# 只看 GPU 利用率和显存
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
# 输出示例：
# 0, NVIDIA A100-SXM4-80GB, 45 %, 12345 MiB, 81920 MiB, 35

# 查看哪些进程在用 GPU
nvidia-smi pmon -i 0        # 监控 GPU 0 的进程
nvidia-smi pmon -s um -i 0  # 只看显存和利用率

# 查看特定 GPU
nvidia-smi -i 0             # 只看 GPU 0
nvidia-smi -i 0,1           # 看 GPU 0 和 1

# 查看 GPU 拓扑（多卡通信）
nvidia-smi topo -m
```

### 1.2 `gpustat` — 更简洁的 GPU 监控

```bash
# 安装
pip install gpustat

# 基础用法
gpustat

# 输出示例：
# [0] NVIDIA A100-SXM4-80GB | 35°C, 45 % | 12345 / 81920 MB | user1(11234M) user2(1111M)

# 持续监控
gpustat -i 1          # 每秒刷新
gpustat --watch       # watch 模式
gpustat -cup          # 显示命令名、用户、PID
```

### 1.3 `nvtop` — GPU 的 htop（交互式）

```bash
# 安装（Ubuntu）
sudo apt install nvtop

# 直接运行
nvtop
# 交互式界面，类似 htop，可实时看 GPU 利用率、显存、进程
```

### 1.4 CUDA 相关

```bash
# 查看 CUDA 版本
nvcc --version
# 或
cat /usr/local/cuda/version.txt

# 查看 CUDA 运行时版本（nvidia-smi 右上角显示的是驱动支持的最高 CUDA 版本）
# 实际用的版本取决于 conda/pip 安装的 pytorch 编译时的 CUDA 版本

# Python 中检查
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

# 查看 cuDNN 版本
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
# 或 Python 中
python -c "import torch; print(torch.backends.cudnn.version())"
```

### 1.5 指定 GPU 运行程序

```bash
# 环境变量指定（最常用）
CUDA_VISIBLE_DEVICES=0 python train.py           # 只用 GPU 0
CUDA_VISIBLE_DEVICES=0,1 python train.py         # 用 GPU 0 和 1
CUDA_VISIBLE_DEVICES="" python train.py           # 不用任何 GPU

# 在脚本中设置
export CUDA_VISIBLE_DEVICES=2,3
python train.py

# Python 代码中（不推荐，但有时需要）
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# 必须在 import torch 之前设置！
```

---

## 2. 内存 / CPU 监控 🔴高频

### 2.1 `free` — 内存概览

```bash
# 以人类可读格式显示
free -h

# 输出示例：
#               total        used        free      shared  buff/cache   available
# Mem:          251Gi        45Gi        12Gi       1.2Gi       194Gi       203Gi
# Swap:          8Gi        0.5Gi        7.5Gi

# 关键概念：
# - total = used + free + buff/cache
# - available ≈ free + buff/cache 中可回收部分（这才是真正可用的）
# - buff/cache：Linux 用空闲内存做磁盘缓存，需要时会自动释放
# - Swap：交换分区，内存不够时把数据换到磁盘（很慢，出现大量 swap 说明内存不够）

# 持续监控
watch -n 1 free -h

# 以 MB 显示
free -m

# 以 GB 显示
free -g
```

### 2.2 `top` / `htop` — 实时系统监控

```bash
# top（系统自带）
top

# top 常用快捷键：
# M — 按内存排序
# P — 按 CPU 排序
# 1 — 显示每个 CPU 核心
# k — 杀进程（输入 PID）
# q — 退出

# top 关键列：
# PID    — 进程 ID
# USER   — 所属用户
# %CPU   — CPU 使用率
# %MEM   — 内存使用率
# RES    — 实际物理内存占用
# VIRT   — 虚拟内存（通常很大，不用太关注）
# COMMAND — 命令名

# htop（增强版，推荐）
sudo apt install htop
htop

# htop 优势：
# - 彩色显示，顶部有 CPU/内存 柱状图
# - 支持鼠标操作
# - 树形显示进程关系（F5）
# - 搜索进程（F3）
# - 过滤进程（F4）
# - 排序（F6）
# - 杀进程（F9）

# 只看某个用户的进程
htop -u username

# 只看特定 PID
htop -p 1234,5678
```

### 2.3 `vmstat` — 虚拟内存统计

```bash
# 每秒刷新一次
vmstat 1

# 输出字段说明：
# procs: r(运行队列) b(阻塞)
# memory: swpd(swap使用) free(空闲) buff(缓冲) cache(缓存)
# swap: si(换入) so(换出) — 如果 si/so 持续非零，说明内存不足
# cpu: us(用户态) sy(内核态) id(空闲) wa(等IO)
```

### 2.4 `/proc/meminfo` — 详细内存信息

```bash
cat /proc/meminfo

# 重点关注：
# MemTotal:       263xxx kB    # 总内存
# MemFree:        12xxx kB     # 完全空闲
# MemAvailable:   203xxx kB    # 实际可用（推荐看这个）
# Buffers:        xxx kB
# Cached:         xxx kB
# SwapTotal:      xxx kB
# SwapFree:       xxx kB
```

---

## 3. 进程管理 🔴高频

### 3.1 `ps` — 查看进程

```bash
# 查看所有进程（最常用）
ps aux

# 字段说明：
# USER  PID  %CPU  %MEM  VSZ  RSS  TTY  STAT  START  TIME  COMMAND
# RSS = 实际物理内存占用（KB）

# 按内存排序（降序）
ps aux --sort=-%mem | head -20

# 按 CPU 排序
ps aux --sort=-%cpu | head -20

# 查看特定用户的进程
ps -u username

# 树形显示进程关系
ps auxf
# 或
pstree -p

# 查找特定进程
ps aux | grep python
ps aux | grep train.py

# 只看 PID 和命令
ps -eo pid,user,%cpu,%mem,cmd --sort=-%mem | head -20

# 查看线程
ps -T -p <PID>     # 某个进程的所有线程
```

### 3.2 查看"谁在跑什么程序"

```bash
# 方法1：查看所有用户的 GPU 进程
nvidia-smi

# 方法2：看所有 python 进程
ps aux | grep python

# 方法3：看所有 GPU 进程的详细命令行
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# 然后用 PID 查完整命令：
cat /proc/<PID>/cmdline | tr '\0' ' '
# 或
ps -p <PID> -o pid,user,%cpu,%mem,cmd

# 方法4：用 fuser 看谁在用 GPU 设备
fuser -v /dev/nvidia*

# 方法5：综合脚本 — 查看所有 GPU 进程的详细信息
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while read line; do
    pid=$(echo $line | cut -d',' -f1 | tr -d ' ')
    mem=$(echo $line | cut -d',' -f2)
    if [ -n "$pid" ]; then
        user=$(ps -o user= -p $pid 2>/dev/null)
        cmd=$(ps -o cmd= -p $pid 2>/dev/null)
        echo "PID=$pid USER=$user GPU_MEM=$mem CMD=$cmd"
    fi
done
```

### 3.3 杀进程

```bash
# 温和终止（SIGTERM，允许进程清理）
kill <PID>
kill -15 <PID>       # 等价

# 强制终止（SIGKILL，立即杀死）
kill -9 <PID>

# 按名称杀
killall python       # 杀所有 python 进程（危险！）
pkill -f "train.py"  # 杀命令行包含 train.py 的进程
pkill -u username     # 杀某用户所有进程（更危险！）

# 组合：杀掉占用某个 GPU 的所有进程
nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | xargs kill -9
```

### 3.4 后台运行 & 守护

```bash
# nohup — 断开终端后继续运行
nohup python train.py > train.log 2>&1 &
# 2>&1 = stderr 也重定向到 stdout
# & = 放到后台

# 查看后台任务
jobs -l

# 前后台切换
Ctrl+Z        # 挂起当前进程（暂停）
bg            # 把挂起的进程放到后台继续跑
fg            # 把后台进程拿回前台
fg %1         # 恢复 job 1

# screen — 虚拟终端（推荐）
screen -S train           # 创建名为 train 的 session
# 在 screen 里运行训练...
Ctrl+A, D                 # 分离（detach），回到原终端
screen -ls                # 列出所有 session
screen -r train           # 重新连接
screen -X -S train quit   # 关闭 session

# tmux — 更现代的终端复用器（推荐）
tmux new -s train         # 创建 session
# 运行训练...
Ctrl+B, D                 # 分离
tmux ls                   # 列出 sessions
tmux attach -t train      # 重新连接
tmux kill-session -t train # 关闭

# tmux 常用操作：
# Ctrl+B, c     — 新建窗口
# Ctrl+B, n/p   — 下/上一个窗口
# Ctrl+B, %     — 左右分屏
# Ctrl+B, "     — 上下分屏
# Ctrl+B, 方向键 — 切换分屏
# Ctrl+B, z     — 最大化/恢复当前分屏
```

### 3.5 `lsof` — 查看进程打开的文件

```bash
# 查看某进程打开的所有文件
lsof -p <PID>

# 查看谁在用某个端口
lsof -i :8080

# 查看某用户打开的文件
lsof -u username

# 查看某个文件被谁打开
lsof /path/to/file
```

---

## 4. 文件与目录操作 🟡中频

### 4.1 基础操作

```bash
# 列出文件
ls -la             # 详细信息 + 隐藏文件
ls -lh             # 人类可读的文件大小
ls -lt             # 按修改时间排序
ls -lS             # 按文件大小排序
ls -R              # 递归列出

# 切换目录
cd /path/to/dir
cd ~               # 回到 home
cd -               # 回到上一个目录
cd ..              # 上一级

# 创建/删除
mkdir -p a/b/c     # 递归创建多级目录
rm -rf dir/        # 强制递归删除（危险！）
rm -i file         # 交互确认后删除

# 复制/移动
cp -r src/ dst/    # 递归复制目录
cp -a src/ dst/    # 保留权限和时间戳
mv old new         # 移动/重命名
rsync -avz src/ dst/  # 增量同步（推荐大文件/目录复制）

# 软链接
ln -s /real/path /link/path    # 创建软链接
ls -la /link/path              # 查看链接指向
```

### 4.2 查找文件

```bash
# find — 按条件查找
find . -name "*.py"                    # 当前目录下所有 .py 文件
find . -name "*.py" -mtime -7         # 最近 7 天修改的
find . -size +100M                     # 大于 100MB 的文件
find . -type d -name "__pycache__"     # 找所有 __pycache__ 目录
find . -name "*.pyc" -delete           # 找到并删除

# locate — 从数据库查找（更快）
sudo updatedb        # 先更新数据库
locate train.py

# which / whereis — 查找命令位置
which python
whereis python
```

### 4.3 查看文件内容

```bash
# 查看整个文件
cat file.txt
less file.txt       # 可翻页（推荐）
more file.txt       # 简单翻页

# 查看前/后 N 行
head -n 20 file.txt
tail -n 20 file.txt
tail -f train.log    # 实时追踪日志（非常常用！）

# 搜索文件内容
grep "error" train.log
grep -r "import torch" .         # 递归搜索
grep -n "def forward" model.py   # 显示行号
grep -i "ERROR" train.log        # 忽略大小写
grep -c "epoch" train.log        # 计数匹配行数
grep -A 3 -B 1 "Error" log.txt  # 显示前1行后3行上下文

# ripgrep（更快的 grep，推荐安装）
rg "import torch" --type py
rg "learning_rate" -g "*.yaml"
```

### 4.4 文件权限

```bash
# 查看权限
ls -la
# -rwxr-xr-- 1 user group  size date  filename
# rwx = 读(4)写(2)执行(1)，三组：所有者/组/其他

# 修改权限
chmod 755 script.sh     # rwxr-xr-x
chmod +x script.sh      # 添加执行权限
chmod -R 644 dir/       # 递归修改

# 修改所有者
chown user:group file
chown -R user:group dir/
```

---

## 5. 磁盘管理 🟡中频

```bash
# 查看磁盘使用情况
df -h                    # 所有挂载点
df -h /home              # 指定目录所在分区

# 查看目录大小
du -sh *                 # 当前目录下每个子项的大小
du -sh /home/user/       # 某个目录的总大小
du -h --max-depth=1 .    # 只看一层
du -sh * | sort -rh      # 按大小排序

# ncdu — 交互式磁盘使用分析器（推荐）
sudo apt install ncdu
ncdu /home/user/

# 查找大文件
find / -size +1G -type f 2>/dev/null | head -20
du -ah . | sort -rh | head -20

# 常见清理操作（深度学习场景）
# 清理 pip 缓存
pip cache purge

# 清理 conda 缓存
conda clean --all

# 清理 PyTorch 模型缓存
rm -rf ~/.cache/torch/hub/
rm -rf ~/.cache/huggingface/

# 清理 __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} +

# 清理 .pyc 文件
find . -name "*.pyc" -delete
```

---

## 6. 网络 🟡中频

```bash
# 查看网络接口
ip addr
ifconfig          # 旧命令，可能需要安装 net-tools

# 查看端口占用
ss -tlnp                  # 所有 TCP 监听端口
netstat -tlnp             # 旧命令，效果同上
lsof -i :8080             # 查看谁在用 8080 端口

# 测试连通性
ping google.com
curl -I https://google.com    # 只看 HTTP 头

# 下载
wget https://example.com/file.tar.gz
wget -c url          # 断点续传
curl -O url          # 下载文件
curl -L url          # 跟随重定向

# SSH
ssh user@server
ssh -p 2222 user@server           # 指定端口
ssh -L 8888:localhost:8888 user@server  # 端口转发（常用于 Jupyter）
ssh -N -f -L 6006:localhost:6006 user@server  # TensorBoard 端口转发

# SCP / rsync — 文件传输
scp file.py user@server:/path/           # 上传
scp user@server:/path/file.py .          # 下载
scp -r dir/ user@server:/path/           # 上传目录
rsync -avz --progress dir/ user@server:/path/  # 增量同步（推荐）

# SSH 免密登录
ssh-keygen -t rsa -b 4096               # 生成密钥对
ssh-copy-id user@server                  # 复制公钥到服务器
```

---

## 7. 压缩与解压 🟡中频

```bash
# tar
tar -czf archive.tar.gz dir/        # 压缩（gzip）
tar -cjf archive.tar.bz2 dir/       # 压缩（bzip2，更小但更慢）
tar -xzf archive.tar.gz             # 解压 gzip
tar -xzf archive.tar.gz -C /target/ # 解压到指定目录
tar -xjf archive.tar.bz2            # 解压 bzip2
tar -tf archive.tar.gz              # 查看内容（不解压）

# zip
zip -r archive.zip dir/
unzip archive.zip
unzip archive.zip -d /target/

# 7z（压缩率更高）
7z a archive.7z dir/
7z x archive.7z
```

---

## 8. 系统信息 🟢低频

```bash
# 系统版本
uname -a                    # 内核版本
cat /etc/os-release         # 发行版信息
lsb_release -a              # Ubuntu 版本

# CPU 信息
lscpu                       # CPU 概要
cat /proc/cpuinfo           # 详细信息
nproc                       # CPU 核心数

# 内存信息
free -h
cat /proc/meminfo

# 磁盘信息
lsblk                       # 块设备列表
fdisk -l                    # 分区信息

# PCI 设备（含 GPU）
lspci | grep -i nvidia      # 查看 NVIDIA GPU 型号

# 系统运行时间 & 负载
uptime
# 输出：10:30:00 up 30 days, load average: 2.50, 3.10, 2.80
# load average: 1分钟, 5分钟, 15分钟 平均负载
# 一般 load average < CPU 核心数 表示正常

# 查看谁在登录
who
w                  # 更详细，显示每个用户在做什么
last               # 登录历史
```

---

## 9. Python / Conda 环境管理 🔴高频

```bash
# Conda 环境
conda env list                        # 列出所有环境
conda create -n myenv python=3.10     # 创建环境
conda activate myenv                  # 激活
conda deactivate                      # 退出
conda remove -n myenv --all           # 删除环境

# 包管理
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip list | grep torch                 # 查看已安装的 torch 相关包
pip show torch                        # 某个包的详细信息

# venv（Python 内置）
python -m venv .venv
source .venv/bin/activate
deactivate

# uv（快速 Python 包管理器）
uv pip install -r requirements.txt
uv pip list
uv venv .venv

# 查看 Python 路径
which python
python -c "import sys; print(sys.executable)"
python -c "import sys; print('\n'.join(sys.path))"
```

---

## 10. 文本处理 🟢低频

```bash
# wc — 统计
wc -l file.txt       # 行数
wc -w file.txt       # 词数
wc -c file.txt       # 字节数

# sort & uniq
sort file.txt
sort -n file.txt     # 数字排序
sort -rn file.txt    # 逆序
sort file.txt | uniq -c | sort -rn    # 统计频率

# awk — 列处理
awk '{print $1, $3}' file.txt         # 打印第1和第3列
awk -F',' '{print $2}' file.csv       # 用逗号分隔

# sed — 流编辑
sed 's/old/new/g' file.txt            # 替换（不修改原文件）
sed -i 's/old/new/g' file.txt         # 替换（原地修改）
sed -n '10,20p' file.txt              # 打印第10-20行

# cut — 按列切割
cut -d',' -f1,3 file.csv              # 取 CSV 第1和第3列

# 管道组合示例
cat train.log | grep "loss" | tail -10           # 看最近 10 条 loss
cat train.log | grep "epoch" | awk '{print $NF}' # 提取每行最后一个字段
```

---

## 11. 深度学习实用组合命令 🔴高频

### 11.1 训练监控一条龙

```bash
# 终端1：实时看 GPU
watch -n 1 nvidia-smi

# 终端2：实时看训练日志
tail -f output/train.log

# 终端3：看系统资源
htop
```

### 11.2 远程训练标准流程

```bash
# 1. SSH 登录服务器
ssh user@gpu-server

# 2. 创建 tmux session
tmux new -s train

# 3. 激活环境
conda activate myenv

# 4. 选择 GPU 并启动训练
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --config configs/experiment.yaml \
    --output_dir output/exp01 \
    2>&1 | tee output/exp01/train.log

# 5. Detach（Ctrl+B, D）安全退出

# 6. 之后重新连接查看
tmux attach -t train
```

### 11.3 批量杀显存泄漏进程

```bash
# 查看残留的 python GPU 进程
nvidia-smi | grep python

# 杀掉所有自己的 python GPU 进程
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I{} bash -c \
    'if [ "$(ps -o user= -p {})" = "$(whoami)" ]; then kill -9 {}; fi'
```

### 11.4 查看训练结果

```bash
# 查看最终 metrics
tail -1 output/exp01/results.json | python -m json.tool

# 统计 log 中的 loss 变化
grep "loss" train.log | awk '{print $NF}' | tail -20

# 查看 checkpoint 文件大小
ls -lh output/exp01/checkpoint-*

# 计算模型参数量
python -c "
import torch
model = torch.load('model.pt', map_location='cpu')
total = sum(p.numel() for p in model.values() if isinstance(p, torch.Tensor))
print(f'Total parameters: {total:,} ({total/1e6:.1f}M)')
"
```

### 11.5 多机多卡训练

```bash
# torchrun（PyTorch 原生）
torchrun --nproc_per_node=4 train.py --args...

# deepspeed
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json

# accelerate（HuggingFace）
accelerate launch --num_processes 4 train.py
```

---

## 12. 常见面试相关问答

### Q1：服务器上 GPU 被别人占满了怎么办？

1. `nvidia-smi` 看谁在用
2. `ps -p <PID> -o user,cmd` 查看具体用户和命令
3. 联系对方协商，或用 `CUDA_VISIBLE_DEVICES` 选择空闲卡
4. 如果有 SLURM 集群管理：`squeue` 查看队列，`sbatch` 提交任务

### Q2：训练到一半断了怎么恢复？

1. 检查是否有 checkpoint：`ls output/checkpoint-*`
2. 加 `--resume_from_checkpoint output/checkpoint-latest`
3. 预防措施：用 `tmux`/`screen`，设置定期保存 checkpoint

### Q3：OOM (Out of Memory) 怎么排查？

1. `nvidia-smi` 确认显存使用
2. 减小 `batch_size`
3. 使用 `gradient_accumulation_steps` 等效大 batch
4. 用 `torch.cuda.amp` 混合精度训练
5. 用 DeepSpeed ZeRO / FSDP 分布式
6. 用 `gradient_checkpointing` 以计算换显存

### Q4：如何查看服务器的硬件配置？

```bash
# 一键查看
echo "=== CPU ===" && lscpu | grep "Model name\|^CPU(s):" && \
echo "=== Memory ===" && free -h | head -2 && \
echo "=== GPU ===" && nvidia-smi -L && \
echo "=== Disk ===" && df -h / /home 2>/dev/null && \
echo "=== OS ===" && cat /etc/os-release | grep PRETTY_NAME
```

---

## 13. 快捷键 & 小技巧 🟢低频

```bash
# Bash 快捷键
Ctrl+C    # 中断当前命令
Ctrl+Z    # 挂起当前命令
Ctrl+D    # 退出 shell / 输入 EOF
Ctrl+L    # 清屏
Ctrl+A    # 光标移到行首
Ctrl+E    # 光标移到行尾
Ctrl+R    # 搜索历史命令
Ctrl+U    # 删除光标前所有内容
Ctrl+K    # 删除光标后所有内容
Ctrl+W    # 删除光标前一个单词
!!        # 重复上一条命令
sudo !!   # 用 sudo 重复上一条命令
!$        # 上一条命令的最后一个参数

# 历史命令
history             # 查看历史
history | grep ssh  # 搜索历史
!123                # 执行历史中第 123 条

# 别名（写入 ~/.bashrc）
alias gs='git status'
alias gp='git pull'
alias ns='nvidia-smi'
alias wns='watch -n 1 nvidia-smi'
alias py='python'
alias jn='jupyter notebook'
alias ta='tmux attach -t'
alias tl='tmux ls'

# 生效
source ~/.bashrc
```

---

*本文档由 LLM 整理，面向深度学习研究者 & 秋招面试场景，持续更新中。*
