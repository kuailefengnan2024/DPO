# ==================================================================================================
# DiffusionDPO 项目最终 DOCKERFILE
#
# 作者: Gemini, 基于一次协作式调试会话.
# 日期: 2025-08-07
#
# 本 Dockerfile 是一次深度调试过程的最终产物。下面的注释解释了每一条命令背后的逻辑，
# 并关联了我们共同解决的具体错误，以便于未来的维护。
# =================================C==================================================================

# --------------------------------------------------------------------------------------------------
# 阶段 1: 基础镜像选择
# --------------------------------------------------------------------------------------------------
# # 已解决的问题:
# - 由容器内 CUDA 工具包、主机 NVIDIA 驱动和 PyTorch 预编译二进制包不匹配导致的 "Segmentation fault" (段错误) 和 "Illegal instruction" (非法指令) 错误。
# - 因使用了不存在的 Docker 标签 (tag) 而导致的 "Image not found" (镜像未找到) 错误。
#
# # 决策逻辑:
# 主机拥有一块 NVIDIA RTX 4090 显卡，其驱动支持 CUDA 12.8。为确保最佳兼容性，我们必须选用
# 一个主版本号匹配 (12.x) 的基础镜像。我们最终选择了 `12.4.1-cudnn-devel-ubuntu22.04`，
# 因为这是一个已知的、稳定的官方标签，与所有 CUDA 12.x 版本的驱动完全兼容。
# `devel` 版本包含了完整的 CUDA 工具链，这在未来需要从源码编译某些库时会非常有用。
#
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# --------------------------------------------------------------------------------------------------
# 阶段 2: 环境与系统依赖
# --------------------------------------------------------------------------------------------------
# # 已解决的问题:
# - `apt-get` 在安装过程中因弹出交互式提问而无限期挂起。
# - 后续步骤中出现的 `curl: not found` (命令未找到) 错误。
#
# # 决策逻辑:
# - 设置 `TZ` 和 `DEBIAN_FRONTEND` 两个环境变量，是为了防止容器在构建过程中停下来，
#   等待用户输入 (例如选择时区)。
# - 我们明确地安装 `python3.9` 和 `curl`。`curl` 是在下一阶段下载 pip 官方安装脚本所必需的。
# - `-o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"` 这段参数至关重要。
#   它强制 `dpkg` (apt-get的底层工具) 在遇到配置文件冲突时自动选择默认行为，
#   从而解决了 `dpkg` 因等待用户输入而构建失败的问题。
# - `--no-install-recommends` 是一个最佳实践，可以保持最终镜像的体积更小。
# - `rm -rf /var/lib/apt/lists/*` 是另一个最佳实践，用于清理 apt 缓存，减小镜像体积。
#
ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" gnupg ca-certificates && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776 && \
    echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------------------------------
# 阶段 3: PYTHON 与 PIP 的安装
# --------------------------------------------------------------------------------------------------
# # 已解决的问题:
# - `python3.9: not found` 以及 `apt-get install python3-pip` 方式的普遍不可靠性。
#
# # 决策逻辑:
# 不再依赖系统自带的 `python3-pip` 包 (这可能导致版本不匹配)，最稳健的方法是使用
# 官方的 `get-pip.py` 引导脚本。我们用 `curl` 下载它，并用我们指定的 `python3.9`
# 来执行。这确保了 `pip` 被正确地安装，并与其对应的 Python 解释器完美关联。
#
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# --------------------------------------------------------------------------------------------------
# 阶段 4: 为方便起见，建立软链接 (SYMLINKS)
# --------------------------------------------------------------------------------------------------
# # 决策逻辑:
# 创建软链接，使得后续的命令以及用户进入容器后，可以直接使用 `python` 和 `pip` 命令，
# 而无需每次都输入完整的版本号。这提升了可读性和便利性。
#
RUN ln -s /usr/bin/python3.9 /usr/bin/python && \
    ln -s /usr/local/bin/pip /usr/bin/pip

# --------------------------------------------------------------------------------------------------
# 阶段 5: 应用程序设置
# --------------------------------------------------------------------------------------------------
WORKDIR /app
COPY requirements.txt .

# --------------------------------------------------------------------------------------------------
# 阶段 6: PYTHON 依赖安装
# --------------------------------------------------------------------------------------------------
# # 已解决的问题:
# - 最初因网络问题导致的 `pip` 下载失败 (`Read timed out`)。
# - 即使使用了正确的 CUDA 基础镜像，依然顽固出现的 `Segmentation fault` 错误。
#
# # 决策逻辑:
# 这是最关键的阶段，我们采用了两步走的策略：
#
# 1. 首先独立安装 PyTorch:
#    我们先安装 `torch`, `torchvision`, 和 `torchaudio`。最关键的是，我们使用了
#    `--index-url https://download.pytorch.org/whl/cu121` 这个参数。它强制 pip
#    从一个专门为 CUDA 12.1 编译的软件源下载，这些版本与我们的基础镜像和主机驱动兼容。
#    这避免了 pip 从标准源中选择一个不兼容版本，从而根治了最终的 `Segmentation fault` 问题。
#
# 2. 安装其余的依赖:
#    在正确的 PyTorch 版本就位后，我们再安装 `requirements.txt` 中的其他包。
#    Pip 会识别出 torch/torchvision 的依赖已经满足，因此会跳过它们，不会再画蛇添足。
#
# - `-i ... --trusted-host ...`: 使用国内的清华镜像源来加速下载，解决了最初的网络超时问题。
# - `--no-cache-dir`: 避免由损坏的缓存引起的问题，并保持镜像体积小。
#
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn \
    -r requirements.txt

# --------------------------------------------------------------------------------------------------
# 阶段 7: 最终的应用程序设置
# --------------------------------------------------------------------------------------------------
# 将项目代码的其余部分复制到容器中。
COPY . .

# 设置容器启动时执行的默认命令。
CMD ["python", "train.py"]
