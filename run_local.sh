#!/usr/bin/env bash
# ==============================================================================
# MiniMind 本地一键脚本（macOS Apple Silicon / Linux）
# 流程: 环境检查 → 安装 → 数据下载 → 冒烟测试 → 完整训练 → 对话评测
#
# 用法:
#   ./run_local.sh check    # 1. 检查本机硬件/软件环境
#   ./run_local.sh install  # 2. 创建 .venv，安装 torch + requirements.txt
#   ./run_local.sh data     # 3. 下载 mini 训练数据集(约 2.8GB)到 dataset/
#   ./run_local.sh smoke    # 4. 合成小数据快速验证 训练→SFT→推理 全链路(约10分钟)
#   ./run_local.sh train    # 5. 完整训练 pretrain + full_sft(M1 Pro 预计数天)
#   ./run_local.sh eval     # 6. 用训好的 full_sft 权重自动对话测试
#   ./run_local.sh all      # 以上全部按序执行
#
# 可选环境变量:
#   TORCH_VERSION=2.6.0     torch 版本(默认取 requirements.txt 注释中仓库测试过的版本)
#   DEVICE=mps|cpu|cuda     训练设备(默认自动检测: MPS 可用则 mps, 否则 cpu)
#   NUM_WORKERS=4           DataLoader 进程数
#   EPOCHS_PRETRAIN=2       预训练轮数
#   EPOCHS_SFT=2            SFT 轮数
#   RESUME=1                断点续训(透传 --from_resume 1)
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
VENV_PY="$(pwd)/$VENV/bin/python"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS_PRETRAIN="${EPOCHS_PRETRAIN:-2}"
EPOCHS_SFT="${EPOCHS_SFT:-2}"
RESUME="${RESUME:-0}"

C_GREEN=$'\033[32m'; C_RED=$'\033[31m'; C_YEL=$'\033[33m'; C_CYAN=$'\033[36m'; C_OFF=$'\033[0m'
info(){ echo "${C_CYAN}[INFO]${C_OFF} $*"; }
ok(){   echo "${C_GREEN}[ OK ]${C_OFF} $*"; }
warn(){ echo "${C_YEL}[WARN]${C_OFF} $*"; }
die(){  echo "${C_RED}[FAIL]${C_OFF} $*" >&2; exit 1; }

# macOS 上防止训练中途系统休眠
CAFFE=""
if [ "$(uname -s)" = "Darwin" ] && command -v caffeinate >/dev/null 2>&1; then
  CAFFE="caffeinate -is"
fi

fsize(){ stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null || echo 0; }

# 选择 Python 3.10 ~ 3.12
pick_python() {
  for cand in python3.11 python3.10 python3 python3.12; do
    command -v "$cand" >/dev/null 2>&1 || continue
    "$cand" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null || continue
    echo "$cand"; return 0
  done
  return 1
}

# 需要 venv 已存在; torch 缺失时兜底返回 cpu
detect_device() {
  "$VENV_PY" - <<'PY' 2>/dev/null || echo cpu
import torch
print("mps" if torch.backends.mps.is_available() else "cpu")
PY
}

# ==============================================================================
# 阶段 1: 环境检查
# ==============================================================================
stage_check() {
  echo "==================== 环境检查 ===================="
  OS="$(uname -s)"; ARCH="$(uname -m)"
  echo "系统:   $OS $ARCH"
  if [ "$OS" = "Darwin" ]; then
    echo "机型:   $(sysctl -n hw.model 2>/dev/null || echo Mac)"
    echo "芯片:   $(sysctl -n machdep.cpu.brand_string)"
    echo "核心:   $(sysctl -n hw.ncpu) 核"
    RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
  else
    echo "核心:   $(nproc 2>/dev/null || echo '?') 核"
    RAM_GB=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0) / 1024 / 1024 ))
  fi
  echo "内存:   ${RAM_GB} GB"
  FREE_GB=$(df -Pk . | awk 'NR==2{printf "%d", $4/1024/1024}')
  echo "磁盘剩余: ${FREE_GB} GB (需约 6GB: 环境 ~3GB + 数据 ~3GB)"

  PY_BIN=$(pick_python) || die "未找到 Python 3.10~3.12，请先安装: brew install python@3.11"
  echo "Python: $PY_BIN ($("$PY_BIN" --version 2>&1))"

  if [ -x "$VENV_PY" ]; then
    "$VENV_PY" -c 'import torch; print("torch:  ", torch.__version__, "| MPS 可用:", torch.backends.mps.is_available())' \
      || warn "venv 中 torch 异常，请重新运行 ./run_local.sh install"
    echo "训练设备: $(detect_device)"
  else
    warn "虚拟环境 $VENV 未创建 → 运行 ./run_local.sh install"
  fi

  [ "${RAM_GB:-0}" -ge 16 ] || warn "内存 ${RAM_GB}GB 偏小，建议 ≥16GB"
  [ "${FREE_GB:-0}" -ge 15 ] || warn "磁盘剩余 ${FREE_GB}GB 偏紧"
  ok "检查完成"
}

# ==============================================================================
# 阶段 2: 安装 (venv + torch + requirements.txt)
# ==============================================================================
stage_install() {
  PY_BIN=$(pick_python) || die "未找到 Python 3.10~3.12"
  info "使用 $PY_BIN 创建虚拟环境 $VENV ..."
  [ -x "$VENV_PY" ] || "$PY_BIN" -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip -q

  info "安装 torch==${TORCH_VERSION} (Apple Silicon 的 arm64 wheel 自带 MPS 支持)..."
  "$VENV/bin/pip" install "torch==${TORCH_VERSION}"

  info "安装 requirements.txt ..."
  "$VENV/bin/pip" install -r requirements.txt

  # 本地产物追加到 .gitignore
  for entry in ".venv/" "checkpoints/" "dataset/smoke_*.jsonl"; do
    grep -qxF "$entry" .gitignore 2>/dev/null || echo "$entry" >> .gitignore
  done

  "$VENV_PY" -c 'import torch, transformers; print("torch:", torch.__version__, "| transformers:", transformers.__version__, "| MPS:", torch.backends.mps.is_available())' \
    || die "依赖安装后自检失败"
  ok "安装完成，训练设备: $(detect_device)"
}

# ==============================================================================
# 阶段 3: 下载数据集 (ModelScope 优先, HF/HF-mirror 兜底)
# ==============================================================================
stage_data() {
  [ -x "$VENV_PY" ] || die "请先运行 ./run_local.sh install"
  mkdir -p dataset
  PRE=dataset/pretrain_t2t_mini.jsonl
  SFT=dataset/sft_t2t_mini.jsonl

  if [ "$(fsize "$PRE")" -gt 104857600 ] && [ "$(fsize "$SFT")" -gt 104857600 ]; then
    ok "数据集已存在，跳过下载"; return 0
  fi

  info "下载 mini 数据集约 2.8GB (pretrain_t2t_mini 1.2G + sft_t2t_mini 1.6G)，视网速可能较久 ..."
  if ! "$VENV/bin/modelscope" download --dataset gongjy/minimind_dataset \
        pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl --local_dir dataset; then
    warn "ModelScope 下载失败，改用 HuggingFace ..."
    HF_DL="$VENV/bin/hf"; [ -x "$HF_DL" ] || HF_DL="$VENV/bin/huggingface-cli"
    if ! "$HF_DL" download jingyaogong/minimind_dataset \
          pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl --repo-type dataset --local-dir dataset; then
      warn "HF 直连失败，最后尝试 hf-mirror 镜像 ..."
      HF_ENDPOINT=https://hf-mirror.com "$HF_DL" download jingyaogong/minimind_dataset \
        pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl --repo-type dataset --local-dir dataset
    fi
  fi

  [ "$(fsize "$PRE")" -gt 104857600 ] || die "$PRE 下载不完整，可手动下载: https://modelscope.cn/datasets/gongjy/minimind_dataset/files"
  [ "$(fsize "$SFT")" -gt 104857600 ] || die "$SFT 下载不完整，可手动下载: https://modelscope.cn/datasets/gongjy/minimind_dataset/files"
  ok "数据集就绪: $PRE ($(($(fsize "$PRE")/1048576))MB), $SFT ($(($(fsize "$SFT")/1048576))MB)"
}

# ==============================================================================
# 阶段 4: 冒烟测试 — 合成小数据跑通 训练→SFT→推理 全链路(不依赖数据集下载)
# 注: 模型在这些数据上学不到东西，输出乱码属正常，本阶段只验证代码链路
# ==============================================================================
stage_smoke() {
  [ -x "$VENV_PY" ] || die "请先运行 ./run_local.sh install"
  DEVICE=$(detect_device)
  [ "$DEVICE" = "mps" ] || warn "未检测到 MPS，将用 CPU 运行，冒烟测试会慢一些"
  info "设备: $DEVICE | 生成合成冒烟数据 ..."
  "$VENV_PY" - <<'PY'
import json, random
random.seed(42)
texts = [
    "人工智能正在改变人类与计算机交互的方式。",
    "深度学习模型通过大量数据自动学习特征表示。",
    " Transformer 架构是现代大语言模型的基石。",
    "注意力机制让模型能够关注输入序列中的不同部分。",
    "预训练加微调是当前主流的两阶段训练范式。",
    "语言模型的本质是对下一个词的概率分布建模。",
    "开源社区推动了小型语言模型的快速发展。",
    "强化学习可以让模型输出更符合人类偏好。",
    "知识蒸馏将大模型的能力迁移到小模型中。",
    "混合专家模型让每个 token 只激活部分参数。",
    "RMS 归一化有助于稳定深层网络的训练。",
    "旋转位置编码为注意力提供了相对位置信息。",
]
with open('dataset/smoke_pretrain.jsonl', 'w', encoding='utf-8') as f:
    for _ in range(200):
        f.write(json.dumps({"text": "。".join(random.sample(texts, 5)) + "。"}, ensure_ascii=False) + "\n")

qas = [
    ("你好，请介绍一下你自己。", "我是minimind，一个小巧但有用的语言模型，很高兴为你服务。"),
    ("什么是深度学习？", "深度学习是利用多层神经网络从数据中自动学习特征的方法。"),
    ("什么是注意力机制？", "注意力机制让模型在处理每个词时动态关注序列中最相关的部分。"),
    ("什么是预训练？", "预训练是在大规模无标注文本上先学习通用语言表示的过程。"),
    ("什么是知识蒸馏？", "知识蒸馏是把大模型学到的能力迁移到小模型的训练技术。"),
    ("什么是混合专家模型？", "混合专家模型包含多个专家网络，每个token只激活其中一部分。"),
    ("什么是强化学习？", "强化学习是通过与环境交互获得奖励来学习策略的方法。"),
    ("再见！", "再见，期待下次为你服务！"),
]
def msg(role, content):
    return {"role": role, "content": content, "reasoning_content": "", "tools": "", "tool_calls": ""}
with open('dataset/smoke_sft.jsonl', 'w', encoding='utf-8') as f:
    for i in range(150):
        q, a = qas[i % len(qas)]
        f.write(json.dumps({"conversations": [msg("user", q), msg("assistant", a)]}, ensure_ascii=False) + "\n")
print("已生成 dataset/smoke_pretrain.jsonl (200条) 和 dataset/smoke_sft.jsonl (150条)")
PY

  info "[1/3] 冒烟预训练 (200 条, 1 epoch) ..."
  (cd trainer && $CAFFE "$VENV_PY" train_pretrain.py --device "$DEVICE" --num_workers 2 \
      --data_path ../dataset/smoke_pretrain.jsonl --save_weight pretrain_smoke \
      --epochs 1 --batch_size 8 --max_seq_len 128 --log_interval 5)

  info "[2/3] 冒烟 SFT (150 条, 1 epoch, 基于 pretrain_smoke) ..."
  (cd trainer && $CAFFE "$VENV_PY" train_full_sft.py --device "$DEVICE" --num_workers 2 \
      --data_path ../dataset/smoke_sft.jsonl --from_weight pretrain_smoke --save_weight full_sft_smoke \
      --epochs 1 --batch_size 4 --max_seq_len 256 --log_interval 5)

  info "[3/3] 冒烟推理 (自动模式 8 个问题, 每条最多 32 token, 输出乱码属正常) ..."
  printf '0\n' | "$VENV_PY" eval_llm.py --weight full_sft_smoke --device "$DEVICE" \
      --max_new_tokens 32 --show_speed 1

  ok "冒烟测试全链路通过: out/pretrain_smoke_768.pth, out/full_sft_smoke_768.pth"
}

# ==============================================================================
# 阶段 5: 完整训练 (真实数据: pretrain_t2t_mini → sft_t2t_mini)
# ==============================================================================
stage_train() {
  [ -x "$VENV_PY" ] || die "请先运行 ./run_local.sh install"
  DEVICE=$(detect_device)
  for f in dataset/pretrain_t2t_mini.jsonl dataset/sft_t2t_mini.jsonl; do
    [ "$(fsize "$f")" -gt 104857600 ] || die "缺少 $f，请先运行 ./run_local.sh data"
  done

  EXTRA=""
  [ "$RESUME" = "1" ] && EXTRA="--from_resume 1"

  warn "参考: RTX 3090 上 pretrain_mini 约 1.2h/轮; ${DEVICE} 预计慢 10~20 倍, 完整训练可能需数天"
  warn "建议在 tmux/screen 中运行; 中断后可用 RESUME=1 ./run_local.sh train 续训(自动跳过已训 step)"

  info "[1/2] 预训练 (epochs=$EPOCHS_PRETRAIN, 设备=$DEVICE) ..."
  (cd trainer && $CAFFE "$VENV_PY" train_pretrain.py --device "$DEVICE" \
      --num_workers "$NUM_WORKERS" --epochs "$EPOCHS_PRETRAIN" $EXTRA)
  [ -f out/pretrain_768.pth ] || die "预训练未产出 out/pretrain_768.pth"

  info "[2/2] SFT (epochs=$EPOCHS_SFT) ..."
  (cd trainer && $CAFFE "$VENV_PY" train_full_sft.py --device "$DEVICE" \
      --num_workers "$NUM_WORKERS" --epochs "$EPOCHS_SFT" $EXTRA)
  [ -f out/full_sft_768.pth ] || die "SFT 未产出 out/full_sft_768.pth"

  ok "训练完成: out/pretrain_768.pth + out/full_sft_768.pth → 运行 ./run_local.sh eval 体验对话"
}

# ==============================================================================
# 阶段 6: 对话评测 (自动模式)
# ==============================================================================
stage_eval() {
  [ -x "$VENV_PY" ] || die "请先运行 ./run_local.sh install"
  [ -f out/full_sft_768.pth ] || die "未找到 out/full_sft_768.pth: 先 ./run_local.sh train, 或从 ModelScope gongjy/minimind-3-pytorch 下载权重放入 out/"
  DEVICE=$(detect_device)
  info "自动对话测试 (8 个内置问题, 每条最多 512 token, 设备=$DEVICE) ..."
  printf '0\n' | "$VENV_PY" eval_llm.py --weight full_sft --device "$DEVICE" \
      --max_new_tokens 512 --show_speed 1
  ok "评测完成; 交互式对话请手动运行: python eval_llm.py --weight full_sft --device $DEVICE"
}

usage() { sed -n '2,25p' "$0" | cut -c3-; }

case "${1:-}" in
  check)   stage_check ;;
  install) stage_install ;;
  data)    stage_data ;;
  smoke)   stage_smoke ;;
  train)   stage_train ;;
  eval)    stage_eval ;;
  all)     stage_check; stage_install; stage_data; stage_smoke; stage_train; stage_eval ;;
  -h|--help|"") usage ;;
  *) usage; die "未知阶段: $1" ;;
esac
