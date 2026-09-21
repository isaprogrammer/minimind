# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MiniMind trains a small LLM (minimind-3: ~64M params, 768 hidden × 8 layers, vocab 6400) from scratch in pure PyTorch, aligned with the Qwen3 architecture/ecosystem. Comments, argparse help strings, and README are primarily Chinese. There is no test suite or linter — verification means running a trainer briefly or `eval_llm.py`.

## Critical: scripts only work from their own directory

Default paths are relative to the script's directory — this is the #1 gotcha:
- `trainer/train_*.py` → run from `trainer/` (weights → `../out`, data ← `../dataset`, tokenizer ← `../model`, resume ckpts → `../checkpoints`)
- `eval_llm.py` → run from repo root (`out/`, `model/`)
- `scripts/eval_toolcall.py`, `serve_openai_api.py`, `chat_api.py`, `convert_model.py` → run from `scripts/` (`../out`, `../model`)
- `scripts/web_demo.py` → run from `scripts/` AND requires a transformers-format model folder copied into `scripts/` first

New scripts in these packages follow the pattern at the top of every file: `__package__ = "<pkg>"` + `sys.path.append(...repo root...)`.

## Setup

```bash
pip install -r requirements.txt   # torch/peft are intentionally commented out — install torch separately for your CUDA/MPS setup
```

Datasets are NOT in the repo. Download jsonl files from ModelScope `gongjy/minimind_dataset` or HF `jingyaogong/minimind_dataset` into `dataset/`. Core files: `pretrain_t2t_mini.jsonl`, `sft_t2t_mini.jsonl`, `rlaif.jsonl`, `dpo.jsonl`, `agent_rl.jsonl` (plus `agent_rl_math.jsonl`). PPO/GRPO/agent training additionally need the `internlm2-1_8b-reward` model downloaded to a directory **sibling to the repo root** (default `--reward_model_path ../../internlm2-1_8b-reward`, relative to `trainer/`).

## Training pipeline

Stages in order (all from `trainer/`; `torchrun --nproc_per_node N` for multi-GPU DDP, plain `python` works for single device):

```bash
cd trainer
python train_pretrain.py      # pretrain_{dim}.pth        (from scratch, pretrain_t2t_mini.jsonl)
python train_full_sft.py      # full_sft_{dim}.pth        (--from_weight pretrain; sft_t2t_mini.jsonl)
python train_lora.py          # lora_{name}_{dim}.pth     (--from_weight full_sft; lora_medical.jsonl)
python train_dpo.py           # dpo_{dim}.pth             (--from_weight full_sft; dpo.jsonl)
python train_ppo.py           # ppo_actor_{dim}.pth       (needs reward model)
python train_grpo.py          # grpo_{dim}.pth            (--loss_type grpo|cispo; needs reward model)
python train_agent.py         # agent_{dim}.pth           (multi-turn tool-call RL)
python train_distillation.py  # full_dist_{dim}.pth       (white-box teacher→student)
```

Weight filename convention: `{stage}_{hidden_size}{_moe?}.pth` in `out/` (gitignored). CLI flags `--from_weight`/`--save_weight`/`--lora_weight`/`--weight` take only the stage prefix — the `{dim}`/`_moe` suffix is appended automatically from `--hidden_size`/`--use_moe 1`. Model size is controlled per-run by `--hidden_size`/`--num_hidden_layers` (not by config files).

Other important flags: `--from_resume 1` (auto-resume from `checkpoints/{stage}_{dim}_resume.pth`, handles changed GPU count), `--use_wandb` (imports `swanlab as wandb` — SwanLab is the default logger, not wandb), `--device`, `--max_seq_len`, `--epochs`, `--batch_size`.

## Eval / inference

```bash
python eval_llm.py --weight full_sft                        # from repo root; interactive/auto chat, native .pth
python eval_llm.py --load_from ./minimind-3                 # transformers-format folder
python eval_llm.py --weight full_sft --lora_weight lora_medical --open_thinking 1
cd scripts && python eval_toolcall.py --weight full_sft     # tool-call eval (--weight agent after RL)
cd scripts && python serve_openai_api.py                    # OpenAI-compatible API (port 8998)
cd scripts && python convert_model.py                       # .pth ↔ transformers format; merge LoRA; Qwen3-compatible export
```

## Architecture

**`model/model_minimind.py`** — the entire model in one file: `MiniMindConfig` (PretrainedConfig subclass), RMSNorm, RoPE with optional YaRN inference scaling (`--inference_rope_scaling`), GQA attention with q/k-norm (SDPA fast path + manual masked fallback), SwiGLU `FeedForward`, `MOEFeedForward` (top-k routing, load-balance aux loss, straight-through top-1 gradient), `MiniMindForCausalLM` with a fully custom `generate()` (manual KV cache, temperature/top-k/top-p/repetition penalty, streamer support). Subclasses `PreTrainedModel`+`GenerationMixin` but training loops are hand-written — no HF Trainer/trl/peft anywhere in the core path.

- The forward returns `MoeCausalLMOutputWithPast`; trainers must use `res.loss + res.aux_loss` (aux_loss is zero for dense models).
- Parameter names are deliberately transformers/Qwen3-aligned: compatibility with `AutoModelForCausalLM`, vLLM, llama.cpp, ollama is a hard constraint. Renaming modules breaks `scripts/convert_model.py` and every released checkpoint.

**`model/model_lora.py`** — hand-rolled LoRA (`apply_lora`/`load_lora`/`save_lora`/`merge_lora`) that monkey-patches forwards on square `nn.Linear` layers; no peft dependency.

**`dataset/lm_dataset.py`** — dataset classes per stage: `PretrainDataset` (`{"text": ...}`), `SFTDataset`/`DPODataset` (`{"conversations": [...]}` with `tools` on system message and `tool_calls` on assistant, rendered through `tokenizer.apply_chat_template`), `RLAIFDataset` (prompt-only), `AgentRLDataset`. **Loss masking is token-level**: SFT/DPO labels are computed by scanning input_ids for the exact spans `<bos>assistant\n` … `<eos>\n` and masking everything else to -100. Any change to the chat template or tokenizer breaks label generation — keep those spans intact. `post_processing_chat` randomly strips empty `<think>\n\n</think>\n\n` blocks; adaptive thinking is purely template-level (`open_thinking` flag), not a separate model.

**`trainer/`** — each `train_*.py` is a standalone script with its own argparse, DDP (`init_distributed_mode` detects `torchrun` via `RANK` env), AMP (`bfloat16`/`float16` GradScaler), cosine LR (`get_lr`), grad accumulation, and checkpointing. Shared in `trainer_utils.py`: `init_model` (builds model + loads `{from_weight}_{dim}.pth`), `lm_checkpoint` (atomic save of both the half-precision weight file and the full resume checkpoint incl. optimizer/scaler/wandb_id), `Logger` (rank-0 only), `SkipBatchSampler` (fast-forward on resume), `LMForRewardModel` (InternLM2 reward wrapper), `safe_math_eval` (sandboxed AST evaluator for the agent's `calculate_math` tool — use this instead of `eval`). `rollout_engine.py` abstracts generation for RL: local torch rollout or an external sglang server over HTTP (`--rollout_engine sglang`).

## Conventions

- Commit messages use `[update]`/`[fix]` prefixes (Chinese body is fine).
- `out/` (weights) and `dataset/*.jsonl` are gitignored/not committed; never assume they exist.
- `train_tokenizer.py` exists but retraining the tokenizer is explicitly discouraged — it invalidates all released weights and the chat template ecosystem.
