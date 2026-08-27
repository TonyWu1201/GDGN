# GDGN Ver2 严格评估

本分支实现了无泄漏的六类评估协议、训练折内预处理、统一实验注册、强基线、M0–M4 逐级模型、图/ESM/预训练/模态消融、层级统计与忠实性干预。本地只运行 smoke test，正式 5 折 × 3 种子训练在远程 GPU 执行。

常用入口：

```bash
uv sync --all-groups
uv run pytest tests/strict_eval -q
uv run python program/smoke_strict_pipeline.py
bash scripts/prepare_ver2_data.sh
bash scripts/run_screening.sh
bash scripts/run_confirmation.sh configs/sweeps/benchmark.yaml 4
```

详细顺序、数据前置条件、GPU 资源建议、中断恢复和结果汇总见 `guidance/完成总结/Ver2/远程GPU训练说明.md`。
