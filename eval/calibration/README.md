# 重排质量门 τ 标定资产

对应 `docs/rag-rerank-标定.md` 记载的标定协议，把「改阈值前先复跑标定」从一人本机脚本变成仓内可复跑资产。

## 文件

| 文件 | 说明 |
|---|---|
| `calibrate_rerank.py` | 可复跑标定脚本：稠密 top-40 → Cross-Encoder 逐对打分 → sigmoid → 取最高分 → A/B/C 分组统计漏放/误拦 → 输出建议 τ 区间 |
| `queries-B组.txt` | 标定文档 §5 全文列出的 B 组（近域）10 条查询 |
| `queries-A组.txt` | **待原作者补充**（标定文档未全文列出 A 组 20 条） |
| `queries-C组.txt` | **待原作者补充**（标定文档未全文列出 C 组 10 条） |

> **诚实标注**：A/C 两组的查询快照目前**不在仓内**——标定文档只全文列出了 B 组 10 条，A 组（域内已覆盖 20 条）与 C 组（完全域外 10 条）只给了示例。缺 A/C 时，脚本只能统计 B 组，**算不出完整的 τ 建议区间**（区间下沿靠 C 组 max、上沿靠 A 组 min）。补齐 A/C 后重跑即可复现 τ=0.55。

## 用法

```bash
# 1. 权重放本地（不入库），models/ 下需就位：
#    models/Qwen3-Embedding-0.6B   （稠密检索）
#    models/bge-reranker-v2-m3     （Cross-Encoder）
# 2. 装依赖
pip install -r requirements.txt sentence-transformers
# 3. 在仓库根目录跑（40 条查询、CPU 约 15 分钟；缺 A/C 时只跑 B 组 10 条）
python -m eval.calibration.calibrate_rerank --config config.json
```

## 输出

- 三组最高分分布（min/中位/max）
- τ 扫表（误拦率 A / 漏放率 B+C）
- 建议 τ 区间（零误拦平台区，及「中部偏下」的建议值）

## 口径提醒（与标定文档 §6.1 一致）

脚本默认 `--top-k 40`，统计的是 **max@40**。线上默认 `max_attempts=2`（不过门会放大到 80 条重试），`max(sigmoid)` 随候选池单调不减，所以 **max@80 的漏放率会高于 max@40**。要按线上口径复核，需另跑 `--top-k 80` 并对照两列。
