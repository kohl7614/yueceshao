"""重排质量门 τ 的标定资产（对应 docs/rag-rerank-标定.md）。

- ``calibrate_rerank.py``：可复跑的标定脚本（稠密 top-40 → 重排 sigmoid → 取最高分 →
  A/B/C 分组统计漏放/误拦 → 输出建议 τ 区间）。
- ``queries-B组.txt``：标定文档 §5 全文列出的 B 组 10 条查询。
- ``queries-A组.txt`` / ``queries-C组.txt``：**待原作者补充**（标定文档未全文列出）。
"""
