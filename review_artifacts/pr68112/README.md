# PR #68112：线性边界查找候选

本分支以 upstream PR #68112 的提交
`cbc5b29ec71c645855551a133e0d22a96c2c35a4` 为基础，保留其第一次
reasoning-end 前缀策略，并尝试降低边界检查开销。

修改包括：

- manager 每次向 streaming 检查传入单个 token，删除重复前缀列表。
- `TokenIdsView` 按需读取索引和切片，避免短切片触发完整历史复制。

对于使用固定长度标记窗口的 K3，候选算法由 O(H+n²) 降至 O(n)；
其他 parser 如果仍扫描完整上下文，不享有这个复杂度保证。

## 验证范围

这是供审阅的候选分支，尚未完成共享 parser、真实 grammar mask、投机验收、
模型 serving、准确率和性能验证。算法探针不能替代这些验证。

从仓库根目录运行：

```bash
uv venv --python 3.12
.venv/bin/python review_artifacts/pr68112/probe.py
```

探针从 Git 历史中的固定提交读取原始方法，从当前工作区读取候选 view，
使用合成整数标记和建模的 manager 循环进行比较；不加载模型、tokenizer、
XGrammar 或 sampler。浅克隆需要先获取上述基础提交。

- [审查记录](REVIEW.md)
- [算法探针](probe.py)
- [运行结果](results.txt)

AI 参与了分析、候选实现和验证记录的编写。此分支没有提交 upstream PR，
不表示已通过完整测试或可以直接合并。

本次提交的 Ruff 检查、格式检查和 Markdown 检查通过。全量 pre-commit
未完成：首次环境安装在 actionlint 阶段中断，按用户要求不等待该检查提交。
