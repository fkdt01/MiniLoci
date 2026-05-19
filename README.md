
交流Q群：33293951

# MiniLoci

> 轻量级会话记忆系统，为 Hermes Agent 提供智能上下文召回

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3.35+-green.svg)](https://sqlite.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## 特性

- **自动保存** — 每轮对话自动入库，无需手动操作
- **按需召回** — 你说"之前""上次""那个"时才搜索，不打扰正常对话
- **语义理解** — 同义词扩展 + 混合搜索（FTS5 + 向量语义）
- **时间加权** — 符合人类记忆规律，越近越重要
- **自动归档** — 部署、配置、架构等关键信息自动永久保存
- **部署不丢** — WAL 模式 + 优雅关闭，重启数据仍在

---

## 快速开始

### 安装位置很重要

MiniLoci 是 Hermes Agent 的 **MemoryProvider 插件**，用户安装时必须放在：

```text
~/.hermes/plugins/miniloci/
```

不要放到 `~/.hermes/plugins/memory/miniloci/`。Hermes 的用户插件扫描路径是 `~/.hermes/plugins/<name>/`，放错目录时可能不会报错，但 MiniLoci 实际不会被加载。

### 1. 克隆插件

```bash
mkdir -p ~/.hermes/plugins
git clone https://github.com/fkdt01/MiniLoci.git ~/.hermes/plugins/miniloci
cd ~/.hermes/plugins/miniloci
```

如果目录已经存在，用更新方式即可：

```bash
cd ~/.hermes/plugins/miniloci
git pull --ff-only origin main
```

### 2. 安装依赖

推荐把依赖安装到 Hermes Agent 正在使用的 Python 环境中。常见本地部署路径如下：

```bash
~/.hermes/hermes-agent/venv/bin/python -m pip install numpy sentence-transformers jieba
```

如果你的 Hermes 使用的是另一个虚拟环境，请把上面的 Python 路径替换成实际运行 Gateway 的 Python。

可选项：

```bash
# 仅当历史数据很多、确实需要更大规模向量检索时再装
~/.hermes/hermes-agent/venv/bin/python -m pip install faiss-cpu
```

MiniLoci 默认使用 SQLite FTS5 + numpy 向量后端；Faiss 只是可选加速，不是必需依赖。

### 3. 启用 MemoryProvider

在 `~/.hermes/config.yaml` 中启用：

```yaml
memory:
  memory_enabled: true
  provider: miniloci
```

只需要设置 `memory.provider: miniloci`。MiniLoci 不需要额外写入 `plugins.enabled`。

### 4. 首次向量模型准备

默认配置为：

```json
{
  "enable_vector": true,
  "vector_model": "BAAI/bge-small-zh-v1.5",
  "vector_backend": "auto",
  "vector_local_files_only": true
}
```

含义：MiniLoci 默认只从本地 HuggingFace 缓存加载向量模型，避免 Hermes Gateway 在运行时被在线 HEAD 请求反复超时阻塞。

如果本机还没有下载过模型，可以临时允许联网下载：

```json
{
  "vector_local_files_only": false
}
```

配置文件位置：

```text
~/.hermes/loci-archive/config.json
```

模型下载完成后，建议把 `vector_local_files_only` 改回 `true`。

如果暂时不需要语义向量召回，也可以关闭向量搜索，只使用 FTS5：

```json
{
  "enable_vector": false
}
```

### 5. 重启或重载 Hermes

本地 systemd 部署通常使用：

```bash
systemctl --user restart hermes-gateway
```

如果你的 Hermes 支持并已经连接到 Gateway，也可以优先用 `/reload-mcp` 或对应的重载方式，避免中断正在进行的会话。MemoryProvider 变更通常建议完整重启一次 Gateway，以确认初始化链路正常。

### 6. 验证安装

```bash
# 确认插件目录正确
ls ~/.hermes/plugins/miniloci/plugin.yaml

# 确认 Hermes 能找到 provider
cd ~/.hermes/hermes-agent
source venv/bin/activate
python -c "from plugins.memory import find_provider_dir; print(find_provider_dir('miniloci'))"

# 预期输出类似：
# /home/<user>/.hermes/plugins/miniloci

# 确认数据库已创建
ls -la ~/.hermes/loci-archive/miniloci.db

# 查看 Gateway 日志中的 MiniLoci 初始化信息
grep -i "miniloci\|loci-archive" ~/.hermes/logs/gateway.log | tail -20
```

### 基础使用

安装完成后无需手动保存。Hermes 每轮正常对话结束时会把 user/assistant turn 写入 MiniLoci。普通对话不会被强行注入历史，只有出现回忆意图时才触发召回。

示例：

> "你还记得上次部署的问题吗？"

MiniLoci 会搜索最近窗口内的相关历史，并把结果注入上下文。

常见触发表达：

- "你还记得……"
- "我们之前讨论过……"
- "上次那个方案……"
- "你说过……"
- "以前配置过……"

### 结构化记忆工具

v1.1+ 之后，MiniLoci 不只保存原始 turns，还会逐步生成可追溯的结构化记忆：

- L0 `turns`：原始对话片段
- L1 `memory_atoms`：偏好、项目事实、故障经验等结构化 atoms
- L2 `scene_blocks`：按场景聚合的记忆块
- L3 `persona_candidate.md`：候选用户画像，只供人工审核，不会自动应用到 Hermes 长期记忆

已暴露的工具包括：

```text
miniloci_search            # 搜索原始历史 turns
miniloci_search_atoms      # 搜索 L1 结构化 atoms
miniloci_search_scenes     # 搜索 L2 场景 blocks
miniloci_persona_candidate # 生成/读取 L3 候选画像，review-only
miniloci_backfill_layers   # 从历史 turns 回填 L1/L2，默认 dry_run=true
```

### 历史数据回填

如果你是从旧版本升级到 v1.6.0，已有的 L0 `turns` 可能还没有对应的 L1/L2 结构化记忆。可以先做 dry run：

```json
{
  "tool": "miniloci_backfill_layers",
  "args": {
    "dry_run": true,
    "limit": 100,
    "since_days": 30
  }
}
```

确认结果合理、并且已经备份数据库后，再执行真实回填：

```json
{
  "tool": "miniloci_backfill_layers",
  "args": {
    "dry_run": false,
    "limit": 100,
    "since_days": 30
  }
}
```

⚠️ 建议真实回填前先备份：

```bash
mkdir -p ~/.hermes/loci-archive/backups
cp ~/.hermes/loci-archive/miniloci.db \
  ~/.hermes/loci-archive/backups/miniloci-before-backfill-$(date +%Y%m%d-%H%M%S).db
```

---

## 搜索原理

```
用户查询: "你还记得部署方案吗？"
         ↓
1. 触发检测 — "还记得"命中触发词
2. jieba分词（不可用时自动使用中文2-4字滑窗） → "部署" "方案"
3. 同义词扩展 → "部署"扩展出 ["上线", "发布", "构建", "Docker", "CI/CD"]
4. FTS5搜索 — SQLite内置 unicode61 + Python 预分词 token soup OR 查询
5. 向量语义（可选）— BAAI/bge-small-zh模型 + numpy backend；Faiss 仅作为可选加速
6. 加权排序 → 返回前5条
```

### 权重公式

```
最终分数 = 关键词匹配×45% + 语义相似×25% + 时间近远×15% + 重要性×15%

时间权重: 今天=1.0, 昨天=0.7, 前天=0.4, 更久=0.1
重要性:   普通=0.33, 重要=0.67, 关键=1.0
```

---

## 架构

```
MiniLociProvider (MemoryProvider)
├── 存储层
│   ├── SQLite + WAL模式
│   ├── FTS5全文索引（unicode61 + Python 预分词）
│   └── numpy向量索引（Faiss可选加速）
├── 召回层
│   ├── 触发检测
│   ├── jieba中文分词
│   ├── 同义词扩展
│   └── 混合排序
└── 归档层
    ├── 9类自动永久保存
    ├── Markdown + Git版本控制
    └── 敏感信息过滤
```

---

## 配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `window_days` | 3 | 短期记忆保留天数（1-30） |
| `fts_weight` | 0.45 | 关键词匹配权重 |
| `vector_weight` | 0.25 | 向量语义权重 |
| `enable_vector` | true | 是否启用向量搜索 |
| `vector_model` | BAAI/bge-small-zh-v1.5 | Embedding模型 |
| `vector_backend` | auto | 向量后端：auto/faiss/numpy；Faiss缺失时自动用numpy |
| `vector_local_files_only` | true | 向量模型默认只从本地 HuggingFace 缓存加载，避免 Gateway 被在线 HEAD 请求超时阻塞；首次下载时可设为 false |
| `default_style` | concise | 回答风格 |
| `auto_cleanup` | true | 自动清理过期数据 |
| `backup_count` | 7 | 保留备份数量 |

配置文件位置：`~/.hermes/loci-archive/config.json`

---

## 永久保存分类

| 类型 | 触发条件 | 示例 |
|------|---------|------|
| system_config | hermes config、config.yaml | 模型切换、provider配置 |
| environment | WSL、Node/Python版本 | 环境变量、PATH设置 |
| project_config | package.json、Dockerfile | 项目配置变更 |
| deployment | 部署、上线、发布 | Docker构建、CI/CD |
| security | SSH配置、密钥 | 认证方式、防火墙 |
| architecture | 技术选型、数据库设计 | 架构决策记录 |
| incident | Bug修复、排查过程 | 故障处理记录 |
| convention | 代码规范、分支策略 | 团队约定 |
| manual | 用户说"记住""记下来" | 手动标记 |

保存位置：`~/.hermes/loci-archive/permanent/`

---

## 同义词表

| 关键词 | 同义词 |
|--------|--------|
| 部署 | 上线、发布、构建、Docker、CI/CD |
| 缓存 | Redis、Cache、内存、加速 |
| 数据库 | DB、MySQL、Postgres、连接池 |
| 性能 | 优化、慢、并发、瓶颈、QPS |
| 前端 | React、Vue、UI、界面、前端框架 |
| 配置 | 设置、参数、env、变量、环境变量 |

---

## 项目结构

```
miniloci/
├── __init__.py          # 主插件代码（~1300行）
├── plugin.yaml          # 插件元数据
├── test_miniloci.py     # 测试套件
└── README.md            # 本文档
```

---

## API 说明

### 触发词

以下表达会触发记忆搜索：

| 触发词 | 示例 | 强度 |
|--------|------|------|
| 还记得 | "你还记得上次那个问题吗" | 强 |
| 之前/上次 | "我们之前讨论过部署" | 中 |
| 我们讨论 | "我们讨论过数据库设计" | 中 |
| 你说过 | "你说过用 Railway 部署" | 中 |
| 那个 | "那个缓存方案怎么样了" | 弱 |
| 以前/早前 | "以前配置过 SSL" | 弱 |

### 手动标记

说以下关键词，当前对话会被永久保存：

- "记住这个"
- "记下来"
- "很重要"
- "别忘"
- "存档"

（"保存"已剔除，避免日常用语误触发）

---

## 测试

```bash
cd ~/.hermes/plugins/miniloci
pytest test_miniloci.py -v
```

### 测试覆盖

- ✅ 初始化与数据库创建
- ✅ 对话保存（user + assistant）
- ✅ 重要性自动检测（8类规则）
- ✅ 永久保存检测（9类自动 + 手动标记）
- ✅ 回忆查询触发检测
- ✅ 敏感信息过滤
- ✅ 时间权重计算
- ✅ 混合搜索召回
- ✅ 向量模型默认本地缓存加载，避免 Gateway 在线 HEAD 超时
- ✅ 配置Schema验证
- ✅ 完整工作流集成

---

## 性能

| 数据量 | 搜索耗时 | 数据库大小 |
|--------|---------|-----------|
| 100条 | 0.28ms | 0.12 MB |
| 500条 | 0.43ms | 0.45 MB |
| 1000条 | 0.58ms | 1.01 MB |
| 2000条 | 1.07ms | 2.23 MB |

测试环境：WSL2 Ubuntu, Python 3.11, SQLite FTS5 unicode61 + Python 预分词；小规模向量用 numpy backend

---

## 依赖

| 依赖 | 必需 | 用途 |
|------|------|------|
| sqlite3 | ✅ | 数据存储 |
| numpy | ✅ | 小规模向量矩阵检索 |
| sentence-transformers | ❌ | Embedding模型（`enable_vector: true` 时需要） |
| jieba | ❌ | 更好的中文分词；缺失时自动滑窗降级 |
| faiss-cpu | ❌ | 大规模向量搜索可选加速 |

---

## 与内置Memory的关系

```
Hermes记忆系统:
┌─────────────────────────────────────────┐
│           系统提示构建层                  │
├─────────────────────────────────────────┤
│  1. SOUL.md          ← 身份/人格        │
│  2. MEMORY.md        ← 内置：环境/项目  │
│  3. USER.md          ← 内置：用户画像    │
│  4. MiniLoci召回      ← 按需：对话历史  │
│  5. 当前对话          ← 实时            │
└─────────────────────────────────────────┘
```

- **内置Memory**：存"知识"，永久保留，全量注入
- **MiniLoci**：存"历史"，3天窗口+永久归档，按需搜索

两者互补不冲突。

---

## 常见问题

**Q: 为什么有时候搜不到内容？**

- 内容超过3天已被清理（普通对话）
- jieba未安装导致分词失败（已修复降级逻辑）
- 查询词没有命中同义词表

**Q: 向量搜索为什么慢？**

首次使用需要模型已在 HuggingFace 本地缓存中。为避免 Gateway 运行时被在线 HEAD 请求反复超时阻塞，默认 `vector_local_files_only: true`；如果需要首次联网下载模型，可临时设为 `false`，下载完成后建议改回 `true`。也可通过 `enable_vector: false` 关闭向量搜索，仅使用 FTS。

**Q: 数据库坏了怎么办？**

```bash
# 检查完整性
sqlite3 ~/.hermes/loci-archive/miniloci.db "PRAGMA integrity_check;"

# 从备份恢复
ls -t ~/.hermes/loci-archive/backups/*.db | head -1
```

---

## 更新日志

### v1.6.0 (2026-05-15)

**新增历史 L1/L2 回填能力：**
- 新增 `backfill_memory_layers(limit, since_days, dry_run)`，可从既有 `turns` 重新抽取 L1 `memory_atoms` 并滚动更新 L2 `scene_blocks`
- 新增 `miniloci_backfill_layers` 工具 schema/handler，默认 `dry_run=true`，避免误写生产库
- 支持 `limit` 限制扫描 turn pairs 数，支持 `since_days` 仅扫描最近 N 天
- 回填保留 `source_turn_ids` 与 `trace_ids`，继续满足 L0→L1→L2 可追溯链路
- 补充 3 个 backfill 回归测试，完整测试扩展到 48 项

### v1.5.0 (2026-05-15)

**新增 L3 persona_candidate 候选画像层：**
- 新增 `generate_persona_candidate()`，基于 L1 atoms 与 L2 scenes 生成 `persona/persona_candidate.md`
- 候选文件包含用户偏好、项目事实、故障经验、相关场景摘要，并保留 `source_turn_ids` / `trace_ids`
- 安全策略：只生成候选草案，`review_required=true`、`applied=false`，不自动覆盖 Hermes 长期 memory / USER
- 新增 `read_persona_candidate()` 与 `miniloci_persona_candidate` 工具 schema/handler
- 补充 2 个 L3 persona candidate 回归测试，完整测试扩展到 45 项

### v1.4.0 (2026-05-15)

**新增 L2 scene_blocks 场景记忆层：**
- 新增 `scene_blocks` 表与 `scene_blocks_fts` 索引，DB schema 升级到 `user_version=7`
- 同一 `scene_name` 下的 L1 atoms 自动聚合为 L2 scene block
- scene summary 使用规则版可追溯摘要，保留 atom 类型与原始 atom 内容
- 每个 scene 保存 `atom_ids`、`source_turn_ids`、`trace_ids`，继续保持从 L2 回溯到 L0 turns
- 新增 `search_scenes(query, limit)` 和 `miniloci_search_scenes` 工具 schema/handler
- 补充 3 个 L2 scene 回归测试，完整测试扩展到 43 项

### v1.3.0 (2026-05-15)

**增强 L1 atoms 去重/冲突检测与工具访问：**
- 新增规则版 `_decide_atom_conflict()`，支持 `store / update / merge / discard` 决策
- 重复但更弱的 instruction 会 discard，不覆盖更完整偏好
- 更具体的 project atom 会 update 旧 atom，同时合并 `source_turn_ids` 与 `trace_ids`
- 中文项目事实冲突检测增加核心实体/动作重合判断，减少同义改写导致的重复 atom
- 新增 `miniloci_search_atoms` 工具 schema 与 handler，可直接搜索结构化 atoms
- 完整测试扩展到 40 项，覆盖 discard/update/tool handler 回归

### v1.2.0 (2026-05-15)

**新增 L1 memory_atoms 结构化记忆层：**
- 新增 `memory_atoms` 表与 `memory_atoms_fts` 索引，DB schema 升级到 `user_version=6`
- `sync_turn()` 现在会用轻量规则提取 instruction / project / incident atoms
- 每条 atom 保存 `source_turn_ids`、`trace_ids`、`source_session_id`，可追溯到底层 turns
- 新增 `search_atoms(query, limit, atom_type)`，用于搜索结构化记忆而不是原始对话片段
- 初版去重：同类型相似 atom 自动合并 source turns，避免重复项目事实无限增长
- 补充 3 个 L1 atoms 回归测试，完整测试扩展到 37 项

### v1.1.0 (2026-05-15)

**P0 记忆架构升级：搜索更稳、结果可追溯、失败可诊断：**
- 混合召回排序改为 RRF（Reciprocal Rank Fusion），降低 FTS/向量分数尺度不一致带来的排序波动
- 新增 `turns.trace_id` 与 v5 DB migration，历史 turns 自动补齐 `turn-{id}` 形式的稳定追溯 ID
- 搜索结果新增 `trace_id`、`source_turn_ids`、`source_session_id`、`search_sources`、`rrf_score`，为后续 L1/L2/L3 分层记忆打基础
- 新增 `health_status()`，暴露 vector/FTS 降级状态、向量数量、队列积压等诊断信息
- 向量召回失败时自动降级保留 FTS/LIKE 结果，并记录 `last_vector_error`
- 测试扩展到 34 项，覆盖 RRF、trace metadata 与 degraded fallback

### v1.0.4 (2026-05-11)

**修复 Gateway 向量模型在线探测阻塞：**
- `_get_vector_model()` 默认向 `SentenceTransformer` 传入 `local_files_only=True`
- 新增 `vector_local_files_only` 配置项；首次联网下载可显式设为 `false`
- 避免 HuggingFace Hub 对已缓存模型的可选文件发在线 HEAD 请求，在网络不稳时反复超时并阻塞召回
- 补充本地缓存加载回归测试

### v1.0.3 (2026-05-11)

**去 native 化修复 MiniLoci 检索链路：**
- FTS 表从 `/tmp/simple` tokenizer 迁移到 SQLite 内置 `unicode61`
- 插入/重建 FTS 时写入 Python 预分词 token soup，支持中文、英文技术词和同义词召回
- 新增 v4 DB migration：自动 drop/recreate `turns_fts` 并从 `turns` 全量重建索引
- LIKE fallback 改用清洗后的关键词/同义词，不再用原始“你还记得...”整句
- Faiss 改为可选后端；未安装时默认使用 numpy 矩阵乘法检索
- 新增 `backfill_vectors()`，可为历史 turns 补齐 embedding 并刷新 numpy 索引
- 修正 SentenceTransformer 缓存策略：默认使用全局 HuggingFace cache，避免空 cache_folder 导致离线加载失败
- 补充 FTS schema/token soup、LIKE fallback、numpy backend、vector backfill 回归测试

### v1.0.2 (2026-05-04)

**稳定中文 FTS5 查询与向量后台任务：**
- 实现方案D查询构造：`jieba` 分词 + 安全关键词清理 + `OR` 查询
- 清理 FTS5 特殊字符与用户输入布尔词，避免 `MATCH` 语法错误
- 在 `_hybrid_search`、LIKE fallback 与 `_tool_search` 中恢复时间窗口过滤
- 工具搜索复用方案D，并返回 `fts_query` 便于调试
- 将向量计算改为单 worker 队列，串行化 SentenceTransformer/Faiss 写入
- 增加向量模型加载锁、编码锁、`torch.set_num_threads(1)` 与 shutdown drain
- 修复非主线程初始化时 `signal.signal()` 导致 provider 初始化中断的问题
- 补充中文 FTS、特殊字符清理、时间过滤、工具搜索与 worker 队列回归测试

### v1.0.1 (2026-05-02)

**修复向量搜索关键 bug：**
- 修复 `IndexFlatIP` 内积相似度计算错误（原 `1.0 - dist` 导致相似度颠倒）
- 修复 `_hybrid_search` 懒加载条件，向量模型首次查询时正确触发加载
- 修复 `_vector_model_loaded` 初始化顺序，避免 AttributeError
- 添加本地模型缓存目录，避免重复下载
- 确保后台线程中模型加载完成后再计算向量，并立即刷盘

### v1.0.0 (2026-05-02)

- 修复FTS5中文搜索（jieba分词 + OR查询 + simple tokenizer）
- 修复jieba缺失时的降级分词逻辑
- 同义词匹配验证通过
- 手动标记保留但剔除"保存"（误触率高）
- 9类自动永久保存 + 1类手动标记

---

## 贡献

欢迎Issue和PR！

## 许可

MIT License

---

*MiniLoci = 微小但顽强的记忆，一颗小小的核心顽强地传递下去* 💜
