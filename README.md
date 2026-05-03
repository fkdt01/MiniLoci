
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

### 安装

```bash
# 克隆到 Hermes 插件目录
git clone https://github.com/yourusername/miniloci.git ~/.hermes/plugins/miniloci

# 安装依赖
pip install jieba faiss-cpu sentence-transformers
```

### 配置

在 `~/.hermes/config.yaml` 中启用：

```yaml
memory:
  memory_enabled: true
  provider: miniloci
```

### 使用

无需额外操作，正常对话即可。当你说：

> "你还记得上次部署的问题吗？"

MiniLoci 会自动搜索相关历史并注入上下文。

---

## 搜索原理

```
用户查询: "你还记得部署方案吗？"
         ↓
1. 触发检测 — "还记得"命中触发词
2. jieba分词 → "部署" "方案"
3. 同义词扩展 → "部署"扩展出 ["上线", "发布", "构建", "Docker", "CI/CD"]
4. FTS5搜索 — SQLite全文索引OR查询
5. 向量语义（可选）— BAAI/bge-small-zh模型找语义相近内容
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
│   ├── FTS5全文索引
│   └── Faiss向量索引（可选）
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

测试环境：WSL2 Ubuntu, Python 3.11, SQLite FTS5 + simple tokenizer

---

## 依赖

| 依赖 | 必需 | 用途 |
|------|------|------|
| sqlite3 | ✅ | 数据存储 |
| jieba | ✅ | 中文分词 |
| faiss-cpu | ❌ | 向量搜索 |
| sentence-transformers | ❌ | Embedding模型 |
| numpy | ❌ | 数值计算 |

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

首次使用需下载模型（约95MB），之后缓存本地。可通过 `enable_vector: false` 关闭。

**Q: 数据库坏了怎么办？**

```bash
# 检查完整性
sqlite3 ~/.hermes/loci-archive/miniloci.db "PRAGMA integrity_check;"

# 从备份恢复
ls -t ~/.hermes/loci-archive/backups/*.db | head -1
```

---

## 更新日志

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
