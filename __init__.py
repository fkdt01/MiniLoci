# MemoryProvider plugin for Hermes Agent
"""
MiniLoci - 轻量级会话记忆系统

核心功能:
- 3天短期记忆窗口，自动清理
- 混合搜索 (FTS5关键词 + 向量语义)
- 重要性自动检测 + 手动标记
- 9类系统自动永久保存
- 部署重启不丢数据 (WAL + 优雅关闭)
- 简洁回答模式 (Concise)
"""

import sqlite3
import time
import json
import signal
import sys
import re
import os
import subprocess
import logging
import threading
import queue
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


class MiniLociProvider:
    """MiniLoci 记忆提供者"""
    
    @property
    def name(self) -> str:
        return "miniloci"
    
    def is_available(self) -> bool:
        return True
    
    def initialize(self, session_id: str, **kwargs):
        """初始化 MiniLoci"""
        self.hermes_home = kwargs.get("hermes_home", os.path.expanduser("~/.hermes"))
        self.db_dir = Path(self.hermes_home) / "loci-archive"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir / "miniloci.db"
        self.permanent_dir = self.db_dir / "permanent"
        self.permanent_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置
        self._config = self._load_config()
        self.window_days = self._config.get("window_days", 3)
        self.fts_weight = self._config.get("fts_weight", 0.45)
        self.vector_weight = self._config.get("vector_weight", 0.25)
        self.enable_vector = self._config.get("enable_vector", True)
        self.vector_model_name = self._config.get("vector_model", "BAAI/bge-small-zh-v1.5")
        self.default_style = self._config.get("default_style", "concise")
        self.auto_cleanup = self._config.get("auto_cleanup", True)
        self.backup_count = self._config.get("backup_count", 7)
        
        # 先初始化所有运行期状态，保证后续任一步失败时 shutdown/sync_turn 都不会访问缺失属性
        self.session_id = session_id
        self._shutdown_requested = threading.Event()
        self._db_lock = threading.RLock()
        self._vector_lock = threading.RLock()
        self._vector_model_lock = threading.Lock()
        self._vector_model = None
        self._vector_model_loaded = False  # 先设置标记，再初始化Faiss
        self._faiss_index = None
        self._vector_map = {}  # faiss_id -> turn_id
        self._next_faiss_id = 0
        self._vector_dimension = 512  # bge-small-zh-v1.5 / all-MiniLM-L6-v2 默认维度
        self._vector_backend = self._config.get("vector_backend", "auto")
        self._vector_ids = []          # numpy backend: row index -> turn_id
        self._vector_matrix = None     # numpy backend: shape=(N, 512), float32
        self._pending_vectors = []
        self._flush_threshold = 100
        self._last_flush_time = time.time()
        self._flush_interval = 1800  # 30分钟
        self._vector_queue = queue.Queue()
        self._vector_worker = None
        self._last_vector_error = None
        self._last_vector_error_time = None
        self._last_fts_error = None
        self._last_fts_error_time = None
        
        # 重要性检测规则
        self.importance_rules = {
            "decision": ["决定", "选择", "采用", "用", "对比", "方案", "选型"],
            "task": ["TODO", "任务", "待办", "下一步", "要做", "完成", "计划"],
            "lesson": ["错误", "问题", "坑", "注意", "别忘了", "教训", "踩坑"],
            "config": ["设置", "配置", "参数", "env", "变量", "修改", "调整"],
            "architecture": ["结构", "设计", "方案", "架构", "模块", "分层", "拆分"],
            "deploy": ["部署", "上线", "发布", "CI/CD", "Docker", "构建", "发布"],
            "security": ["密钥", "密码", "token", "secret", "权限", "认证", "鉴权"],
            "performance": ["优化", "慢", "缓存", "并发", "瓶颈", "性能", "调优"]
        }
        
        # 同义词表
        self.synonyms = {
            "部署": ["上线", "发布", "构建", "Docker", "CI/CD", "deploy"],
            "缓存": ["Redis", "Cache", "内存", "加速", "cache", "缓存配置", "Redis配置"],
            "数据库": ["DB", "MySQL", "Postgres", "SQLite", "SQL", "数据", "连接池"],
            "认证": ["登录", "Auth", "Token", "JWT", "OAuth", "权限", "鉴权"],
            "架构": ["结构", "设计", "分层", "模块", "微服务", "架构"],
            "性能": ["优化", "慢", "并发", "瓶颈", "加速", "调优", "QPS", "吞吐量"],
            "错误": ["Bug", "异常", "崩溃", "失败", "问题", "报错", "故障"],
            "配置": ["设置", "参数", "env", "变量", "config", "配置", "环境变量"],
            "安全": ["密钥", "密码", "secret", "加密", "漏洞", "防护"],
            "前端": ["React", "Vue", "界面", "UI", "页面", "客户端", "前端框架"],
            "后端": ["API", "Server", "服务", "接口", "服务端"],
            "记住": ["记录", "保存", "存档", "配置", "设置"],
            "框架": ["React", "Vue", "Angular", "前端", "框架选型"],
        }
        
        # 手动标记关键词（"保存"已剔除，避免日常用语误触发）
        self.manual_markers = ["记住", "记下来", "很重要", "别忘", "存档"]
        
        # 回忆查询触发词
        self.recall_patterns = ["还记得", "之前", "上次", "我们讨论", "你说过", "那个", "以前", "早前"]
        
        # 数据库
        self._db = None
        self._init_db()
        
        # 信号处理（gateway 常在非主线程初始化 provider，不能让 signal 注册失败中断初始化）
        self._register_shutdown_hooks()
        
        # 会话恢复
        self._recover_orphaned_sessions()
        
        # 自动清理
        if self.auto_cleanup:
            self._run_daily_cleanup()
        
        # 初始化向量后端与单 worker 队列 (如果启用)。Faiss 只是可选加速，缺失时回退到 numpy。
        if self.enable_vector:
            try:
                self._init_vector_backend()
                self._load_vectors_from_db()
                self._start_vector_worker()
                vector_count = self._vector_count()
                logger.info(f"Vector search ready: backend={self._vector_backend}, "
                           f"vectors_loaded={vector_count}, "
                           f"model_loaded={self._vector_model_loaded}")
            except Exception as e:
                logger.warning(f"Vector search initialization failed: {e}")
                self.enable_vector = False
        
        logger.info(f"MiniLoci initialized for session {session_id}")
    
    def _load_simple_extension(self):
        """兼容旧配置的可选 tokenizer 扩展加载。

        MiniLoci 不再依赖 /tmp/simple。FTS 默认使用 SQLite 内置 unicode61 +
        Python 预分词 token soup；只有用户显式配置 simple_extension_path 时才尝试加载。
        """
        ext_path = self._config.get("simple_extension_path") if hasattr(self, "_config") else None
        if not ext_path:
            return False
        try:
            self._db.enable_load_extension(True)
            self._db.load_extension(str(ext_path))
            logger.info(f"Optional tokenizer extension loaded: {ext_path}")
            return True
        except Exception as e:
            logger.warning(f"Optional tokenizer extension unavailable: {e}")
            return False
    
    # ==================== 数据库操作 ====================
    
    def _init_db(self):
        """初始化数据库和WAL模式"""
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA temp_store=MEMORY")
        
        # 可选加载 tokenizer 扩展（默认不需要；FTS 使用 unicode61 + Python token soup）
        self._load_simple_extension()
        
        # 创建表
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                start_time REAL NOT NULL,
                end_time REAL,
                platform TEXT,
                status TEXT DEFAULT 'active',
                turn_count INTEGER DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                importance INTEGER DEFAULT 1,
                tags TEXT,
                metadata TEXT,
                trace_id TEXT,
                vector BLOB,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            
            CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                content,
                tags,
                tokenize='unicode61',
                content='turns',
                content_rowid='id'
            );
            
            CREATE TABLE IF NOT EXISTS memory_atoms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                priority INTEGER DEFAULT 50,
                source_turn_ids TEXT NOT NULL,
                trace_ids TEXT NOT NULL,
                source_session_id TEXT,
                scene_name TEXT,
                metadata TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memory_atoms_fts USING fts5(
                content,
                type,
                scene_name,
                tokenize='unicode61',
                content='memory_atoms',
                content_rowid='id'
            );
            
            CREATE INDEX IF NOT EXISTS idx_turns_time ON turns(timestamp);
            CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
            CREATE INDEX IF NOT EXISTS idx_turns_importance ON turns(importance);
            CREATE INDEX IF NOT EXISTS idx_atoms_type ON memory_atoms(type);
            CREATE INDEX IF NOT EXISTS idx_atoms_session ON memory_atoms(source_session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
        """)
        
        # 数据库迁移
        self._migrate_db()
        self._db.commit()
        logger.info(f"Database initialized: {self.db_path}")
    
    def _migrate_db(self):
        """数据库版本迁移"""
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        
        if version < 2:
            # v1 -> v2: 添加vector列
            try:
                self._db.execute("ALTER TABLE turns ADD COLUMN vector BLOB")
            except sqlite3.OperationalError:
                pass  # 列已存在
            version = 2
        
        if version < 3:
            # v2 -> v3: 添加stats表
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    operation TEXT,
                    duration_ms REAL,
                    success INTEGER
                )
            """)
            version = 3

        # v3 -> v4: 去掉 /tmp/simple tokenizer 强依赖，重建为 unicode61 + Python 预分词索引。
        # 即使 user_version 已经是 4，只要检测到 legacy schema 也会重建，保证生产 DB 自修复。
        if version < 4 or self._fts_needs_rebuild():
            self._rebuild_fts_index()
            version = max(version, 4)

        if version < 5:
            # v4 -> v5: 为每条 turn 增加稳定 trace_id，供 L1/L2/L3 摘要追溯到底层原文。
            try:
                self._db.execute("ALTER TABLE turns ADD COLUMN trace_id TEXT")
            except sqlite3.OperationalError:
                pass
            self._db.execute("UPDATE turns SET trace_id = 'turn-' || id WHERE trace_id IS NULL OR trace_id = ''")
            version = 5

        if version < 6:
            # v5 -> v6: 创建轻量 L1 memory_atoms 结构化记忆层。
            self._create_memory_atoms_tables()
            version = 6
        
        self._db.execute(f"PRAGMA user_version = {version}")

    def _create_fts_table(self):
        """创建内置 tokenizer 的 FTS5 表。"""
        self._db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                content,
                tags,
                tokenize='unicode61',
                content='turns',
                content_rowid='id'
            )
        """)

    def _create_memory_atoms_tables(self):
        """创建轻量 L1 结构化记忆表与 FTS 索引。"""
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS memory_atoms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                priority INTEGER DEFAULT 50,
                source_turn_ids TEXT NOT NULL,
                trace_ids TEXT NOT NULL,
                source_session_id TEXT,
                scene_name TEXT,
                metadata TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_atoms_fts USING fts5(
                content,
                type,
                scene_name,
                tokenize='unicode61',
                content='memory_atoms',
                content_rowid='id'
            );
            CREATE INDEX IF NOT EXISTS idx_atoms_type ON memory_atoms(type);
            CREATE INDEX IF NOT EXISTS idx_atoms_session ON memory_atoms(source_session_id);
        """)

    def _fts_needs_rebuild(self) -> bool:
        """检测 FTS schema 是否仍依赖 legacy simple tokenizer 或缺失。"""
        try:
            row = self._db.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'turns_fts'"
            ).fetchone()
            if not row or not row[0]:
                return True
            sql = row[0].lower()
            return "tokenize='simple'" in sql or 'tokenize="simple"' in sql or "unicode61" not in sql
        except Exception:
            return True

    def _rebuild_fts_index(self):
        """安全重建 FTS 索引：drop legacy table，创建 unicode61 表，并写入 token soup。"""
        logger.info("Rebuilding MiniLoci FTS index with unicode61/token-soup")
        try:
            self._db.execute("DROP TABLE IF EXISTS turns_fts")
        except Exception as e:
            logger.warning(f"Dropping legacy FTS table failed, retrying after disabling extensions: {e}")
            self._db.execute("DROP TABLE IF EXISTS turns_fts")

        self._create_fts_table()
        rows = self._db.execute("SELECT id, content, tags FROM turns ORDER BY id").fetchall()
        if rows:
            self._db.executemany(
                "INSERT INTO turns_fts (rowid, content, tags) VALUES (?, ?, ?)",
                [
                    (
                        row_id,
                        self._tokenize_for_fts(content or ""),
                        self._tokenize_for_fts(tags or ""),
                    )
                    for row_id, content, tags in rows
                ]
            )
        logger.info(f"Rebuilt MiniLoci FTS index rows={len(rows)}")
    
    def _register_shutdown_hooks(self):
        """注册优雅关闭信号"""
        def _graceful_shutdown(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        try:
            signal.signal(signal.SIGTERM, _graceful_shutdown)
            signal.signal(signal.SIGINT, _graceful_shutdown)
        except ValueError:
            logger.debug("Signal hooks skipped: MiniLoci initialized outside main thread")
    
    def _start_vector_worker(self):
        """启动单 worker 向量队列，串行化 SentenceTransformer/Faiss/SQLite 写入。"""
        if self._vector_worker and self._vector_worker.is_alive():
            return
        self._shutdown_requested.clear()
        self._vector_worker = threading.Thread(
            target=self._vector_worker_loop,
            name="miniloci-vector-worker",
            daemon=True,
        )
        self._vector_worker.start()
    
    def _vector_worker_loop(self):
        """串行处理向量任务，避免多线程同时调用 torch/sentence-transformers。"""
        while not self._shutdown_requested.is_set() or not self._vector_queue.empty():
            try:
                item = self._vector_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if item is None:
                    return
                user_rowid, user_content, asst_rowid, assistant_content = item
                model = self._get_vector_model()
                if model is None:
                    logger.debug("Vector model not available, skipping vector computation")
                    continue
                self._add_vector_async(user_rowid, user_content)
                self._add_vector_async(asst_rowid, assistant_content)
                self._flush_vectors()
            except Exception as e:
                logger.debug(f"Vector computation failed (non-fatal): {e}")
            finally:
                self._vector_queue.task_done()
    
    def _enqueue_vector_task(self, user_rowid: int, user_content: str, asst_rowid: int, assistant_content: str):
        """提交向量任务到单 worker 队列。"""
        if not self.enable_vector or self._shutdown_requested.is_set():
            return
        if self._vector_worker is None or not self._vector_worker.is_alive():
            self._start_vector_worker()
        self._vector_queue.put((user_rowid, user_content, asst_rowid, assistant_content))
    
    def _drain_vector_tasks(self, timeout: float = 30.0):
        """等待已提交向量任务完成，shutdown 时避免 daemon 线程仍在跑 torch。"""
        if not getattr(self, "_vector_worker", None):
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._vector_queue.unfinished_tasks == 0:
                return
            time.sleep(0.05)
        logger.warning("Timed out waiting for vector tasks to finish")
    
    def _stop_vector_worker(self):
        """停止向量 worker。"""
        self._shutdown_requested.set()
        worker = getattr(self, "_vector_worker", None)
        if not worker:
            return
        self._vector_queue.put(None)
        worker.join(timeout=5)
        if worker.is_alive():
            logger.warning("Vector worker did not stop within timeout")
    
    def _flush_and_close(self):
        """刷盘并关闭数据库"""
        try:
            if self._db:
                self._db.commit()
                if hasattr(self, 'session_id') and self.session_id:
                    self._db.execute(
                        "UPDATE sessions SET end_time = ?, status = 'closed' WHERE id = ?",
                        (time.time(), self.session_id)
                    )
                    self._db.commit()
                self._db.close()
                self._db = None
                logger.info("Database flushed and closed")
        except Exception as e:
            logger.error(f"Error during flush and close: {e}")
    
    def shutdown(self):
        """关闭时刷盘"""
        try:
            if self.enable_vector:
                self._drain_vector_tasks()
                self._flush_vectors()
                self._stop_vector_worker()
            self._flush_and_close()
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
    
    def _recover_orphaned_sessions(self):
        """恢复异常关闭的会话"""
        try:
            five_min_ago = time.time() - 300
            orphans = self._db.execute("""
                SELECT s.id, MAX(t.timestamp) as last_time
                FROM sessions s
                LEFT JOIN turns t ON s.id = t.session_id
                WHERE s.status = 'active'
                GROUP BY s.id
                HAVING last_time < ? OR last_time IS NULL
            """, (five_min_ago,)).fetchall()
            
            for session_id, _ in orphans:
                self._db.execute(
                    "UPDATE sessions SET status = 'abnormal' WHERE id = ?",
                    (session_id,)
                )
                self._db.execute(
                    """INSERT INTO turns (session_id, timestamp, role, content, metadata)
                    VALUES (?, ?, 'system', '', ?)""",
                    (session_id, time.time(), json.dumps({"recovered": True, "reason": "abnormal_shutdown"}))
                )
            
            if orphans:
                self._db.commit()
                logger.info(f"Recovered {len(orphans)} orphaned sessions")
        except Exception as e:
            logger.warning(f"Session recovery failed: {e}")
    
    def _run_daily_cleanup(self):
        """清理过期数据"""
        try:
            now = time.time()
            three_days_ago = now - self.window_days * 24 * 3600
            seven_days_ago = now - 7 * 24 * 3600
            thirty_days_ago = now - 30 * 24 * 3600
            
            # 删除普通记录
            cursor = self._db.execute("""
                DELETE FROM turns 
                WHERE importance = 1 AND timestamp < ?
            """, (three_days_ago,))
            deleted_normal = cursor.rowcount
            
            # 删除重要记录
            cursor = self._db.execute("""
                DELETE FROM turns 
                WHERE importance = 2 AND timestamp < ?
            """, (seven_days_ago,))
            deleted_important = cursor.rowcount
            
            # 删除关键记录(非永久)
            cursor = self._db.execute("""
                DELETE FROM turns 
                WHERE importance = 3 AND timestamp < ? 
                AND (metadata IS NULL OR json_extract(metadata, '$.permanent') IS NULL)
            """, (thirty_days_ago,))
            deleted_critical = cursor.rowcount
            
            # 清理孤儿会话
            self._db.execute("""
                DELETE FROM sessions 
                WHERE status != 'active' 
                AND id NOT IN (SELECT DISTINCT session_id FROM turns)
            """)
            
            self._db.commit()
            total = deleted_normal + deleted_important + deleted_critical
            if total > 0:
                logger.info(f"Cleanup: removed {total} expired turns")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    # ==================== 向量搜索 ====================
    
    def _init_vector_backend(self):
        """初始化向量后端。Faiss 可选；缺失时使用 numpy 精确检索。"""
        requested = str(self._config.get("vector_backend", self._vector_backend or "auto")).lower()
        if requested not in {"auto", "faiss", "numpy"}:
            requested = "auto"

        self._faiss_index = None
        self._vector_backend = "numpy"

        if requested in {"auto", "faiss"}:
            try:
                self._init_faiss()
                self._vector_backend = "faiss"
            except Exception as e:
                if requested == "faiss":
                    logger.warning(f"Faiss requested but unavailable; falling back to numpy: {e}")
                else:
                    logger.info(f"Faiss unavailable; using numpy vector backend: {e}")

        if self._vector_backend == "numpy":
            logger.info("Numpy vector backend initialized")

    def _init_faiss(self):
        """初始化Faiss索引"""
        import faiss
        
        self._faiss_index = faiss.IndexFlatIP(self._vector_dimension)  # 使用内积索引，更简单稳定
        logger.info("Faiss index initialized (IndexFlatIP)")
    
    def _get_vector_model(self):
        """懒加载向量模型 - 首次调用时才加载"""
        with self._vector_model_lock:
            if self._vector_model is not None:
                return self._vector_model
            
            if self._vector_model_loaded:
                return None  # 已尝试加载但失败了
            
            try:
                from sentence_transformers import SentenceTransformer
                
                # 限制 torch CPU 并行度，降低 WSL/native 扩展在线程场景下的崩溃概率
                try:
                    import torch
                    torch.set_num_threads(1)
                except Exception:
                    pass
                
                # 设置HF镜像（如果环境变量未设置）
                if "HF_ENDPOINT" not in os.environ:
                    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                
                # 缓存策略：默认使用 HuggingFace 全局缓存；只有显式配置且目录已有内容时才传 cache_folder。
                # 这样不会因为 ~/.hermes/loci-archive/models 是空目录而离线加载失败。
                model_kwargs = {}
                configured_cache = self._config.get("vector_cache_dir")
                if configured_cache:
                    cache_dir = Path(os.path.expanduser(str(configured_cache)))
                    if cache_dir.exists() and any(cache_dir.iterdir()):
                        model_kwargs["cache_folder"] = str(cache_dir)

                # Gateway 运行时必须默认只读本地缓存：
                # SentenceTransformer/HF Hub 即使模型已缓存，也会对若干可选文件发 HEAD 请求；
                # 网络不稳时这些请求会反复超时并阻塞向量召回。需要联网首次下载时，显式配置
                # vector_local_files_only=false。
                local_files_only = bool(self._config.get("vector_local_files_only", True))
                model_kwargs["local_files_only"] = local_files_only
                
                logger.info(f"Loading vector model: {self.vector_model_name} (this may take a moment)...")
                self._vector_model = SentenceTransformer(
                    self.vector_model_name,
                    **model_kwargs
                )
                self._vector_model_loaded = True
                logger.info(f"Vector model loaded: {self.vector_model_name}")
                return self._vector_model
                
            except Exception as e:
                logger.warning(f"Failed to load vector model: {e}")
                self._vector_model = None
                self._vector_model_loaded = True
                return None
    
    def _embed(self, text: str) -> list:
        """文本转向量"""
        model = self._get_vector_model()
        if model is None:
            return []
        with self._vector_lock:
            return model.encode(text, normalize_embeddings=True).tolist()
    
    def _load_vectors_from_db(self):
        """启动时从 SQLite 加载向量到 numpy 矩阵，并在 Faiss 可用时同步加入 Faiss。"""
        cutoff = time.time() - self.window_days * 24 * 3600
        rows = self._db.execute("""
            SELECT id, vector FROM turns
            WHERE timestamp > ? AND vector IS NOT NULL
            ORDER BY id
        """, (cutoff,)).fetchall()

        self._vector_ids = []
        self._vector_matrix = None
        self._vector_map = {}
        self._next_faiss_id = 0

        vectors = []
        ids = []
        for turn_id, vec_blob in rows:
            try:
                vec = np.frombuffer(vec_blob, dtype=np.float32)
                if len(vec) == self._vector_dimension:
                    vectors.append(vec)
                    ids.append(turn_id)
            except Exception:
                continue

        if not vectors:
            return

        vectors_array = np.vstack(vectors).astype(np.float32)
        with self._vector_lock:
            self._vector_ids = list(ids)
            self._vector_matrix = vectors_array
            if self._faiss_index is not None:
                self._faiss_index.add(vectors_array)
                for turn_id in ids:
                    self._vector_map[self._next_faiss_id] = turn_id
                    self._next_faiss_id += 1
        logger.info(f"Loaded {len(vectors)} vectors into {self._vector_backend} backend")

    def _vector_count(self) -> int:
        """当前已加载向量数量。"""
        if self._faiss_index is not None:
            try:
                return int(self._faiss_index.ntotal)
            except Exception:
                pass
        return len(self._vector_ids)

    def _has_vector_index(self) -> bool:
        return self._vector_count() > 0

    def _append_vector_to_index(self, turn_id: int, vec_array: np.ndarray) -> bool:
        """把单条向量加入当前内存索引（numpy 始终可用，Faiss 可选）。"""
        vec_array = np.asarray(vec_array, dtype=np.float32).reshape(-1)
        if len(vec_array) != self._vector_dimension:
            logger.debug(f"Skip vector with unexpected dimension: {len(vec_array)}")
            return False

        with self._vector_lock:
            if turn_id not in self._vector_ids:
                if self._vector_matrix is None:
                    self._vector_matrix = vec_array.reshape(1, -1)
                else:
                    self._vector_matrix = np.vstack([self._vector_matrix, vec_array.reshape(1, -1)])
                self._vector_ids.append(turn_id)

            if self._faiss_index is not None and turn_id not in self._vector_map.values():
                self._faiss_index.add(vec_array.reshape(1, -1))
                self._vector_map[self._next_faiss_id] = turn_id
                self._next_faiss_id += 1
        return True

    def _add_vector_async(self, turn_id: int, text: str):
        """异步添加向量；Faiss 缺失时仍使用 numpy backend。"""
        if not self.enable_vector:
            return
        
        try:
            vec = self._embed(text)
            if not vec:
                return
            vec_array = np.array(vec, dtype=np.float32)
            if self._append_vector_to_index(turn_id, vec_array):
                self._pending_vectors.append((turn_id, vec_array.tobytes()))
                self._check_flush()
        except Exception as e:
            logger.debug(f"Vector add failed (non-fatal): {e}")
    
    def _check_flush(self):
        """检查是否需要持久化"""
        now = time.time()
        if len(self._pending_vectors) >= self._flush_threshold:
            self._flush_vectors()
        elif now - self._last_flush_time >= self._flush_interval:
            self._flush_vectors()
    
    def _flush_vectors(self):
        """批量写回向量到SQLite"""
        if not self._pending_vectors:
            return
        
        try:
            with self._db_lock:
                pending = list(self._pending_vectors)
                self._db.executemany(
                    "UPDATE turns SET vector = ? WHERE id = ?",
                    [(vec_blob, turn_id) for turn_id, vec_blob in pending]
                )
                self._db.commit()
                count = len(pending)
                self._pending_vectors = []
                self._last_flush_time = time.time()
            logger.debug(f"Flushed {count} vectors to database")
        except Exception as e:
            logger.warning(f"Vector flush failed: {e}")

    def backfill_vectors(self, *, limit: Optional[int] = None, since_days: Optional[int] = None, batch_size: int = 32) -> Dict[str, Any]:
        """为已有 turns 补齐缺失向量，并刷新内存索引。"""
        if not self.enable_vector:
            return {"updated": 0, "scanned": 0, "error": "vector disabled"}

        model = self._get_vector_model()
        if model is None:
            return {"updated": 0, "scanned": 0, "error": "vector model unavailable"}

        params = []
        where = ["vector IS NULL"]
        if since_days is not None:
            where.append("timestamp > ?")
            params.append(time.time() - since_days * 24 * 3600)
        sql = f"SELECT id, content FROM turns WHERE {' AND '.join(where)} ORDER BY id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows = self._db.execute(sql, tuple(params)).fetchall()

        updated = 0
        for start in range(0, len(rows), max(1, batch_size)):
            batch = rows[start:start + max(1, batch_size)]
            texts = [content for _, content in batch]
            try:
                with self._vector_lock:
                    embeddings = model.encode(texts, normalize_embeddings=True)
                vectors = np.asarray(embeddings, dtype=np.float32)
                if vectors.ndim == 1:
                    vectors = vectors.reshape(1, -1)
                pending = []
                for (turn_id, _), vec in zip(batch, vectors):
                    if len(vec) != self._vector_dimension:
                        continue
                    blob = np.asarray(vec, dtype=np.float32).tobytes()
                    pending.append((blob, turn_id))
                    self._append_vector_to_index(turn_id, vec)
                if pending:
                    with self._db_lock:
                        self._db.executemany("UPDATE turns SET vector = ? WHERE id = ?", pending)
                        self._db.commit()
                    updated += len(pending)
            except Exception as e:
                logger.warning(f"Vector backfill batch failed: {e}")

        return {"updated": updated, "scanned": len(rows), "backend": self._vector_backend}
    
    def _mark_degraded(self, component: str, error: Exception):
        """记录非致命降级状态，避免召回路径静默失败。"""
        message = str(error)
        now = time.time()
        if component == "vector":
            self._last_vector_error = message
            self._last_vector_error_time = now
        elif component == "fts":
            self._last_fts_error = message
            self._last_fts_error_time = now

    def health_status(self) -> Dict[str, Any]:
        """返回 MiniLoci 当前健康状态，供诊断/系统提示使用。"""
        return {
            "provider": self.name,
            "db_path": str(getattr(self, "db_path", "")),
            "window_days": getattr(self, "window_days", None),
            "enable_vector": getattr(self, "enable_vector", False),
            "vector_backend": getattr(self, "_vector_backend", None),
            "vector_count": self._vector_count() if hasattr(self, "_vector_ids") else 0,
            "vector_queue_size": self._vector_queue.qsize() if hasattr(self, "_vector_queue") else 0,
            "last_vector_error": getattr(self, "_last_vector_error", None),
            "last_fts_error": getattr(self, "_last_fts_error", None),
            "degraded": bool(getattr(self, "_last_vector_error", None) or getattr(self, "_last_fts_error", None)),
        }

    def _attach_trace_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """给搜索结果补齐可追溯元数据。"""
        turn_id = data.get('id')
        if turn_id is not None:
            data.setdefault('trace_id', f"turn-{turn_id}")
            data.setdefault('source_turn_ids', [turn_id])
        if 'source_session_id' not in data and turn_id is not None:
            row = self._db.execute("SELECT session_id, trace_id FROM turns WHERE id = ?", (turn_id,)).fetchone()
            if row:
                data['source_session_id'] = row[0]
                data['trace_id'] = row[1] or data.get('trace_id') or f"turn-{turn_id}"
        elif 'session_id' in data:
            data.setdefault('source_session_id', data.get('session_id'))
        return data

    def _rrf_merge_ranked_results(self, fts_results: List[Dict], vec_results: List[Dict], limit: int = 5, *, k: int = 60) -> List[Dict]:
        """用 Reciprocal Rank Fusion 合并 FTS/LIKE 与向量 ranked list。

        RRF 只依赖各检索器内部排名，不依赖原始分数尺度；同一条记录同时被
        FTS 与向量命中时会累加得分，从而自然优先。
        """
        merged: Dict[Any, Dict[str, Any]] = {}

        def add_ranked(items: List[Dict], source: str):
            for rank, item in enumerate(items):
                item_id = item.get('id')
                if item_id is None:
                    continue
                score = 1.0 / (k + rank + 1)
                if item_id not in merged:
                    merged[item_id] = {
                        'rrf': 0.0,
                        'sources': [],
                        'data': item.copy()
                    }
                merged[item_id]['rrf'] += score
                if source not in merged[item_id]['sources']:
                    merged[item_id]['sources'].append(source)
                # 优先保留含完整内容的 FTS/LIKE 数据；vector 结果通常只有 id/score。
                if source == 'fts' or 'content' not in merged[item_id]['data']:
                    merged[item_id]['data'].update(item)

        add_ranked(fts_results, 'fts')
        add_ranked(vec_results, 'vector')

        ranked = sorted(merged.values(), key=lambda x: x['rrf'], reverse=True)
        return ranked[:limit]

    def _vector_search(self, query_vec: list, limit: int = 10) -> List[Dict]:
        """向量搜索：优先 Faiss；无 Faiss 时使用 numpy 矩阵乘法。"""
        if not query_vec:
            return []
        
        try:
            with self._vector_lock:
                query_array = np.array(query_vec, dtype=np.float32).reshape(-1)
                if len(query_array) != self._vector_dimension:
                    return []

                if self._faiss_index is not None and self._faiss_index.ntotal > 0:
                    distances, indices = self._faiss_index.search(query_array.reshape(1, -1), limit)
                    vector_map_snapshot = dict(self._vector_map)
                    pairs = [
                        (vector_map_snapshot.get(int(idx)), max(0.0, float(dist)))
                        for dist, idx in zip(distances[0], indices[0])
                        if idx >= 0
                    ]
                elif self._vector_matrix is not None and len(self._vector_ids) > 0:
                    matrix = np.asarray(self._vector_matrix, dtype=np.float32)
                    scores = matrix @ query_array
                    top_indices = np.argsort(-scores)[:limit]
                    ids_snapshot = list(self._vector_ids)
                    pairs = [
                        (ids_snapshot[int(idx)], max(0.0, float(scores[int(idx)])))
                        for idx in top_indices
                    ]
                else:
                    return []

            results = []
            for turn_id, similarity in pairs:
                if turn_id:
                    results.append({
                        'id': turn_id,
                        'score': similarity,
                        'source': 'vector'
                    })
            return results
        except Exception as e:
            logger.debug(f"Vector search failed: {e}")
            return []
    # ==================== 重要性检测 ====================
    
    def _detect_importance(self, user_msg: str, assistant_msg: str) -> Tuple[int, List[str]]:
        """检测对话重要性"""
        combined = user_msg + " " + assistant_msg
        matched_rules = []
        
        for rule_name, keywords in self.importance_rules.items():
            if any(kw in combined for kw in keywords):
                matched_rules.append(rule_name)
        
        # 计算重要性级别
        if len(matched_rules) >= 3:
            importance = 3  # 关键
        elif len(matched_rules) >= 1:
            importance = 2  # 重要
        else:
            importance = 1  # 普通
        
        return importance, matched_rules
    
    def _detect_permanent_save(self, user_msg: str, assistant_msg: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """检测是否需要永久保存 - 仅自动检测，剔除手动标记"""
        combined = user_msg + " " + assistant_msg
        
        # 1. 用户手动标记
        if any(m in user_msg for m in self.manual_markers):
            return True, "manual", "user_marked"
        
        # 2. 系统配置变更
        system_config_patterns = [
            "hermes config", "config.yaml", "工具启用", "模型切换",
            "provider", "api_key", "base_url", "memory.provider"
        ]
        if any(p in combined for p in system_config_patterns):
            return True, "system_config", "config_change"
        
        # 2. 环境信息
        env_patterns = [
            "WSL", "Node版本", "Python版本", "npm版本", "pip安装",
            "系统路径", "环境变量", "PATH设置"
        ]
        if any(p in combined for p in env_patterns):
            return True, "environment", "env_info"
        
        # 3. 项目配置
        project_config_patterns = [
            "package.json", "tsconfig", "vite.config", "Dockerfile",
            "docker-compose", "nginx.conf", ".env文件"
        ]
        if any(p in combined for p in project_config_patterns):
            return True, "project_config", "project_setup"
        
        # 4. 部署记录
        deploy_patterns = [
            "部署", "上线", "发布", "CI/CD", "Docker构建",
            "服务器配置", "环境变量设置", "域名配置"
        ]
        if any(p in combined for p in deploy_patterns):
            return True, "deployment", "deploy_record"
        
        # 5. 安全凭证
        security_patterns = [
            "SSH配置", "密钥位置", "认证方式", "权限设置",
            "防火墙", "HTTPS配置", "SSL证书"
        ]
        if any(p in combined for p in security_patterns):
            return True, "security", "security_config"
        
        # 6. 架构决策
        architecture_patterns = [
            "技术选型", "数据库设计", "API设计", "微服务",
            "架构图", "模块划分", "服务拆分"
        ]
        if any(p in combined for p in architecture_patterns):
            return True, "architecture", "arch_decision"
        
        # 7. 故障记录
        incident_patterns = [
            "Bug修复", "错误解决", "排查过程", "故障处理",
            "crash", "报错解决", "性能问题"
        ]
        if any(p in combined for p in incident_patterns):
            return True, "incident", "troubleshooting"
        
        # 8. 重要约定
        convention_patterns = [
            "代码规范", "命名约定", "提交规范", "分支策略",
            "Code Review", "PR规范", "文档规范"
        ]
        if any(p in combined for p in convention_patterns):
            return True, "convention", "team_agreement"
        
        return False, None, None
    
    # ==================== 永久保存 ====================
    
    def _save_permanent(self, session_id: str, user_msg: str, assistant_msg: str,
                       ptype: str, trigger: str, timestamp: float):
        """保存到永久存储"""
        try:
            type_dir = self.permanent_dir / ptype
            type_dir.mkdir(exist_ok=True)
            
            existing_file = self._find_existing_record(type_dir, user_msg + assistant_msg)
            
            if existing_file:
                return self._update_permanent(existing_file, user_msg, assistant_msg, timestamp)
            else:
                return self._create_permanent(type_dir, ptype, trigger, session_id,
                                              user_msg, assistant_msg, timestamp)
        except Exception as e:
            logger.warning(f"Permanent save failed: {e}")
    
    def _find_existing_record(self, type_dir: Path, content: str) -> Optional[Path]:
        """查找相同主题的记录"""
        keywords = self._extract_keywords(content, top_n=5)
        
        for md_file in type_dir.glob("*.md"):
            try:
                file_content = md_file.read_text(encoding="utf-8")
                file_keywords = self._extract_keywords(file_content, top_n=10)
                overlap = len(set(keywords) & set(file_keywords))
                if overlap >= 2:
                    return md_file
            except Exception:
                continue
        
        return None
    
    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """提取关键词"""
        words = re.findall(r'[\u4e00-\u9fa5]{2,8}', text)
        word_freq = Counter(words)
        stop_words = {"这个", "那个", "我们", "你们", "他们", "什么", "怎么", "还是",
                      "但是", "因为", "所以", "然后", "现在", "今天", "明天", "昨天",
                      "这里", "那里", "这样", "那样", "可以", "需要", "进行", "完成"}
        filtered = [(w, c) for w, c in word_freq.items() if w not in stop_words and len(w) >= 2]
        return [w for w, c in sorted(filtered, key=lambda x: x[1], reverse=True)[:top_n]]
    
    def _create_permanent(self, type_dir: Path, ptype: str, trigger: str, session_id: str,
                          user_msg: str, assistant_msg: str, timestamp: float) -> Path:
        """创建新的永久记录"""
        date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
        keywords = self._extract_keywords(user_msg + assistant_msg, top_n=3)
        keyword = "_".join(keywords) if keywords else "record"
        filename = f"{date_str}_{trigger}_{keyword}.md"
        filepath = type_dir / filename
        
        # 过滤敏感信息
        safe_user = self._filter_sensitive(user_msg)
        safe_assistant = self._filter_sensitive(assistant_msg)
        
        content = f"""# {ptype} - {trigger}

**创建时间**: {datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")}  
**最新更新**: {datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")}  
**会话**: {session_id}  
**触发**: {trigger}

## 版本历史

### v1 ({datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")})

**用户**: {safe_user}

**助手**: {safe_assistant}

## 关键信息提取

- 类型: {ptype}
- 关键词: {keyword}
- 重要性: 永久保存
- 版本: 1

---
*自动保存 by MiniLoci*
"""
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        self._git_commit_permanent(filepath, f"{ptype}: create new record")
        logger.info(f"Created permanent record: {filepath}")
        return filepath
    
    def _update_permanent(self, existing_file: Path, user_msg: str,
                          assistant_msg: str, timestamp: float) -> Path:
        """更新已有记录"""
        try:
            existing_content = existing_file.read_text(encoding="utf-8")
            
            # 提取版本号
            version_match = re.search(r'- 版本: (\d+)', existing_content)
            current_version = int(version_match.group(1)) if version_match else 1
            new_version = current_version + 1
            
            # 过滤敏感信息
            safe_user = self._filter_sensitive(user_msg)
            safe_assistant = self._filter_sensitive(assistant_msg)
            
            # 新版本条目
            new_entry = f"""
### v{new_version} ({datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")})

**用户**: {safe_user}

**助手**: {safe_assistant}
"""
            
            # 插入到版本历史
            insert_point = existing_content.find("## 关键信息提取")
            if insert_point != -1:
                updated_content = (
                    existing_content[:insert_point] +
                    new_entry + "\n" +
                    existing_content[insert_point:]
                )
            else:
                updated_content = existing_content + new_entry
            
            # 更新元数据
            updated_content = re.sub(
                r'\*\*最新更新\*\*: .+',
                f'**最新更新**: {datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")}',
                updated_content
            )
            updated_content = re.sub(
                r'- 版本: \d+',
                f'- 版本: {new_version}',
                updated_content
            )
            
            # 写回
            with open(existing_file, "w", encoding="utf-8") as f:
                f.write(updated_content)
            
            self._git_commit_permanent(existing_file, f"{existing_file.parent.name}: update to v{new_version}")
            logger.info(f"Updated permanent record: {existing_file} -> v{new_version}")
            return existing_file
            
        except Exception as e:
            logger.warning(f"Update permanent failed: {e}")
            return existing_file
    
    def _git_commit_permanent(self, filepath: Path, message: str):
        """Git版本控制"""
        try:
            permanent_dir = filepath.parent.parent
            git_dir = permanent_dir / ".git"
            
            if not git_dir.exists():
                subprocess.run(["git", "init"], cwd=permanent_dir, capture_output=True, check=False)
                gitignore = permanent_dir / ".gitignore"
                gitignore.write_text("*.db\n*.db-wal\n*.db-shm\nbackups/\n", encoding="utf-8")
            
            subprocess.run(["git", "add", str(filepath)], cwd=permanent_dir, capture_output=True, check=False)
            subprocess.run(["git", "commit", "-m", message], cwd=permanent_dir, capture_output=True, check=False)
        except Exception:
            pass
    
    def _filter_sensitive(self, text: str) -> str:
        """过滤敏感信息"""
        patterns = [
            (r'[A-Za-z0-9_\-]{32,}', '[TOKEN]'),
            (r'password\s*=\s*["\'][^"\']+["\']', 'password = [HIDDEN]'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'secret = [HIDDEN]'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'api_key = [HIDDEN]'),
            (r'[0-9a-f]{40,}', '[HASH]'),
        ]
        
        filtered = text
        for pattern, replacement in patterns:
            filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)
        
        return filtered
    
    # ==================== 核心API ====================
    
    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = ""):
        """每轮对话后保存"""
        if not session_id:
            session_id = getattr(self, 'session_id', '')
        if not session_id:
            return
        
        self.session_id = session_id
        
        # 检测永久保存
        is_permanent, ptype, trigger = self._detect_permanent_save(user_content, assistant_content)
        
        if is_permanent:
            self._save_permanent(session_id, user_content, assistant_content, ptype, trigger, time.time())
        
        # 检测重要性
        importance, tags = self._detect_importance(user_content, assistant_content)
        if is_permanent:
            importance = max(importance, 3)
        
        # 构建metadata
        metadata = {"source": "auto"}
        if is_permanent:
            metadata["permanent"] = True
            metadata["permanent_type"] = ptype
            metadata["permanent_trigger"] = trigger
        
        # 确保会话存在
        self._db.execute(
            """INSERT OR IGNORE INTO sessions (id, start_time, platform)
            VALUES (?, ?, 'cli')""",
            (session_id, time.time())
        )
        
        # 保存用户消息；先插入再用 rowid 派生稳定 trace_id，保证可回溯且不依赖外部 UUID。
        cursor = self._db.execute(
            """INSERT INTO turns (session_id, timestamp, role, content, importance, tags, metadata)
            VALUES (?, ?, 'user', ?, ?, ?, ?)""",
            (session_id, time.time(), user_content, importance, json.dumps(tags), json.dumps(metadata))
        )
        user_rowid = cursor.lastrowid
        user_trace_id = f"turn-{user_rowid}"
        user_metadata = dict(metadata)
        user_metadata["trace_id"] = user_trace_id
        self._db.execute(
            "UPDATE turns SET trace_id = ?, metadata = ? WHERE id = ?",
            (user_trace_id, json.dumps(user_metadata), user_rowid)
        )
        
        # 保存助手消息
        cursor = self._db.execute(
            """INSERT INTO turns (session_id, timestamp, role, content, importance, tags, metadata)
            VALUES (?, ?, 'assistant', ?, ?, ?, ?)""",
            (session_id, time.time(), assistant_content, importance, json.dumps(tags), json.dumps(metadata))
        )
        asst_rowid = cursor.lastrowid
        asst_trace_id = f"turn-{asst_rowid}"
        asst_metadata = dict(metadata)
        asst_metadata["trace_id"] = asst_trace_id
        self._db.execute(
            "UPDATE turns SET trace_id = ?, metadata = ? WHERE id = ?",
            (asst_trace_id, json.dumps(asst_metadata), asst_rowid)
        )
        
        # 同步FTS：写入 Python 预分词 token soup，避免依赖 native 中文 tokenizer。
        try:
            self._db.execute(
                "INSERT INTO turns_fts (rowid, content, tags) VALUES (?, ?, ?)",
                (user_rowid, self._tokenize_for_fts(user_content), self._tokenize_for_fts(json.dumps(tags, ensure_ascii=False)))
            )
            self._db.execute(
                "INSERT INTO turns_fts (rowid, content, tags) VALUES (?, ?, ?)",
                (asst_rowid, self._tokenize_for_fts(assistant_content), self._tokenize_for_fts(json.dumps(tags, ensure_ascii=False)))
            )
        except Exception as e:
            logger.debug(f"FTS sync failed: {e}")
        
        # 轻量 L1 结构化记忆提取：先用规则提取稳定事实/长期指令，后续可替换为 LLM 提取器。
        self._extract_and_store_atoms(
            session_id,
            user_content,
            assistant_content,
            user_rowid,
            asst_rowid,
            [user_trace_id, asst_trace_id],
            importance,
            tags,
        )

        # 更新会话计数
        self._db.execute(
            "UPDATE sessions SET turn_count = turn_count + 2 WHERE id = ?",
            (session_id,)
        )
        
        # 立即提交
        self._db.commit()
        
        # 异步计算向量（不阻塞主流程）：提交到单 worker 队列，避免并发调用 torch/Faiss
        self._enqueue_vector_task(user_rowid, user_content, asst_rowid, assistant_content)
    
    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """召回相关记忆"""
        if not self._is_recall_query(query):
            return ""
        
        results = self._hybrid_search(query)
        return self._format_results(results, style=self.default_style)
    
    def _is_recall_query(self, query: str) -> bool:
        """检测回忆查询"""
        return any(p in query for p in self.recall_patterns)
    
    def _strip_recall_triggers(self, query: str) -> str:
        """剔除“你还记得/之前/上次”等触发词，保留实质查询内容。"""
        clean_query = query or ""
        for pattern in self.recall_patterns:
            clean_query = clean_query.replace(pattern, "")
        clean_query = re.sub(r"[你我他她它们咱的那个这个一下还记得吗呢吧？?。！!，,、]+", " ", clean_query)
        clean_query = re.sub(r"\s+", " ", clean_query).strip()
        return clean_query or (query or "")

    def _segment_text(self, text: str) -> List[str]:
        """中英混合分词：优先 jieba；无 jieba 时使用中文 2-4 字滑窗 + ASCII token。"""
        text = str(text or "")
        words: List[str] = []

        # ASCII 技术词先保留，避免 MiniLoci/Faiss/Docker/CI 这类词丢失。
        words.extend(re.findall(r"[A-Za-z][A-Za-z0-9_]*|[0-9]+[A-Za-z0-9_]*", text))

        try:
            import jieba
            for token in jieba.lcut(text):
                token = token.strip()
                if token:
                    words.append(token)
        except ImportError:
            for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
                if 2 <= len(chunk) <= 8:
                    words.append(chunk)
                for i in range(len(chunk)):
                    for size in (4, 3, 2):
                        if i + size <= len(chunk):
                            words.append(chunk[i:i + size])

        # 对含中英文混合的短片段再拆一次 ASCII/中文，减少 jieba 不可用或 JSON tags 的噪声。
        for chunk in re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_]*", text):
            words.append(chunk)

        seen = set()
        unique_words = []
        for word in words:
            word = str(word).strip()
            if not word or len(word) < 2:
                continue
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
        return unique_words

    def _search_terms(self, text: str, *, expand_synonyms: bool = True) -> List[str]:
        """生成安全搜索词，供 FTS MATCH、token soup 和 LIKE fallback 复用。"""
        expanded = []
        for word in self._segment_text(text):
            expanded.append(word)
            if expand_synonyms and word in self.synonyms:
                expanded.extend(self.synonyms[word])
        return self._sanitize_fts_keywords(expanded)

    def _tokenize_for_fts(self, *texts: str) -> str:
        """把原文转换成 FTS 可索引的 token soup。"""
        terms: List[str] = []
        for text in texts:
            terms.extend(self._search_terms(text, expand_synonyms=True))
        seen = set()
        unique_terms = []
        for term in terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        return " ".join(unique_terms)

    def _expand_query(self, query: str) -> str:
        """同义词扩展 - 先剔除触发词，再分词扩展，输出安全 OR 查询。"""
        clean_query = self._strip_recall_triggers(query)
        safe_keywords = self._search_terms(clean_query, expand_synonyms=True)
        return " OR ".join(safe_keywords)

    def _sanitize_fts_keywords(self, keywords) -> List[str]:
        """清理 FTS5 关键词，避免 / - \" * ^ 等特殊字符或用户输入的布尔词破坏 MATCH 语法。"""
        safe_keywords = []
        seen = set()
        for kw in keywords:
            if kw is None:
                continue
            # FTS5 特殊字符统一转为空格；保留中文、英文、数字、下划线。
            safe_kw = re.sub(r'[/\-"*^:(){}\[\]\\]', ' ', str(kw)).strip()
            for part in re.split(r'\s+', safe_kw):
                part = part.strip()
                if not part or len(part) < 2:
                    continue
                if part.upper() in {"OR", "AND", "NOT", "NEAR"}:
                    continue
                if part not in seen:
                    seen.add(part)
                    safe_keywords.append(part)
        return safe_keywords
    
    def _hybrid_search(self, query: str, limit: int = 5) -> List[Dict]:
        """混合搜索 - FTS5 unicode61/token soup + 可选 numpy/Faiss 向量检索。"""
        cutoff = time.time() - self.window_days * 24 * 3600
        
        # 使用_expand_query进行分词和同义词扩展
        fts_results = []
        try:
            or_query = self._expand_query(query)
            
            # 加回时间过滤，只搜索窗口期内的记录
            rows = self._db.execute("""
                SELECT t.id, t.content, t.role, t.timestamp, t.importance, t.tags, t.session_id, t.trace_id
                FROM turns_fts
                JOIN turns t ON turns_fts.rowid = t.id
                WHERE turns_fts MATCH ? AND t.timestamp > ?
                ORDER BY rank
                LIMIT ?
            """, (or_query, cutoff, limit * 3)).fetchall()
            
            for row in rows:
                fts_results.append({
                    'id': row[0],
                    'content': row[1],
                    'role': row[2],
                    'timestamp': row[3],
                    'importance': row[4],
                    'tags': json.loads(row[5]) if row[5] else [],
                    'source_session_id': row[6],
                    'trace_id': row[7] or f"turn-{row[0]}",
                    'source_turn_ids': [row[0]],
                    'fts_score': 0.8
                })
        except Exception as e:
            logger.debug(f"FTS search failed: {e}")
            self._mark_degraded("fts", e)
            # 降级到 LIKE 查询：使用清洗后的关键词/同义词，而不是原始“你还记得...”整句。
            try:
                like_keywords = self._search_terms(self._strip_recall_triggers(query), expand_synonyms=True)[:12]
                if not like_keywords:
                    like_keywords = self._sanitize_fts_keywords([self._strip_recall_triggers(query)])[:3]
                if like_keywords:
                    clauses = " OR ".join(["content LIKE ?" for _ in like_keywords])
                    params = [f"%{kw}%" for kw in like_keywords]
                    rows = self._db.execute(f"""
                        SELECT id, content, role, timestamp, importance, tags, session_id, trace_id
                        FROM turns
                        WHERE ({clauses}) AND timestamp > ?
                        ORDER BY importance DESC, timestamp DESC
                        LIMIT ?
                    """, (*params, cutoff, limit * 3)).fetchall()
                else:
                    rows = []
                
                for row in rows:
                    fts_results.append({
                        'id': row[0],
                        'content': row[1],
                        'role': row[2],
                        'timestamp': row[3],
                        'importance': row[4],
                        'tags': json.loads(row[5]) if row[5] else [],
                        'source_session_id': row[6],
                        'trace_id': row[7] or f"turn-{row[0]}",
                        'source_turn_ids': [row[0]],
                        'fts_score': 0.5
                    })
            except Exception as e2:
                logger.debug(f"LIKE fallback failed: {e2}")
                self._mark_degraded("fts", e2)
        
        # 向量搜索：Faiss 可选；numpy backend 只要内存矩阵有数据即可搜索。
        vec_results = []
        if self.enable_vector and self._has_vector_index():
            try:
                query_vec = self._embed(query)
                if query_vec:
                    vec_results = self._vector_search(query_vec, limit * 3)
            except Exception as e:
                logger.debug(f"Vector search failed: {e}")
                self._mark_degraded("vector", e)
        
        # RRF 合并排序：FTS/LIKE 与向量各自产生 ranked list，避免依赖不同检索器的原始分数尺度。
        merged = self._rrf_merge_ranked_results(fts_results, vec_results, limit=limit * 3)

        # 向量结果通常只有 id/score，需要补齐 turns 原文数据。
        enriched = []
        for item in merged:
            data = item['data']
            if 'content' not in data:
                row = self._db.execute(
                    "SELECT content, role, timestamp, importance, tags, session_id, trace_id FROM turns WHERE id = ?",
                    (data['id'],)
                ).fetchone()
                if not row:
                    continue
                data.update({
                    'content': row[0],
                    'role': row[1],
                    'timestamp': row[2],
                    'importance': row[3],
                    'tags': json.loads(row[4]) if row[4] else [],
                    'source_session_id': row[5],
                    'trace_id': row[6] or f"turn-{data['id']}",
                    'source_turn_ids': [data['id']]
                })
            self._attach_trace_metadata(data)
            data['search_sources'] = item.get('sources', [])
            data['rrf_score'] = item.get('rrf', 0.0)
            time_weight = self._calc_time_weight(data['timestamp'])
            importance = data.get('importance', 1)
            # RRF 为主；时间和重要性只作为轻量稳定偏置，避免旧的分数尺度问题回流。
            item['total'] = item.get('rrf', 0.0) + time_weight * 0.003 + (importance / 3.0) * 0.003
            enriched.append(item)

        sorted_results = sorted(enriched, key=lambda x: x['total'], reverse=True)
        return [r['data'] for r in sorted_results[:limit]]

    # ==================== L1 结构化记忆 atoms ====================

    def _infer_atom_scene(self, text: str, atom_type: str) -> str:
        """为轻量 atom 生成粗粒度场景名。"""
        if "MiniLoci" in text or "miniloci" in text:
            return "MiniLoci 记忆系统"
        if "Hermes" in text or "hermes" in text:
            return "Hermes Agent"
        if atom_type == "instruction":
            return "用户沟通偏好"
        if atom_type == "project":
            return "项目决策"
        return atom_type

    def _build_atom_content(self, atom_type: str, user_msg: str, assistant_msg: str) -> str:
        """生成独立可读的 atom 文本。"""
        combined = re.sub(r"\s+", " ", f"{user_msg} {assistant_msg}").strip()
        if atom_type == "instruction":
            return f"用户要求 AI 以后回答时：{user_msg.strip()}"
        if atom_type == "project":
            if "决定" in user_msg:
                return f"用户在项目中作出决策：{user_msg.strip()}"
            return f"项目事实：{combined[:240]}"
        if atom_type == "incident":
            return f"问题/故障经验：{combined[:240]}"
        return combined[:240]

    def _extract_atoms_from_turn(self, user_msg: str, assistant_msg: str, importance: int, tags: List[str]) -> List[Dict[str, Any]]:
        """轻量规则版 L1 提取器：先覆盖长期指令、项目决策、故障经验。"""
        combined = f"{user_msg} {assistant_msg}"
        atoms = []

        instruction_markers = ["以后", "以后都", "从现在开始", "以后回答", "不要", "必须", "请记住", "记住"]
        if any(marker in user_msg for marker in instruction_markers) and any(
            kw in user_msg for kw in ["回答", "格式", "表格", "语气", "风格", "必须", "不要", "用"]
        ):
            atoms.append({"type": "instruction", "priority": 90})

        project_markers = ["决定", "方案", "部署", "采用", "选择", "配置", "架构", "MiniLoci", "Hermes", "Docker"]
        if importance >= 2 and any(marker in combined for marker in project_markers):
            atoms.append({"type": "project", "priority": 75})

        incident_markers = ["错误", "报错", "失败", "崩溃", "故障", "修复", "问题", "坑"]
        if any(marker in combined for marker in incident_markers):
            atoms.append({"type": "incident", "priority": 80})

        # 同一 turn 内相同 type 只保留一次。
        deduped = []
        seen = set()
        for atom in atoms:
            if atom["type"] in seen:
                continue
            seen.add(atom["type"])
            content = self._build_atom_content(atom["type"], user_msg, assistant_msg)
            atom.update({"content": content, "scene_name": self._infer_atom_scene(content, atom["type"])})
            deduped.append(atom)
        return deduped

    def _atom_similarity_terms(self, content: str) -> List[str]:
        return self._search_terms(content, expand_synonyms=False)[:16]

    def _find_similar_atom(self, atom_type: str, content: str) -> Optional[int]:
        """用 FTS/LIKE 查找同类型近似 atom，避免重复写入。"""
        terms = self._atom_similarity_terms(content)
        if not terms:
            return None
        fts_query = " OR ".join(terms)
        try:
            row = self._db.execute("""
                SELECT a.id
                FROM memory_atoms_fts f
                JOIN memory_atoms a ON f.rowid = a.id
                WHERE memory_atoms_fts MATCH ? AND a.type = ?
                ORDER BY rank
                LIMIT 1
            """, (fts_query, atom_type)).fetchone()
            if row:
                return int(row[0])
        except Exception:
            pass

        clauses = " OR ".join(["content LIKE ?" for _ in terms[:8]])
        try:
            row = self._db.execute(
                f"SELECT id FROM memory_atoms WHERE type = ? AND ({clauses}) ORDER BY updated_at DESC LIMIT 1",
                (atom_type, *[f"%{t}%" for t in terms[:8]])
            ).fetchone()
            return int(row[0]) if row else None
        except Exception:
            return None

    def _sync_atom_fts(self, atom_id: int, content: str, atom_type: str, scene_name: str):
        try:
            # FTS5 external-content tables do not reliably support normal DELETE syntax.
            # INSERT OR REPLACE is sufficient for our rowid-owned lightweight atom index.
            self._db.execute(
                "INSERT OR REPLACE INTO memory_atoms_fts (rowid, content, type, scene_name) VALUES (?, ?, ?, ?)",
                (atom_id, self._tokenize_for_fts(content), atom_type, self._tokenize_for_fts(scene_name))
            )
        except Exception as e:
            logger.debug(f"Atom FTS sync failed: {e}")

    def _upsert_memory_atom(self, atom: Dict[str, Any], session_id: str, source_turn_ids: List[int], trace_ids: List[str]):
        now = time.time()
        existing_id = self._find_similar_atom(atom["type"], atom["content"])
        if existing_id:
            row = self._db.execute(
                "SELECT source_turn_ids, trace_ids, content, priority, scene_name FROM memory_atoms WHERE id = ?",
                (existing_id,)
            ).fetchone()
            if not row:
                return
            merged_turn_ids = sorted(set(json.loads(row[0] or "[]") + source_turn_ids))
            merged_trace_ids = []
            for trace_id in json.loads(row[1] or "[]") + trace_ids:
                if trace_id not in merged_trace_ids:
                    merged_trace_ids.append(trace_id)
            content = row[2] if len(row[2]) >= len(atom["content"]) else atom["content"]
            priority = max(int(row[3] or 0), int(atom.get("priority", 50)))
            scene_name = row[4] or atom.get("scene_name")
            metadata = {"dedup": "merged", "updated_from_turn_ids": source_turn_ids}
            self._db.execute("""
                UPDATE memory_atoms
                SET content = ?, priority = ?, source_turn_ids = ?, trace_ids = ?, scene_name = ?, metadata = ?, updated_at = ?
                WHERE id = ?
            """, (
                content,
                priority,
                json.dumps(merged_turn_ids),
                json.dumps(merged_trace_ids),
                scene_name,
                json.dumps(metadata, ensure_ascii=False),
                now,
                existing_id,
            ))
            self._sync_atom_fts(existing_id, content, atom["type"], scene_name or "")
            return

        cursor = self._db.execute("""
            INSERT INTO memory_atoms (content, type, priority, source_turn_ids, trace_ids, source_session_id, scene_name, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            atom["content"],
            atom["type"],
            atom.get("priority", 50),
            json.dumps(source_turn_ids),
            json.dumps(trace_ids),
            session_id,
            atom.get("scene_name"),
            json.dumps({"extractor": "rules-v1"}, ensure_ascii=False),
            now,
            now,
        ))
        self._sync_atom_fts(cursor.lastrowid, atom["content"], atom["type"], atom.get("scene_name") or "")

    def _extract_and_store_atoms(self, session_id: str, user_msg: str, assistant_msg: str, user_rowid: int, asst_rowid: int, trace_ids: List[str], importance: int, tags: List[str]):
        atoms = self._extract_atoms_from_turn(user_msg, assistant_msg, importance, tags)
        for atom in atoms:
            self._upsert_memory_atom(atom, session_id, [user_rowid, asst_rowid], trace_ids)

    def _format_atom_row(self, row) -> Dict[str, Any]:
        return {
            "id": row[0],
            "content": row[1],
            "type": row[2],
            "priority": row[3],
            "source_turn_ids": json.loads(row[4] or "[]"),
            "trace_ids": json.loads(row[5] or "[]"),
            "source_session_id": row[6],
            "scene_name": row[7],
            "metadata": json.loads(row[8] or "{}"),
            "created_at": row[9],
            "updated_at": row[10],
        }

    def search_atoms(self, query: str, limit: int = 5, atom_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索 L1 结构化记忆 atoms。"""
        terms = self._search_terms(query, expand_synonyms=True)
        if not terms:
            return []
        fts_query = " OR ".join(terms)
        params = [fts_query]
        type_clause = ""
        if atom_type:
            type_clause = " AND a.type = ?"
            params.append(atom_type)
        params.append(limit)
        try:
            rows = self._db.execute(f"""
                SELECT a.id, a.content, a.type, a.priority, a.source_turn_ids, a.trace_ids,
                       a.source_session_id, a.scene_name, a.metadata, a.created_at, a.updated_at
                FROM memory_atoms_fts f
                JOIN memory_atoms a ON f.rowid = a.id
                WHERE memory_atoms_fts MATCH ?{type_clause}
                ORDER BY rank
                LIMIT ?
            """, tuple(params)).fetchall()
            return [self._format_atom_row(row) for row in rows]
        except Exception as e:
            self._mark_degraded("fts", e)
            like_terms = terms[:8]
            clauses = " OR ".join(["content LIKE ?" for _ in like_terms])
            params = [f"%{t}%" for t in like_terms]
            type_clause = ""
            if atom_type:
                type_clause = " AND type = ?"
                params.append(atom_type)
            params.append(limit)
            rows = self._db.execute(f"""
                SELECT id, content, type, priority, source_turn_ids, trace_ids,
                       source_session_id, scene_name, metadata, created_at, updated_at
                FROM memory_atoms
                WHERE ({clauses}){type_clause}
                ORDER BY priority DESC, updated_at DESC
                LIMIT ?
            """, tuple(params)).fetchall()
            return [self._format_atom_row(row) for row in rows]

    def _calc_time_weight(self, timestamp: float) -> float:
        """计算时间权重"""
        now = time.time()
        age_hours = (now - timestamp) / 3600
        
        if age_hours <= 24:
            return 1.0
        elif age_hours <= 48:
            return 0.7
        elif age_hours <= 72:
            return 0.4
        else:
            return 0.1
    
    def _format_results(self, results: List[Dict], style: str = "concise") -> str:
        """格式化搜索结果"""
        if not results:
            return ""
        
        if style == "caveman":
            parts = ["## 记忆"]
            for r in results:
                time_str = datetime.fromtimestamp(r['timestamp']).strftime("%m-%d")
                parts.append(f"- [{time_str}] {r['role']}: {r['content'][:50]}...")
            return "\n".join(parts)
        
        elif style == "terse":
            parts = ["## 相关"]
            for r in results:
                parts.append(f"- {r['content'][:80]}")
            return "\n".join(parts)
        
        else:  # concise (默认)
            parts = ["## 相关记忆"]
            for r in results:
                time_str = datetime.fromtimestamp(r['timestamp']).strftime("%m-%d %H:%M")
                tag_str = ", ".join(r.get('tags', [])) if r.get('tags') else ""
                parts.append(
                    f"- [{time_str}] {r['role']}: {r['content'][:100]}... "
                    f"(重要性:{r.get('importance', 1)}, 标签:{tag_str})"
                )
            return "\n".join(parts)
    
    # ==================== 配置管理 ====================
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        config_path = Path(self.hermes_home) / "loci-archive" / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def save_config(self, values: Dict[str, Any]) -> None:
        """保存配置"""
        config_path = Path(self.hermes_home) / "loci-archive" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(values, f, indent=2, ensure_ascii=False)
        self._config = values
    
    def get_config_schema(self) -> List[Dict[str, Any]]:
        """配置字段定义"""
        return [
            {
                "key": "window_days",
                "description": "短期记忆保留天数",
                "default": 3,
                "type": "integer",
                "min": 1,
                "max": 30
            },
            {
                "key": "fts_weight",
                "description": "关键词匹配权重(0-1)",
                "default": 0.45,
                "type": "float",
                "min": 0.0,
                "max": 1.0
            },
            {
                "key": "vector_weight",
                "description": "向量语义权重(0-1)",
                "default": 0.25,
                "type": "float",
                "min": 0.0,
                "max": 1.0
            },
            {
                "key": "enable_vector",
                "description": "是否启用向量搜索",
                "default": True,
                "type": "boolean"
            },
            {
                "key": "vector_model",
                "description": "Embedding模型名称",
                "default": "BAAI/bge-small-zh-v1.5",
                "type": "string",
                "choices": [
                    "BAAI/bge-small-zh-v1.5",
                    "all-MiniLM-L6-v2",
                    "paraphrase-MiniLM-L3-v2"
                ]
            },
            {
                "key": "vector_backend",
                "description": "向量检索后端：auto/faiss/numpy；默认 auto，Faiss 缺失时自动使用 numpy",
                "default": "auto",
                "type": "string",
                "choices": ["auto", "faiss", "numpy"]
            },
            {
                "key": "vector_local_files_only",
                "description": "向量模型默认只从本地 HuggingFace 缓存加载，避免 Gateway 被在线 HEAD 请求超时阻塞；首次下载时可显式设为 false",
                "default": True,
                "type": "boolean"
            },
            {
                "key": "default_style",
                "description": "默认回答风格",
                "default": "concise",
                "type": "string",
                "choices": ["normal", "concise", "terse", "caveman"]
            },
            {
                "key": "auto_cleanup",
                "description": "是否启用自动清理",
                "default": True,
                "type": "boolean"
            },
            {
                "key": "backup_count",
                "description": "保留备份数量",
                "default": 7,
                "type": "integer",
                "min": 1,
                "max": 30
            }
        ]
    
    # ==================== 系统提示与工具 ====================
    
    def system_prompt_block(self) -> str:
        """系统提示中的MiniLoci状态"""
        try:
            stats = self._db.execute("""
                SELECT 
                    COUNT(*) as total_turns,
                    COUNT(DISTINCT session_id) as sessions,
                    SUM(CASE WHEN importance >= 2 THEN 1 ELSE 0 END) as important_turns
                FROM turns
                WHERE timestamp > ?
            """, (time.time() - self.window_days * 24 * 3600,)).fetchone()
            
            return f"""# MiniLoci 状态
最近{self.window_days}天: {stats[0]} 轮对话, {stats[1]} 个会话, {stats[2]} 条重要记录
触发词: "你还记得""之前""上次""我们讨论过""那个..."""
        except Exception:
            return "# MiniLoci 状态\n记忆系统运行中"
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """暴露搜索工具"""
        return [{
            "name": "miniloci_search",
            "description": "搜索MiniLoci记忆库中的历史对话",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "days": {"type": "integer", "description": "搜索最近N天", "default": 3},
                    "limit": {"type": "integer", "description": "返回条数", "default": 5}
                },
                "required": ["query"]
            }
        }]
    
    def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        """处理工具调用"""
        if tool_name == "miniloci_search":
            return self._tool_search(args)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    
    def _tool_search(self, args: Dict[str, Any]) -> str:
        """工具搜索实现：复用方案D的分词、清洗、OR 查询与时间过滤。"""
        query = args.get("query", "")
        days = args.get("days", self.window_days)
        limit = args.get("limit", 5)
        
        cutoff = time.time() - days * 24 * 3600
        fts_query = self._expand_query(query)
        
        try:
            results = self._db.execute("""
                SELECT t.content, t.role, t.timestamp, t.importance, t.tags
                FROM turns_fts f
                JOIN turns t ON f.rowid = t.id
                WHERE t.timestamp > ? AND turns_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (cutoff, fts_query, limit)).fetchall()
            
            return json.dumps({
                "query": query,
                "fts_query": fts_query,
                "results": [
                    {
                        "role": r[1],
                        "time": datetime.fromtimestamp(r[2]).strftime("%m-%d %H:%M"),
                        "content": r[0][:200],
                        "importance": r[3],
                        "tags": json.loads(r[4]) if r[4] else []
                    }
                    for r in results
                ]
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"query": query, "fts_query": fts_query, "error": str(e)}, ensure_ascii=False)
    
    def on_session_end(self, messages: List[Dict[str, Any]]):
        """会话结束标记"""
        if not getattr(self, "_db", None):
            return
        if hasattr(self, 'session_id') and self.session_id:
            try:
                self._db.execute(
                    "UPDATE sessions SET end_time = ?, status = 'closed' WHERE id = ?",
                    (time.time(), self.session_id)
                )
                self._db.commit()
            except Exception as e:
                logger.warning(f"Session end marking failed: {e}")
    
    # ==================== 备份机制 ====================
    
    def _daily_backup(self):
        """每日备份"""
        try:
            backup_dir = self.db_dir / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"sessions_{timestamp}.db"
            
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            # 保留最近N个备份
            backups = sorted(backup_dir.glob("sessions_*.db"))
            for old in backups[:-self.backup_count]:
                old.unlink()
            
            logger.info(f"Database backed up: {backup_path}")
        except Exception as e:
            logger.warning(f"Backup failed: {e}")
    
    def _check_db_integrity(self) -> bool:
        """检查数据库完整性"""
        try:
            result = self._db.execute("PRAGMA integrity_check;").fetchone()
            return result[0] == "ok" if result else False
        except Exception as e:
            logger.error(f"Database integrity check failed: {e}")
            return False


# ==================== 插件注册入口 ====================

def register(ctx):
    """注册MiniLoci插件"""
    provider = MiniLociProvider()
    
    # 检查是否有 register_memory_provider 方法
    if hasattr(ctx, 'register_memory_provider'):
        ctx.register_memory_provider(provider)
        logger.info("MiniLoci registered as memory provider")
    else:
        logger.warning("Context does not support memory provider registration")
    
    return provider
