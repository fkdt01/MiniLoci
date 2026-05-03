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
        self._pending_vectors = []
        self._flush_threshold = 100
        self._last_flush_time = time.time()
        self._flush_interval = 1800  # 30分钟
        self._vector_queue = queue.Queue()
        self._vector_worker = None
        
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
        
        # 初始化Faiss与单 worker 向量队列 (如果启用)
        if self.enable_vector:
            try:
                self._init_faiss()
                self._load_vectors_from_db()
                self._start_vector_worker()
                logger.info(f"Vector search ready: faiss_index={self._faiss_index is not None}, "
                           f"vectors_loaded={len(self._vector_map)}, "
                           f"model_loaded={self._vector_model_loaded}")
            except Exception as e:
                logger.warning(f"Vector search initialization failed: {e}")
                self.enable_vector = False
        
        logger.info(f"MiniLoci initialized for session {session_id}")
    
    def _load_simple_extension(self):
        """加载simple tokenizer扩展"""
        try:
            import os
            # 设置jieba字典路径
            jieba_dict_path = '/tmp/simple/build/cppjieba/src/cppjieba'
            if os.path.exists(jieba_dict_path):
                os.chdir(jieba_dict_path)
            
            # 加载扩展
            self._db.enable_load_extension(True)
            self._db.load_extension('/tmp/simple/build/src/libsimple')
            logger.info("Simple tokenizer extension loaded")
        except Exception as e:
            logger.warning(f"Failed to load simple extension: {e}")
    
    # ==================== 数据库操作 ====================
    
    def _init_db(self):
        """初始化数据库和WAL模式"""
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA temp_store=MEMORY")
        
        # 加载simple tokenizer扩展
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
                vector BLOB,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            
            CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                content, 
                tags,
                tokenize='simple',
                content='turns', 
                content_rowid='id'
            );
            
            CREATE INDEX IF NOT EXISTS idx_turns_time ON turns(timestamp);
            CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
            CREATE INDEX IF NOT EXISTS idx_turns_importance ON turns(importance);
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
        
        self._db.execute(f"PRAGMA user_version = {version}")
    
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
    
    def _init_faiss(self):
        """初始化Faiss索引"""
        import faiss
        
        dimension = 512  # bge-small-zh-v1.5
        self._faiss_index = faiss.IndexFlatIP(dimension)  # 使用内积索引，更简单稳定
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
                
                # 同时设置本地缓存目录，避免重复下载
                cache_dir = Path(self.hermes_home) / "loci-archive" / "models"
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Loading vector model: {self.vector_model_name} (this may take a moment)...")
                self._vector_model = SentenceTransformer(
                    self.vector_model_name,
                    cache_folder=str(cache_dir)
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
        """启动时从SQLite加载向量"""
        cutoff = time.time() - self.window_days * 24 * 3600
        rows = self._db.execute("""
            SELECT id, vector FROM turns 
            WHERE timestamp > ? AND vector IS NOT NULL
        """, (cutoff,)).fetchall()
        
        if not rows:
            return
        
        vectors = []
        for turn_id, vec_blob in rows:
            try:
                vec = np.frombuffer(vec_blob, dtype=np.float32)
                if len(vec) == 512:
                    vectors.append(vec)
                    self._vector_map[self._next_faiss_id] = turn_id
                    self._next_faiss_id += 1
            except Exception:
                continue
        
        if vectors:
            vectors_array = np.array(vectors)
            self._faiss_index.add(vectors_array)
            logger.info(f"Loaded {len(vectors)} vectors into Faiss")
    
    def _add_vector_async(self, turn_id: int, text: str):
        """异步添加向量"""
        if not self.enable_vector or not self._faiss_index:
            return
        
        try:
            vec = self._embed(text)
            if not vec:
                return
            
            with self._vector_lock:
                vec_array = np.array([vec], dtype=np.float32)
                self._faiss_index.add(vec_array)
                
                faiss_id = self._next_faiss_id
                self._vector_map[faiss_id] = turn_id
                self._next_faiss_id += 1
                
                # 加入待持久化队列
                self._pending_vectors.append((turn_id, np.array(vec, dtype=np.float32).tobytes()))
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
    
    def _vector_search(self, query_vec: list, limit: int = 10) -> List[Dict]:
        """Faiss向量搜索"""
        if not self._faiss_index or not query_vec:
            return []
        
        try:
            with self._vector_lock:
                query_array = np.array([query_vec], dtype=np.float32)
                distances, indices = self._faiss_index.search(query_array, limit)
                vector_map_snapshot = dict(self._vector_map)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                turn_id = vector_map_snapshot.get(int(idx))
                if turn_id:
                    # IndexFlatIP 返回的是内积，对于归一化向量，内积=余弦相似度
                    # 范围 [-1, 1]，裁剪到 [0, 1] 作为最终分数
                    similarity = max(0.0, float(dist))
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
        
        # 保存用户消息
        cursor = self._db.execute(
            """INSERT INTO turns (session_id, timestamp, role, content, importance, tags, metadata)
            VALUES (?, ?, 'user', ?, ?, ?, ?)""",
            (session_id, time.time(), user_content, importance, json.dumps(tags), json.dumps(metadata))
        )
        user_rowid = cursor.lastrowid
        
        # 保存助手消息
        cursor = self._db.execute(
            """INSERT INTO turns (session_id, timestamp, role, content, importance, tags, metadata)
            VALUES (?, ?, 'assistant', ?, ?, ?, ?)""",
            (session_id, time.time(), assistant_content, importance, json.dumps(tags), json.dumps(metadata))
        )
        asst_rowid = cursor.lastrowid
        
        # 同步FTS
        try:
            self._db.execute(
                "INSERT INTO turns_fts (rowid, content, tags) VALUES (?, ?, ?)",
                (user_rowid, user_content, json.dumps(tags))
            )
            self._db.execute(
                "INSERT INTO turns_fts (rowid, content, tags) VALUES (?, ?, ?)",
                (asst_rowid, assistant_content, json.dumps(tags))
            )
        except Exception as e:
            logger.debug(f"FTS sync failed: {e}")
        
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
    
    def _expand_query(self, query: str) -> str:
        """同义词扩展 - 先剔除触发词，再分词扩展"""
        # 1. 剔除触发词，保留实质内容
        clean_query = query
        for pattern in self.recall_patterns:
            clean_query = clean_query.replace(pattern, '')
        clean_query = clean_query.strip('？?吗呢吧')
        if not clean_query:
            clean_query = query  # 如果剔完为空，用原查询
        
        # 2. 分词
        try:
            import jieba
            words = jieba.lcut(clean_query)
        except ImportError:
            # jieba未安装时，按2-4字滑动窗口拆分
            words = []
            for i in range(len(clean_query) - 1):
                for size in [4, 3, 2]:
                    if i + size <= len(clean_query):
                        chunk = clean_query[i:i + size]
                        if len(chunk) >= 2 and not chunk.isdigit():
                            words.append(chunk)
            # 去重
            seen = set()
            unique_words = []
            for w in words:
                if w not in seen:
                    seen.add(w)
                    unique_words.append(w)
            words = unique_words[:20]
        
        # 3. 扩展同义词
        expanded = set()
        for word in words:
            word = word.strip()
            if len(word) >= 2:
                expanded.add(word)
                if word in self.synonyms:
                    expanded.update(self.synonyms[word])
        
        # 4. 构建FTS查询 - 使用简单OR连接
        if expanded:
            safe_keywords = self._sanitize_fts_keywords(expanded)
            if safe_keywords:
                return " OR ".join(safe_keywords)
        
        fallback_keywords = self._sanitize_fts_keywords([clean_query])
        return " OR ".join(fallback_keywords) if fallback_keywords else clean_query

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
        """混合搜索 - 使用simple tokenizer + FTS5"""
        cutoff = time.time() - self.window_days * 24 * 3600
        
        # 使用_expand_query进行分词和同义词扩展
        fts_results = []
        try:
            or_query = self._expand_query(query)
            
            # 加回时间过滤，只搜索窗口期内的记录
            rows = self._db.execute("""
                SELECT t.id, t.content, t.role, t.timestamp, t.importance, t.tags
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
                    'fts_score': 0.8
                })
        except Exception as e:
            logger.debug(f"FTS search failed: {e}")
            # 降级到LIKE查询（也加时间过滤）
            try:
                rows = self._db.execute("""
                    SELECT id, content, role, timestamp, importance, tags
                    FROM turns
                    WHERE content LIKE ? AND timestamp > ?
                    ORDER BY importance DESC, timestamp DESC
                    LIMIT ?
                """, (f"%{query}%", cutoff, limit * 3)).fetchall()
                
                for row in rows:
                    fts_results.append({
                        'id': row[0],
                        'content': row[1],
                        'role': row[2],
                        'timestamp': row[3],
                        'importance': row[4],
                        'tags': json.loads(row[5]) if row[5] else [],
                        'fts_score': 0.5
                    })
            except Exception as e2:
                logger.debug(f"LIKE fallback failed: {e2}")
        
        # 向量搜索：只在模型已由后台 worker 加载且索引已有数据时启用。
        # prefetch 不能主动触发模型加载，否则首次召回会被 sentence-transformers 冷启动阻塞。
        vec_results = []
        if self.enable_vector and self._vector_model is not None and self._faiss_index is not None:
            try:
                with self._vector_lock:
                    has_vectors = self._faiss_index.ntotal > 0
                if has_vectors:
                    query_vec = self._embed(query)
                    if query_vec:
                        vec_results = self._vector_search(query_vec, limit * 3)
            except Exception as e:
                logger.debug(f"Vector search failed: {e}")
        
        # 合并排序
        combined = {}
        
        # FTS分数归一化
        if fts_results:
            for r in fts_results:
                combined[r['id']] = {
                    'fts': r['fts_score'],
                    'vec': 0,
                    'data': r
                }
        
        # 向量分数归一化
        if vec_results:
            max_vec = max(r['score'] for r in vec_results) if vec_results else 1.0
            for r in vec_results:
                normalized = r['score'] / max_vec if max_vec > 0 else 0.0
                if r['id'] in combined:
                    combined[r['id']]['vec'] = normalized
                else:
                    row = self._db.execute(
                        "SELECT content, role, timestamp, importance, tags FROM turns WHERE id = ?",
                        (r['id'],)
                    ).fetchone()
                    if row:
                        combined[r['id']] = {
                            'fts': 0,
                            'vec': normalized,
                            'data': {
                                'id': r['id'],
                                'content': row[0],
                                'role': row[1],
                                'timestamp': row[2],
                                'importance': row[3],
                                'tags': json.loads(row[4]) if row[4] else []
                            }
                        }
        
        # 加权计算
        for item in combined.values():
            data = item['data']
            time_weight = self._calc_time_weight(data['timestamp'])
            importance = data['importance']
            
            item['total'] = (
                item['fts'] * self.fts_weight +
                item['vec'] * self.vector_weight +
                time_weight * 0.15 +
                (importance / 3.0) * 0.15
            )
        
        # 排序返回
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x['total'],
            reverse=True
        )
        
        return [r['data'] for r in sorted_results[:limit]]

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
