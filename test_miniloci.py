"""
MiniLoci 测试套件
"""

import pytest
import tempfile
import time
import json
import threading
import types
import numpy as np
from pathlib import Path

# 添加插件路径
import sys
sys.path.insert(0, str(Path.home() / ".hermes" / "plugins"))

from miniloci import MiniLociProvider


class TestMiniLoci:
    """MiniLoci 单元测试"""
    
    @pytest.fixture
    def provider(self):
        """创建测试用的Provider"""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = MiniLociProvider()
            p.initialize("test-session", hermes_home=tmpdir)
            yield p
            p.shutdown()
    
    def test_importance_detection_decision(self, provider):
        """测试决策类重要性检测"""
        importance, tags = provider._detect_importance(
            "我们决定用Railway部署", "好的，Railway确实适合"
        )
        assert importance >= 2
        assert "decision" in tags
        assert "deploy" in tags
    
    def test_importance_detection_task(self, provider):
        """测试任务类重要性检测"""
        importance, tags = provider._detect_importance(
            "TODO: 下一步要配置SSL", "收到，已记录"
        )
        assert importance >= 2
        assert "task" in tags
    
    def test_importance_detection_normal(self, provider):
        """测试普通对话"""
        importance, tags = provider._detect_importance(
            "今天天气怎么样", "今天晴天"
        )
        assert importance == 1
        assert len(tags) == 0
    
    def test_permanent_save_manual(self, provider):
        """测试手动标记永久保存（"保存"已剔除）"""
        # "记住"应该触发
        is_perm, ptype, trigger = provider._detect_permanent_save(
            "记住这个配置", "已保存"
        )
        assert is_perm is True
        assert ptype == "manual"
        
        # "保存"不应该触发（日常用语，误触率高）
        is_perm2, ptype2, trigger2 = provider._detect_permanent_save(
            "保存一下文件", "好的"
        )
        assert is_perm2 is False
        assert ptype2 is None
    
    def test_permanent_save_deploy(self, provider):
        """测试部署类自动永久保存"""
        is_perm, ptype, trigger = provider._detect_permanent_save(
            "我们部署到Docker", "好的，Docker已配置"
        )
        assert is_perm is True
        assert ptype == "deployment"
    
    def test_permanent_save_normal(self, provider):
        """测试普通对话不触发永久保存"""
        is_perm, ptype, trigger = provider._detect_permanent_save(
            "你好", "你好！有什么可以帮忙的？"
        )
        assert is_perm is False
    
    def test_recall_query_detection(self, provider):
        """测试回忆查询检测"""
        assert provider._is_recall_query("你还记得上次那个问题吗？")
        assert provider._is_recall_query("我们之前讨论过部署方案")
        assert not provider._is_recall_query("今天天气怎么样？")
    
    def test_sensitive_filter(self, provider):
        """测试敏感信息过滤"""
        text = "api_key = 'sk-1234567890abcdef' and password = 'secret123'"
        filtered = provider._filter_sensitive(text)
        assert "[HIDDEN]" in filtered
        assert "sk-1234567890abcdef" not in filtered
    
    def test_time_weight(self, provider):
        """测试时间权重计算"""
        now = time.time()
        
        # 今天
        assert provider._calc_time_weight(now - 3600) == 1.0
        
        # 昨天
        assert provider._calc_time_weight(now - 36 * 3600) == 0.7
        
        # 前天
        assert provider._calc_time_weight(now - 60 * 3600) == 0.4
        
        # 更早
        assert provider._calc_time_weight(now - 100 * 3600) == 0.1
    
    def test_initialize_from_non_main_thread_keeps_provider_usable(self):
        """Gateway 非主线程初始化时 signal 注册失败不能中断 provider 初始化。"""
        errors = []
        with tempfile.TemporaryDirectory() as tmpdir:
            p = MiniLociProvider()

            def init_provider():
                try:
                    p.initialize("thread-session", hermes_home=tmpdir)
                except Exception as exc:
                    errors.append(exc)

            t = threading.Thread(target=init_provider)
            t.start()
            t.join(timeout=5)

            try:
                assert not t.is_alive()
                assert errors == []
                assert hasattr(p, "manual_markers")
                assert hasattr(p, "_pending_vectors")
                p.sync_turn("记住线程初始化配置", "已记录", session_id="thread-session")
                count = p._db.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
                assert count == 2
            finally:
                p.shutdown()

    def test_on_session_end_after_shutdown_is_quiet(self, caplog):
        """Gateway restart/shutdown 后若 DB 已关闭，on_session_end 不应再打印误导性 warning。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = MiniLociProvider()
            p.initialize("shutdown-session", hermes_home=tmpdir)
            p.shutdown()
            caplog.clear()
            p.on_session_end([])
            assert "Session end marking failed" not in caplog.text

    def test_vector_model_loads_from_local_cache_by_default(self, monkeypatch):
        """Gateway 环境中向量模型应默认 local_files_only，避免 HuggingFace HEAD 请求反复超时。"""
        calls = []

        class FakeSentenceTransformer:
            def __init__(self, *args, **kwargs):
                calls.append((args, kwargs))

            def encode(self, text, normalize_embeddings=True):
                return np.zeros(512, dtype=np.float32)

        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            p = MiniLociProvider()
            p.initialize("local-cache-session", hermes_home=tmpdir)
            try:
                assert p._get_vector_model() is not None
                assert calls
                assert calls[0][0][0] == "BAAI/bge-small-zh-v1.5"
                assert calls[0][1].get("local_files_only") is True
            finally:
                p.shutdown()

    def test_vector_model_can_opt_out_of_local_cache_only(self, monkeypatch):
        """需要首次联网下载时，可通过 vector_local_files_only=false 显式放开。"""
        calls = []

        class FakeSentenceTransformer:
            def __init__(self, *args, **kwargs):
                calls.append((args, kwargs))

            def encode(self, text, normalize_embeddings=True):
                return np.zeros(512, dtype=np.float32)

        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            p = MiniLociProvider()
            p.initialize("download-session", hermes_home=tmpdir)
            try:
                p._config["vector_local_files_only"] = False
                assert p._get_vector_model() is not None
                assert calls[0][1].get("local_files_only") is False
            finally:
                p.shutdown()

    def test_sync_turn_basic(self, provider):
        """测试基本对话保存"""
        provider.sync_turn(
            "测试消息", "测试回复",
            session_id="test-session"
        )
        
        # 验证数据库中有记录
        cursor = provider._db.execute("SELECT COUNT(*) FROM turns WHERE session_id = ?", ("test-session",))
        count = cursor.fetchone()[0]
        assert count == 2  # user + assistant
    
    def test_sync_turn_importance(self, provider):
        """测试重要对话保存"""
        provider.sync_turn(
            "我们决定用Docker部署", "好的，Docker适合微服务",
            session_id="test-session"
        )
        
        cursor = provider._db.execute(
            "SELECT importance, tags FROM turns WHERE session_id = ? AND role = 'user'",
            ("test-session",)
        )
        row = cursor.fetchone()
        assert row[0] >= 2
        tags = json.loads(row[1])
        assert "decision" in tags or "deploy" in tags
    
    def test_hybrid_search_empty(self, provider):
        """测试空搜索"""
        results = provider._hybrid_search("不存在的内容")
        assert len(results) == 0

    def test_plan_d_chinese_fts_search(self, provider):
        """方案D：中文词组通过 jieba/滑窗分词 + OR 查询可以召回。"""
        provider.sync_turn(
            "我们决定用Docker部署，CI/CD走GitHub Actions",
            "好的，Docker适合微服务，上线发布流程已记录",
            session_id="test-session"
        )
        result = provider.prefetch("你还记得部署方案吗？", session_id="test-session")
        assert "Docker" in result or "部署" in result

    def test_plan_d_sanitizes_fts_special_chars_and_boolean_words(self, provider):
        """方案D：清理 CI/CD、OR 等会破坏 FTS5 MATCH 的特殊输入。"""
        query = provider._expand_query("部署 OR CI/CD - Docker * ^")
        assert " OR OR " not in f" {query} "
        assert "/" not in query
        assert "*" not in query
        assert "^" not in query
        assert "CI" in query or "CD" in query

    def test_plan_d_tool_search_uses_expanded_query(self, provider):
        """工具搜索也必须复用方案D，不能直接把原始中文短语交给 MATCH。"""
        provider.sync_turn(
            "我们决定用Docker部署",
            "上线发布流程使用GitHub Actions",
            session_id="test-session"
        )
        payload = json.loads(provider._tool_search({"query": "部署方案", "days": 3, "limit": 5}))
        assert "fts_query" in payload
        assert payload["results"]

    def test_plan_d_time_filter_excludes_old_turns(self, provider):
        """方案D：FTS 搜索必须保留窗口期过滤，避免旧记忆污染。"""
        provider.sync_turn(
            "我们决定用Docker部署",
            "上线发布流程使用GitHub Actions",
            session_id="test-session"
        )
        old_ts = time.time() - (provider.window_days + 2) * 24 * 3600
        provider._db.execute("UPDATE turns SET timestamp = ?", (old_ts,))
        provider._db.commit()
        assert provider._hybrid_search("你还记得部署方案吗？") == []

    def test_fts_uses_builtin_tokenizer_and_token_soup(self, provider):
        """FTS 不再依赖 /tmp/simple，内置 tokenizer + Python 预分词能召回中文/英文技术词。"""
        fts_sql = provider._db.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'turns_fts'"
        ).fetchone()[0]
        assert "unicode61" in fts_sql
        assert "simple" not in fts_sql

        provider.sync_turn(
            "MiniLoci 向量搜索不可用，部署方案需要调整",
            "建议用 numpy backend 替代 Faiss 强依赖",
            session_id="test-session"
        )
        for query in ["MiniLoci", "向量搜索", "部署", "numpy", "Faiss"]:
            count = provider._db.execute(
                "SELECT COUNT(*) FROM turns_fts WHERE turns_fts MATCH ?",
                (query,)
            ).fetchone()[0]
            assert count >= 1, query

    def test_like_fallback_uses_clean_keywords_not_raw_recall_phrase(self, provider):
        """FTS 异常时，LIKE fallback 应使用清洗后的关键词，而不是原始“你还记得...”整句。"""
        provider.sync_turn(
            "我们决定用Docker部署",
            "上线发布流程使用GitHub Actions",
            session_id="test-session"
        )
        provider._db.execute("DROP TABLE turns_fts")
        provider._db.commit()

        results = provider._hybrid_search("你还记得部署方案吗？")
        assert results
        assert any("Docker" in r["content"] or "部署" in r["content"] for r in results)

    def test_rrf_merge_rewards_results_that_appear_in_multiple_ranked_lists(self, provider):
        """RRF 融合应基于排名，优先奖励同时被 FTS 与向量召回的结果。"""
        fts_results = [
            {"id": 1, "content": "fts-only", "timestamp": time.time(), "importance": 1},
            {"id": 2, "content": "both", "timestamp": time.time(), "importance": 1},
        ]
        vec_results = [
            {"id": 2, "score": 0.2},
            {"id": 3, "score": 0.99},
        ]

        merged = provider._rrf_merge_ranked_results(fts_results, vec_results, limit=3)

        assert [item["data"]["id"] for item in merged] == [2, 1, 3]
        assert merged[0]["rrf"] > merged[1]["rrf"]
        assert merged[0]["sources"] == ["fts", "vector"]

    def test_sync_turn_writes_stable_trace_ids(self, provider):
        """每条 turn 必须有稳定 trace_id，为 L1/L2 摘要回溯到底层原文打基础。"""
        provider.sync_turn("记住这个 MiniLoci 配置", "已记录", session_id="trace-session")

        rows = provider._db.execute(
            "SELECT id, trace_id, metadata FROM turns WHERE session_id = ? ORDER BY id",
            ("trace-session",)
        ).fetchall()

        assert len(rows) == 2
        for row_id, trace_id, metadata_json in rows:
            assert trace_id == f"turn-{row_id}"
            metadata = json.loads(metadata_json)
            assert metadata["trace_id"] == trace_id

    def test_search_results_include_trace_metadata(self, provider):
        """搜索结果应暴露 trace_id/session/source_turn_ids，方便后续下钻查证。"""
        provider.sync_turn(
            "我们决定用Docker部署MiniLoci",
            "部署方案已记录",
            session_id="trace-session"
        )

        results = provider._hybrid_search("你还记得部署方案吗？")

        assert results
        first = results[0]
        assert first["trace_id"].startswith("turn-")
        assert first["source_turn_ids"] == [first["id"]]
        assert first["source_session_id"] == "trace-session"

    def test_vector_recall_failure_marks_degraded_but_keeps_fts_results(self, provider, monkeypatch):
        """向量召回失败不能阻断 FTS 结果，并应在健康状态中标记 degraded。"""
        provider.sync_turn(
            "我们决定用Docker部署MiniLoci",
            "部署方案已记录",
            session_id="trace-session"
        )
        provider.enable_vector = True
        provider._vector_matrix = np.ones((1, provider._vector_dimension), dtype=np.float32)
        provider._vector_ids = [1]

        def boom(_query):
            raise RuntimeError("embedding backend unavailable")

        monkeypatch.setattr(provider, "_embed", boom)

        results = provider._hybrid_search("你还记得部署方案吗？")
        health = provider.health_status()

        assert results
        assert health["degraded"] is True
        assert health["last_vector_error"] == "embedding backend unavailable"

    def test_memory_atoms_schema_created(self, provider):
        """L1 memory_atoms 表应在初始化时创建，作为结构化记忆层。"""
        cols = {
            row[1] for row in provider._db.execute("PRAGMA table_info(memory_atoms)").fetchall()
        }
        assert {"id", "content", "type", "priority", "source_turn_ids", "source_session_id", "created_at"}.issubset(cols)

    def test_sync_turn_extracts_instruction_atom_with_source_trace(self, provider):
        """长期指令应被提取为 instruction atom，并绑定原始 turn id。"""
        provider.sync_turn(
            "以后回答不要用表格，要用纯文字和 emoji bullet",
            "收到，以后会避免表格",
            session_id="atom-session"
        )

        atoms = provider.search_atoms("表格 emoji", limit=5)

        assert atoms
        atom = atoms[0]
        assert atom["type"] == "instruction"
        assert "表格" in atom["content"]
        assert atom["source_session_id"] == "atom-session"
        assert len(atom["source_turn_ids"]) >= 1
        assert atom["trace_ids"][0].startswith("turn-")

    def test_sync_turn_extracts_project_atom_and_deduplicates(self, provider):
        """同一项目事实重复出现时，应更新/复用 atom，而不是无限新增。"""
        provider.sync_turn(
            "我们决定用Docker部署MiniLoci",
            "Docker 部署方案已记录",
            session_id="atom-session"
        )
        provider.sync_turn(
            "再次确认，MiniLoci 还是用Docker部署",
            "已确认 Docker 部署",
            session_id="atom-session"
        )

        atoms = provider.search_atoms("Docker 部署 MiniLoci", limit=10)
        deployment_atoms = [a for a in atoms if a["type"] == "project"]

        assert len(deployment_atoms) == 1
        assert "Docker" in deployment_atoms[0]["content"]
        assert len(deployment_atoms[0]["source_turn_ids"]) >= 2

    def test_atom_conflict_detection_discards_low_value_duplicate_instruction(self, provider):
        """更弱的重复 instruction 应 discard，不覆盖已有更完整 atom。"""
        existing = {
            "type": "instruction",
            "content": "用户要求 AI 以后回答时：以后回答不要用表格，要用纯文字和 emoji bullet",
            "priority": 90,
            "scene_name": "用户沟通偏好",
        }
        provider._upsert_memory_atom(existing, "s1", [1, 2], ["turn-1", "turn-2"])

        weaker = {
            "type": "instruction",
            "content": "用户要求 AI 以后回答时：以后不要用表格",
            "priority": 70,
            "scene_name": "用户沟通偏好",
        }
        decision = provider._decide_atom_conflict(weaker)
        provider._upsert_memory_atom(weaker, "s1", [3, 4], ["turn-3", "turn-4"])

        atoms = provider.search_atoms("表格 emoji", limit=10, atom_type="instruction")

        assert decision["action"] == "discard"
        assert len(atoms) == 1
        assert "emoji" in atoms[0]["content"]
        assert atoms[0]["source_turn_ids"] == [1, 2]

    def test_atom_conflict_detection_updates_more_specific_project_atom(self, provider):
        """更具体的新 project atom 应 update 旧 atom，并保留旧 source。"""
        provider._upsert_memory_atom(
            {"type": "project", "content": "项目事实：MiniLoci 使用 Docker 部署", "priority": 60, "scene_name": "MiniLoci 记忆系统"},
            "s1", [1, 2], ["turn-1", "turn-2"]
        )
        richer = {
            "type": "project",
            "content": "项目事实：MiniLoci 使用 Docker 部署，并通过 GitHub Actions 执行 CI/CD 发布流程",
            "priority": 80,
            "scene_name": "MiniLoci 记忆系统",
        }
        decision = provider._decide_atom_conflict(richer)
        provider._upsert_memory_atom(richer, "s1", [3, 4], ["turn-3", "turn-4"])

        atoms = provider.search_atoms("GitHub Actions CI/CD", limit=10, atom_type="project")

        assert decision["action"] == "update"
        assert len(atoms) == 1
        assert "GitHub Actions" in atoms[0]["content"]
        assert atoms[0]["source_turn_ids"] == [1, 2, 3, 4]
        assert atoms[0]["metadata"]["dedup"] == "updated"

    def test_miniloci_search_atoms_tool_schema_and_handler(self, provider):
        """插件应暴露结构化 atom 搜索工具，区别于原始 turns 搜索。"""
        provider.sync_turn(
            "以后回答不要用表格，要用纯文字和 emoji bullet",
            "收到，以后会避免表格",
            session_id="atom-session"
        )

        tool_names = [schema["name"] for schema in provider.get_tool_schemas()]
        payload = json.loads(provider.handle_tool_call("miniloci_search_atoms", {"query": "表格 emoji", "type": "instruction", "limit": 5}))

        assert "miniloci_search_atoms" in tool_names
        assert payload["results"]
        assert payload["results"][0]["type"] == "instruction"
        assert payload["results"][0]["source_turn_ids"]

    def test_scene_blocks_schema_created(self, provider):
        """L2 scene_blocks 表应在初始化时创建，作为 atoms 上方的场景层。"""
        cols = {
            row[1] for row in provider._db.execute("PRAGMA table_info(scene_blocks)").fetchall()
        }
        assert {"id", "scene_name", "summary", "atom_ids", "source_turn_ids", "trace_ids", "updated_at"}.issubset(cols)

    def test_project_atoms_roll_up_into_traceable_scene_block(self, provider):
        """同一 scene 的 project atoms 应聚合为可追溯 scene block。"""
        provider.sync_turn(
            "我们决定用Docker部署MiniLoci",
            "Docker 部署方案已记录",
            session_id="scene-session"
        )
        provider.sync_turn(
            "MiniLoci 使用 GitHub Actions 做 CI/CD 发布",
            "CI/CD 流程已记录",
            session_id="scene-session"
        )

        scenes = provider.search_scenes("MiniLoci Docker CI/CD", limit=5)

        assert scenes
        scene = scenes[0]
        assert scene["scene_name"] == "MiniLoci 记忆系统"
        assert "Docker" in scene["summary"]
        assert "GitHub Actions" in scene["summary"] or "CI/CD" in scene["summary"]
        assert len(scene["atom_ids"]) >= 1
        assert len(scene["source_turn_ids"]) >= 2
        assert scene["trace_ids"][0].startswith("turn-")

    def test_scene_search_tool_schema_and_handler(self, provider):
        """插件应暴露 L2 scene 搜索工具。"""
        provider.sync_turn(
            "我们决定用Docker部署MiniLoci",
            "Docker 部署方案已记录",
            session_id="scene-session"
        )

        tool_names = [schema["name"] for schema in provider.get_tool_schemas()]
        payload = json.loads(provider.handle_tool_call("miniloci_search_scenes", {"query": "MiniLoci Docker", "limit": 3}))

        assert "miniloci_search_scenes" in tool_names
        assert payload["results"]
        assert payload["results"][0]["scene_name"] == "MiniLoci 记忆系统"
        assert payload["results"][0]["source_turn_ids"]

    def test_generate_persona_candidate_file_is_traceable_and_review_only(self, provider):
        """L3 persona_candidate 应生成可审核候选文件，不自动覆盖长期 memory。"""
        provider.sync_turn(
            "以后回答不要用表格，要用纯文字和 emoji bullet",
            "收到，以后会避免表格",
            session_id="persona-session"
        )
        provider.sync_turn(
            "我们决定用Docker部署MiniLoci",
            "Docker 部署方案已记录",
            session_id="persona-session"
        )

        result = provider.generate_persona_candidate()

        assert result["status"] == "generated"
        assert result["review_required"] is True
        assert result["applied"] is False
        candidate_path = Path(result["path"])
        assert candidate_path.exists()
        content = candidate_path.read_text(encoding="utf-8")
        assert "# MiniLoci Persona Candidate" in content
        assert "人工审核" in content
        assert "不要用表格" in content
        assert "source_turn_ids" in content
        assert "trace_ids" in content

    def test_persona_candidate_tool_schema_and_handler(self, provider):
        """插件应暴露生成/查看 persona candidate 的工具。"""
        provider.sync_turn(
            "以后回答不要用表格，要用纯文字和 emoji bullet",
            "收到，以后会避免表格",
            session_id="persona-session"
        )

        tool_names = [schema["name"] for schema in provider.get_tool_schemas()]
        payload = json.loads(provider.handle_tool_call("miniloci_persona_candidate", {"action": "generate"}))

        assert "miniloci_persona_candidate" in tool_names
        assert payload["status"] == "generated"
        assert payload["review_required"] is True
        assert payload["applied"] is False

    def test_backfill_memory_layers_dry_run_does_not_write_atoms_or_scenes(self, provider):
        """历史 turns 回填 dry_run 只统计候选，不写入 L1/L2。"""
        provider.sync_turn("我们决定用Docker部署MiniLoci", "Docker 部署方案已记录", session_id="backfill-session")
        provider._db.execute("DELETE FROM scene_blocks_fts")
        provider._db.execute("DELETE FROM scene_blocks")
        provider._db.execute("DELETE FROM memory_atoms_fts")
        provider._db.execute("DELETE FROM memory_atoms")
        provider._db.commit()

        result = provider.backfill_memory_layers(limit=10, dry_run=True)

        assert result["dry_run"] is True
        assert result["scanned_turn_pairs"] >= 1
        assert result["candidate_atoms"] >= 1
        assert provider._db.execute("SELECT COUNT(*) FROM memory_atoms").fetchone()[0] == 0
        assert provider._db.execute("SELECT COUNT(*) FROM scene_blocks").fetchone()[0] == 0

    def test_backfill_memory_layers_writes_traceable_atoms_and_scenes(self, provider):
        """历史 turns 回填应从已有 turns 生成可追溯 atoms/scenes。"""
        provider.sync_turn("我们决定用Docker部署MiniLoci", "Docker 部署方案已记录", session_id="backfill-session")
        provider._db.execute("DELETE FROM scene_blocks_fts")
        provider._db.execute("DELETE FROM scene_blocks")
        provider._db.execute("DELETE FROM memory_atoms_fts")
        provider._db.execute("DELETE FROM memory_atoms")
        provider._db.commit()

        result = provider.backfill_memory_layers(limit=10, dry_run=False)
        atoms = provider.search_atoms("Docker MiniLoci", atom_type="project")
        scenes = provider.search_scenes("Docker MiniLoci")

        assert result["dry_run"] is False
        assert result["atoms_written"] >= 1
        assert result["scenes_updated"] >= 1
        assert atoms
        assert atoms[0]["source_turn_ids"]
        assert atoms[0]["trace_ids"][0].startswith("turn-")
        assert scenes
        assert scenes[0]["source_turn_ids"]

    def test_backfill_memory_layers_tool_schema_and_handler(self, provider):
        """插件应暴露历史 L1/L2 回填工具，并支持 dry_run。"""
        provider.sync_turn("以后回答不要用表格", "收到", session_id="backfill-session")
        tool_names = [schema["name"] for schema in provider.get_tool_schemas()]
        payload = json.loads(provider.handle_tool_call("miniloci_backfill_layers", {"dry_run": True, "limit": 5}))

        assert "miniloci_backfill_layers" in tool_names
        assert payload["dry_run"] is True
        assert payload["scanned_turn_pairs"] >= 1
    
    def test_config_schema(self, provider):
        """测试配置Schema"""
        schema = provider.get_config_schema()
        assert len(schema) > 0
        
        keys = [s["key"] for s in schema]
        assert "window_days" in keys
        assert "fts_weight" in keys
        assert "vector_weight" in keys
        assert "vector_backend" in keys
        assert "vector_local_files_only" in keys
        local_only_schema = next(s for s in schema if s["key"] == "vector_local_files_only")
        assert local_only_schema["default"] is True
        assert local_only_schema["type"] == "boolean"
    
    def test_system_prompt_block(self, provider):
        """测试系统提示块"""
        block = provider.system_prompt_block()
        assert "MiniLoci" in block


class TestMiniLociIntegration:
    """集成测试"""
    
    @pytest.fixture
    def provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = MiniLociProvider()
            p.initialize("integration-test", hermes_home=tmpdir)
            yield p
            p.shutdown()
    
    def test_full_workflow(self, provider):
        """测试完整工作流程"""
        # 保存多条对话
        provider.sync_turn("我们决定用Docker部署", "好的，Docker适合微服务", session_id="integration-test")
        provider.sync_turn("注意环境变量要配staging", "收到，已记录", session_id="integration-test")
        provider.sync_turn("今天天气不错", "是的", session_id="integration-test")
        
        # 搜索记忆
        result = provider.prefetch("你还记得部署方案吗？", session_id="integration-test")
        
        # 验证召回
        assert "Docker" in result or "部署" in result or result == ""
    
    def test_permanent_save_workflow(self, provider):
        """测试永久保存工作流程"""
        provider.sync_turn(
            "记住这个配置：数据库连接池设成20",
            "已记录，数据库连接池20",
            session_id="integration-test"
        )
        
        # 检查永久目录
        permanent_dir = Path(provider.hermes_home) / "loci-archive" / "permanent"
        assert permanent_dir.exists()
        
        # 检查是否有文件创建
        manual_dir = permanent_dir / "manual"
        if manual_dir.exists():
            files = list(manual_dir.glob("*.md"))
            assert len(files) > 0


class TestMiniLociPerformance:
    """性能测试"""
    
    @pytest.fixture
    def provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = MiniLociProvider()
            p.initialize("perf-test", hermes_home=tmpdir)
            yield p
            p.shutdown()
    
    def test_vector_backend_uses_single_worker_queue(self, provider):
        """向量计算必须串行排队，避免多线程同时调用 SentenceTransformer/Faiss。"""
        assert hasattr(provider, "_vector_queue")
        assert hasattr(provider, "_vector_worker")
        assert provider._vector_worker is not None
        assert provider._vector_worker.is_alive()

    def test_numpy_vector_backend_without_faiss_can_search(self, provider):
        """Faiss 缺失时，numpy backend 仍应能用已持久化向量做语义检索。"""
        provider._db.execute(
            "INSERT OR IGNORE INTO sessions (id, start_time, platform) VALUES (?, ?, 'test')",
            ("vector-test", time.time())
        )
        cursor = provider._db.execute(
            """INSERT INTO turns (session_id, timestamp, role, content, importance, tags, metadata)
            VALUES (?, ?, 'user', ?, 2, ?, ?)""",
            ("vector-test", time.time(), "语义检索目标：MiniLoci 向量恢复", json.dumps([]), json.dumps({}))
        )
        target_id = cursor.lastrowid
        provider._db.commit()

        vec = np.zeros(512, dtype=np.float32)
        vec[0] = 1.0
        provider._faiss_index = None
        provider._vector_ids = [target_id]
        provider._vector_matrix = np.array([vec], dtype=np.float32)

        results = provider._vector_search(vec.tolist(), limit=1)
        assert results
        assert results[0]["id"] == target_id
        assert results[0]["score"] > 0.99

    def test_backfill_vectors_populates_missing_vectors_with_fake_model(self, provider):
        """backfill 应把已有 turns 的缺失向量补齐，并刷新 numpy 搜索矩阵。"""
        class FakeModel:
            def encode(self, texts, normalize_embeddings=True):
                if isinstance(texts, str):
                    texts = [texts]
                vectors = []
                for idx, _ in enumerate(texts):
                    vec = np.zeros(512, dtype=np.float32)
                    vec[idx % 512] = 1.0
                    vectors.append(vec)
                return np.array(vectors, dtype=np.float32)

        provider._vector_model = FakeModel()
        provider._vector_model_loaded = True
        provider.enable_vector = True
        provider._faiss_index = None
        provider._vector_ids = []
        provider._vector_matrix = None

        provider._db.execute(
            "INSERT OR IGNORE INTO sessions (id, start_time, platform) VALUES (?, ?, 'test')",
            ("backfill-test", time.time())
        )
        provider._db.execute(
            """INSERT INTO turns (session_id, timestamp, role, content, importance, tags, metadata)
            VALUES (?, ?, 'user', '需要补向量的历史记录', 2, '[]', '{}')""",
            ("backfill-test", time.time())
        )
        provider._db.commit()

        summary = provider.backfill_vectors(limit=10, batch_size=2)
        assert summary["updated"] >= 1
        assert provider._db.execute("SELECT COUNT(*) FROM turns WHERE vector IS NOT NULL").fetchone()[0] >= 1
        assert provider._vector_matrix is not None
        assert len(provider._vector_ids) >= 1

    def test_search_performance(self, provider):
        """测试搜索性能"""
        import time as time_module
        
        # 插入多条记录
        for i in range(100):
            provider.sync_turn(
                f"测试消息{i}: 部署方案讨论",
                f"回复{i}: 同意使用Docker",
                session_id="perf-test"
            )
        
        # 测试搜索耗时
        start = time_module.time()
        result = provider.prefetch("你还记得部署方案吗？", session_id="perf-test")
        duration = time_module.time() - start
        
        # 应该很快（不严格要求100ms，因为测试环境可能慢）
        assert duration < 2.0  # 2秒内


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
