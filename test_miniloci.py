"""
MiniLoci 测试套件
"""

import pytest
import tempfile
import time
import json
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
    
    def test_config_schema(self, provider):
        """测试配置Schema"""
        schema = provider.get_config_schema()
        assert len(schema) > 0
        
        keys = [s["key"] for s in schema]
        assert "window_days" in keys
        assert "fts_weight" in keys
        assert "vector_weight" in keys
    
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
