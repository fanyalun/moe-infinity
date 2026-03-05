import torch

class HybridStats:
    """
    用于统计 Cache 命中率的全局静态类。
    """
    # 全局统计
    total_hits = 0
    total_queries = 0
    
    # 模态细分统计
    vision_hits = 0
    vision_total = 0
    
    text_hits = 0
    text_total = 0
    
    # 记录层级信息 (可选，用于 debug)
    layer_stats = {}

    @classmethod
    def reset(cls):
        cls.total_hits = 0
        cls.total_queries = 0
        cls.vision_hits = 0
        cls.vision_total = 0
        cls.text_hits = 0
        cls.text_total = 0
        cls.layer_stats = {}

    @classmethod
    def update(cls, layer_id, hits, queries, vision_stats=None, text_stats=None):
        """
        更新统计信息
        :param layer_id: 当前层 ID
        :param hits: 本次 batch 的总命中数 (TopK - KeepK 部分)
        :param queries: 本次 batch 的总查询数 (TopK - KeepK 部分)
        :param vision_stats: (vision_hits, vision_queries) tuple
        :param text_stats: (text_hits, text_queries) tuple
        """
        # 1. 更新全局
        cls.total_hits += hits
        cls.total_queries += queries
        
        # 2. 更新 Vision
        if vision_stats is not None:
            v_hits, v_total = vision_stats
            cls.vision_hits += v_hits
            cls.vision_total += v_total
            
        # 3. 更新 Text
        if text_stats is not None:
            t_hits, t_total = text_stats
            cls.text_hits += t_hits
            cls.text_total += t_total

        # 4. (可选) 更新层级统计
        if layer_id not in cls.layer_stats:
            cls.layer_stats[layer_id] = {"hits": 0, "total": 0}
        cls.layer_stats[layer_id]["hits"] += hits
        cls.layer_stats[layer_id]["total"] += queries

    @classmethod
    def get_summary(cls):
        """
        获取格式化的统计字符串 (用于打印)
        """
        # Global Rate
        global_rate = (cls.total_hits / cls.total_queries * 100) if cls.total_queries > 0 else 0.0
        
        # Vision Rate
        vision_rate = (cls.vision_hits / cls.vision_total * 100) if cls.vision_total > 0 else 0.0
        
        # Text Rate
        text_rate = (cls.text_hits / cls.text_total * 100) if cls.text_total > 0 else 0.0
        
        return (
            f"[Cache Stats] "
            f"Global: {global_rate:.2f}% ({cls.total_hits}/{cls.total_queries}) | "
            f"Vision: {vision_rate:.2f}% | "
            f"Text: {text_rate:.2f}%"
        )

    @classmethod
    def print_summary(cls):
        print(cls.get_summary())

    # === [Fix] 新增此方法 ===
    @classmethod
    def get_metrics_dict(cls):
        """
        返回字典格式的统计信息 (用于 JSON 保存)
        """
        global_rate = (cls.total_hits / cls.total_queries) if cls.total_queries > 0 else 0.0
        vision_rate = (cls.vision_hits / cls.vision_total) if cls.vision_total > 0 else 0.0
        text_rate = (cls.text_hits / cls.text_total) if cls.text_total > 0 else 0.0

        return {
            "global": {
                "hits": cls.total_hits,
                "total": cls.total_queries,
                "rate": global_rate
            },
            "vision": {
                "hits": cls.vision_hits,
                "total": cls.vision_total,
                "rate": vision_rate
            },
            "text": {
                "hits": cls.text_hits,
                "total": cls.text_total,
                "rate": text_rate
            },
            "layer_stats": cls.layer_stats
        }