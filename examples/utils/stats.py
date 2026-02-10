import torch

class HybridStats:
    """
    用于统计 Cache 命中率的全局静态类。
    [性能优化版]: 异步记录 + 细分 Decode/Prefill 统计
    """
    # 存储异步记录
    _async_records = []

    @classmethod
    def reset(cls):
        cls._async_records = []

    @classmethod
    def update(cls, layer_id, query_count, 
               online_hits_tensor=None, 
               offline_vision_hits_tensor=None, 
               offline_text_hits_tensor=None,
               is_decode=False):
        """
        [非阻塞更新]
        :param query_count: 本次 Batch 实际参与检索的 slot 总数 (已扣除被 Mask 掉的部分)
        :param is_decode: 是否为 Decode 阶段 (用于计算 Online Rate)
        """
        cls._async_records.append({
            "id": layer_id,
            "queries": query_count, 
            "on_hits": online_hits_tensor,        # 0-dim Tensor
            "off_v_hits": offline_vision_hits_tensor, # 0-dim Tensor
            "off_t_hits": offline_text_hits_tensor,    # 0-dim Tensor
            "is_decode": is_decode
        })

    @classmethod
    def _synchronize_and_aggregate(cls):
        """
        [同步点] 聚合所有统计数据
        """
        stats = {
            "total_queries": 0,    # 有效总查询数 (Prefill + Decode)
            "decode_queries": 0,   # Decode 阶段查询数 (用于 Online Rate)
            "online_hits": 0,
            "offline_vision_hits": 0,
            "offline_text_hits": 0,
            "layer_stats": {}
        }
        
        for record in cls._async_records:
            lid = record["id"]
            q = record["queries"]
            is_dec = record["is_decode"]
            
            # 同步获取数值
            on_h = record["on_hits"].item() if record["on_hits"] is not None else 0
            off_v = record["off_v_hits"].item() if record["off_v_hits"] is not None else 0
            off_t = record["off_t_hits"].item() if record["off_t_hits"] is not None else 0
            
            # 全局累加
            stats["total_queries"] += q
            if is_dec:
                stats["decode_queries"] += q

            stats["online_hits"] += on_h
            stats["offline_vision_hits"] += off_v
            stats["offline_text_hits"] += off_t
            
            # 层级累加
            if lid not in stats["layer_stats"]:
                stats["layer_stats"][lid] = {
                    "queries": 0, "decode_queries": 0, 
                    "on": 0, "off_v": 0, "off_t": 0
                }
            stats["layer_stats"][lid]["queries"] += q
            if is_dec:
                stats["layer_stats"][lid]["decode_queries"] += q
            stats["layer_stats"][lid]["on"] += on_h
            stats["layer_stats"][lid]["off_v"] += off_v
            stats["layer_stats"][lid]["off_t"] += off_t
            
        return stats

    @classmethod
    def get_summary(cls):
        stats = cls._synchronize_and_aggregate()
        
        # 1. Global Rate (全阶段两模态命中率)
        # 分母: Total Valid Queries (已扣除 Prefill 强制重算的 Token)
        # 分子: Online + Offline Hits (如你所要求，包含 Online 命中的部分)
        total_q = stats["total_queries"] if stats["total_queries"] > 0 else 1
        total_hits = stats["online_hits"] + stats["offline_vision_hits"] + stats["offline_text_hits"]
        global_rate = total_hits / total_q * 100
        
        # 2. Online Rate (在线命中率)
        # 分母: 仅 Decode Queries (因为 Prefill 不使用 Online Cache)
        decode_q = stats["decode_queries"] if stats["decode_queries"] > 0 else 1
        on_rate = stats["online_hits"] / decode_q * 100
        
        # 3. Breakdown (辅助参考)
        off_v_rate = stats["offline_vision_hits"] / total_q * 100
        off_t_rate = stats["offline_text_hits"] / total_q * 100
        
        return (
            f"[Cache Stats]\n"
            f"  Global Rate : {global_rate:.2f}% ({total_hits}/{stats['total_queries']}) [Incl. Online]\n"
            f"  Online Rate : {on_rate:.2f}% ({stats['online_hits']}/{stats['decode_queries']}) [Decode Only]\n"
            f"  Offline Vis : {off_v_rate:.2f}%\n"
            f"  Offline Txt : {off_t_rate:.2f}%"
        )

    @classmethod
    def print_summary(cls):
        print(cls.get_summary())

    @classmethod
    def get_metrics_dict(cls):
        stats = cls._synchronize_and_aggregate()
        total_q = stats["total_queries"] if stats["total_queries"] > 0 else 1
        decode_q = stats["decode_queries"] if stats["decode_queries"] > 0 else 1

        return {
            "global": {
                "hits": stats["online_hits"] + stats["offline_vision_hits"] + stats["offline_text_hits"],
                "total": stats["total_queries"],
                "rate": (stats["online_hits"] + stats["offline_vision_hits"] + stats["offline_text_hits"]) / total_q
            },
            "online": {
                "hits": stats["online_hits"],
                "total": stats["decode_queries"],
                "rate": stats["online_hits"] / decode_q
            },
            "breakdown": {
                "online_hits": stats["online_hits"],
                "offline_vision_hits": stats["offline_vision_hits"],
                "offline_text_hits": stats["offline_text_hits"]
            },
            "layer_stats": stats["layer_stats"]
        }