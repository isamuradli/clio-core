"""
CTE replication chimod (clio_cte_replication).

Sits between the cache and the core. Forwards the authoritative write
down ``next_pool_id`` and acks on that; the durable extra copies are made
asynchronously by a periodic sweep (``replicate_period_ms``), so steady
-state put latency is not charged the replica writes -- but sustained
write throughput is, because the sweep competes for the same targets.

``num_replicas`` is the number of ADDITIONAL copies beyond the
authoritative one. On a single node with one RAM target there is nowhere
else to put them, so 1 is the honest single-node setting; raise it only
when the pipeline has more than one node or more than one target.
"""
from jarvis_clio_core.cte_interposer import CteInterposerService


class ClioCteReplication(CteInterposerService):
    mod_name = 'clio_cte_replication'
    # 561.0 / next 512.0 -- the coherence test's ids (see clio_cte_cache).
    default_pool_id = 561.0
    default_next_pool_id = 512.0

    def _policy_menu(self):
        return [
            {
                'name': 'num_replicas',
                'msg': ('Number of ADDITIONAL durable copies beyond the '
                        'authoritative write'),
                'type': int,
                'default': 1,
            },
            {
                'name': 'cache_score',
                'msg': 'DPE score assigned to the cached/primary copy',
                'type': float,
                'default': 1.0,
            },
            {
                'name': 'replica_score',
                'msg': ('DPE score assigned to replica copies. Lower than '
                        'cache_score so replicas land on slower tiers and '
                        'do not evict primaries'),
                'type': float,
                'default': 0.2,
            },
            {
                'name': 'replicate_period_ms',
                'msg': ('Period of the async replication sweep. Lower '
                        'closes the durability window sooner and steals '
                        'more bandwidth from the foreground path'),
                'type': int,
                'default': 50,
            },
        ]

    def _policy_entry(self):
        return {
            'num_replicas': int(self.config.get('num_replicas', 1)),
            'cache_score': float(self.config.get('cache_score', 1.0)),
            'replica_score': float(self.config.get('replica_score', 0.2)),
            'replicate_period_ms': int(
                self.config.get('replicate_period_ms', 50)),
        }
