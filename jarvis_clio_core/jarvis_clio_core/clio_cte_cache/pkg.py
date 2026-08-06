"""
CTE node-local cache chimod (clio_cte_cache).

Top of the interposition chain. Keeps a node-local UNTRANSFORMED copy of
each blob in the core's REPLICA_CACHE slot, which the SHM zero-IPC fast
path and raw task reads serve directly.

Write-through, and synchronous about the part that matters: PutBlob
forwards the authoritative write down ``next_pool_id`` FIRST and only
acks once that succeeds, then writes the raw cache copy best-effort. So a
cache failure degrades reads and never loses an acked write, and the
replica is never stale. GetBlob serves from the replica when it covers
the request, otherwise forwards down and re-populates.

Because it sits ABOVE the compressor and replication links, its copy is
the untransformed bytes -- a read hit costs no decompression.
"""
from jarvis_clio_core.cte_interposer import CteInterposerService


class ClioCteCache(CteInterposerService):
    mod_name = 'clio_cte_cache'
    # 563.0 / next 561.0: the ids the coherence integration test uses for
    # the cache -> replication -> core stack. Kept identical so a jarvis
    # deployment and that test describe the same topology.
    default_pool_id = 563.0
    default_next_pool_id = 561.0

    def _policy_menu(self):
        return [
            {
                'name': 'min_score',
                'msg': ('Minimum blob score to admit into the node-local '
                        'cache. Blobs the DPE scored below this forward '
                        'straight down without being cached'),
                'type': float,
                'default': 0.5,
            },
        ]

    def _policy_entry(self):
        return {'min_score': float(self.config.get('min_score', 0.5))}
