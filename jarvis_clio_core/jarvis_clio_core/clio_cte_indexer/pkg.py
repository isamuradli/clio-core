"""
CTE indexer chimod (clio_cte_indexer).

Interposes to observe blob traffic and maintain a searchable index, which
is what backs SemanticSearch/TagQuery/BlobQuery. Indexing itself happens
on a periodic sweep (``index_sweep_period_ms``) rather than inline, so
the foreground put path pays interposition but not index construction.

``index_log_path`` enables the module WAL: with it set, a restarted
runtime restores the index from the log instead of rescanning storage.
Left empty the index is rebuilt by scanning, which on a large pool is the
difference between a fast restart and a very slow one.
"""
from jarvis_clio_core.cte_interposer import CteInterposerService


class ClioCteIndexer(CteInterposerService):
    mod_name = 'clio_cte_indexer'
    # 564.0 is the id the indexer_restart and semantic_bench paths use.
    # It is the TOP of the full chain, so this is the pool a client's
    # CLIO_CTE_POOL should name when the indexer is deployed.
    default_pool_id = 564.0
    default_next_pool_id = 563.0

    def _policy_menu(self):
        return [
            {
                'name': 'index_sweep_period_ms',
                'msg': ('Period of the background index sweep. This is '
                        'foreground-competing work: shorter keeps the '
                        'index fresher and costs more write bandwidth'),
                'type': int,
                'default': 100,
            },
            {
                'name': 'tag_re',
                'msg': 'Regex selecting which tags to index',
                'type': str,
                'default': '.*',
            },
            {
                'name': 'blob_re',
                'msg': 'Regex selecting which blobs to index',
                'type': str,
                'default': '.*',
            },
            {
                'name': 'index_log_path',
                'msg': ('Path to the index WAL. Empty disables it, and a '
                        'restart then rebuilds the index by rescanning '
                        'storage. Must be node-local and writable'),
                'type': str,
                'default': '',
            },
        ]

    def _policy_entry(self):
        entry = {
            'index_sweep_period_ms': int(
                self.config.get('index_sweep_period_ms', 100)),
            'tag_re': self.config.get('tag_re', '.*'),
            'blob_re': self.config.get('blob_re', '.*'),
        }
        # Emit only when set -- the runtime short-circuits on an empty
        # path, and writing an explicit "" would be a behaviour change.
        index_log_path = self.config.get('index_log_path', '')
        if index_log_path:
            entry['index_log_path'] = index_log_path
        return entry
