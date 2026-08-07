"""
IOWarp FUSE adapter.

Mounts the CTE-backed virtual filesystem at a configured path by launching
the `clio_cte_fuse` binary (built with CLIO_CTE_ENABLE_FUSE_ADAPTER=ON).
Supports both bare-metal and container deployment modes.
"""
from jarvis_cd.core.pkg import Service
from jarvis_cd.shell import Exec, PsshExecInfo
from jarvis_cd.shell.process import Kill
from jarvis_cd.util.container_utils import container_kwargs
import time


class ClioCteLibfuse(Service):
    """IOWarp FUSE adapter — mounts the CTE filesystem at a configured path."""

    def _init(self):
        self.binary = 'clio_cte_fuse'

    def _configure_menu(self):
        return [
            {
                'name': 'mountpoint',
                'msg': 'Absolute path to mount the CTE filesystem.',
                'type': str,
                'default': '${HOME}/clio_cte',
            },
            {
                'name': 'log_level',
                'msg': 'CTP log level for the FUSE daemon',
                'type': str,
                'choices': ['debug', 'info', 'warning', 'error'],
                'default': 'info',
            },
            {
                'name': 'extra_fuse_args',
                'msg': 'Extra CLI flags forwarded to clio_cte_fuse / libfuse.',
                'type': str,
                'default': '-f',
            },
            {
                'name': 'cte_pool',
                'msg': ('Pool the FUSE client binds its CTE client to, as '
                        '"<major>.<minor>". Empty binds to the CTE core '
                        '(512.0). Set this to the TOP of an interposer '
                        'chain -- e.g. 564.0 for indexer -> cache -> '
                        'replication -> core -- otherwise the chain is '
                        'bypassed and no I/O traverses it.'),
                'type': str,
                'default': '',
            },
            # ---- small-I/O amortization (issue #933) --------------------
            # Every default below is the adapter's historical behaviour, so
            # an unconfigured mount is unchanged. They matter because that
            # behaviour costs one kernel->userspace upcall per operation,
            # which is the whole small-I/O deficit: at 4 KiB / 1 rank CTE
            # measured 15.0 MiB/s (3,840 ops/s) against JuiceFS's 540 MiB/s
            # (138,300 ops/s) on the same node. JuiceFS is also FUSE; it
            # ships 1.0s attr/entry caches and a client-side write buffer,
            # so the kernel absorbs most ops. At 1 MiB the two invert and
            # CTE wins, which is what identifies this as per-op count
            # rather than bandwidth or the CTE data path.
            {
                'name': 'attr_timeout',
                'msg': ('Seconds the kernel may cache file attributes. 0 '
                        'sends every getattr to the chimod. This also gates '
                        'READ throughput, not just metadata: at 0 the kernel '
                        'revalidates constantly and the page cache never '
                        'survives. JuiceFS uses 1.0. Costs a staleness '
                        'window on st_nlink for hard links.'),
                'type': float,
                'default': 0.0,
            },
            {
                'name': 'entry_timeout',
                'msg': ('Seconds the kernel may cache directory entries '
                        '(name -> inode). JuiceFS uses 1.0.'),
                'type': float,
                'default': 0.0,
            },
            {
                'name': 'negative_timeout',
                'msg': ('Seconds the kernel may cache failed lookups. Helps '
                        'create-heavy paths, which probe before creating.'),
                'type': float,
                'default': 0.0,
            },
            {
                'name': 'writeback',
                'msg': ('Enable FUSE_CAP_WRITEBACK_CACHE so the kernel '
                        'buffers and coalesces writes instead of sending '
                        'each one through synchronously. The kernel then '
                        'owns file size, so a partial-page write is padded '
                        'to page granularity and readers observe the '
                        "kernel's size rather than the chimod's."),
                'type': bool,
                'default': False,
            },
            {
                'name': 'max_write_kib',
                'msg': ('Max bytes per write upcall, in KiB. 0 keeps the '
                        'libfuse default. The kernel clamps an over-large '
                        'request rather than rejecting it.'),
                'type': int,
                'default': 0,
            },
            {
                'name': 'max_background',
                'msg': ('Max async requests in flight before the kernel '
                        'throttles the queue. 0 keeps the default.'),
                'type': int,
                'default': 0,
            },
            {
                'name': 'max_readahead_kib',
                'msg': 'Kernel readahead window in KiB. 0 keeps the default.',
                'type': int,
                'default': 0,
            },
            {
                'name': 'sleep',
                'msg': 'Seconds to wait after launch for the FUSE handshake.',
                'type': int,
                'default': 2,
            },
        ]

    def _configure(self, **kwargs):
        super()._configure(**kwargs)
        self.setenv('CTP_LOG_LEVEL', self.config['log_level'])
        self.setenv('CLIO_WITH_RUNTIME', '0')

        # Bind the adapter's CTE client to a specific pool (issue #886).
        # Interposers forward every method they do not override, so a
        # client pointed at core still WORKS with a chain deployed -- it
        # just enters below every interposer, and the cache/replication/
        # indexer pools sit there doing nothing while their sweeps run.
        # There is no error and no log line for that; a benchmark simply
        # reports the chain as free. Hence: set it explicitly, or leave
        # empty and inherit the 512.0 default deliberately.
        cte_pool = str(self.config.get('cte_pool', '') or '').strip()
        if cte_pool:
            self.setenv('CLIO_CTE_POOL', cte_pool)

        # Small-I/O tuning (issue #933). Emitted unconditionally, including
        # the zeros: the adapter echoes its resolved tuning to stderr at
        # mount, and a benchmark that cannot tell a tuned mount from an
        # untuned one is how the deficit went unexplained in the first place.
        # Sent as env because libfuse's argv parser owns "-o" and an
        # unrecognized key there aborts the mount.
        self.setenv('CLIO_CTE_FUSE_ATTR_TIMEOUT',
                    str(float(self.config.get('attr_timeout', 0.0))))
        self.setenv('CLIO_CTE_FUSE_ENTRY_TIMEOUT',
                    str(float(self.config.get('entry_timeout', 0.0))))
        self.setenv('CLIO_CTE_FUSE_NEGATIVE_TIMEOUT',
                    str(float(self.config.get('negative_timeout', 0.0))))
        self.setenv('CLIO_CTE_FUSE_WRITEBACK',
                    '1' if self.config.get('writeback', False) else '0')
        self.setenv('CLIO_CTE_FUSE_MAX_WRITE_KIB',
                    str(int(self.config.get('max_write_kib', 0))))
        self.setenv('CLIO_CTE_FUSE_MAX_BACKGROUND',
                    str(int(self.config.get('max_background', 0))))
        self.setenv('CLIO_CTE_FUSE_MAX_READAHEAD_KIB',
                    str(int(self.config.get('max_readahead_kib', 0))))

    def start(self):
        mp = self.config['mountpoint']
        extra = self.config.get('extra_fuse_args', '').strip()

        # Hack: idempotent tear-down before bring-up so a prior
        # scancel-killed run that left a dangling FUSE mount doesn't
        # poison this run's Mkdir/clio_cte_fuse with "Transport endpoint
        # is not connected". stop() is already a fusermount3 -u + Kill
        # of the binary; calling it here just makes start() idempotent.
        self.stop()

        # container_kwargs routes the mkdir into the run's apptainer instance
        # (jarvis only wraps when exec_info.container is set). REQUIRED here:
        # under tmp_bind_root the in-container /tmp is a different directory
        # from the host /tmp, so the mountpoint must be created in-container.
        Exec(f'mkdir -p {mp}',
             PsshExecInfo(env=self.mod_env, hostfile=self.hostfile,
                          **container_kwargs(self))).run()

        fuse_cmd = f'{self.binary} {mp} {extra}'.strip()
        self.log(f"Mounting IOWarp CTE FUSE at {mp}: {fuse_cmd}")
        # Shell-background the daemon inside a synchronous wrapped Exec: the
        # `nohup ... &` detaches it from the wrap's `bash -c` shell, and it
        # parents into the apptainer instance, whose mount/shm namespaces it
        # must share with the runtime and the fio that writes through the
        # mount. Daemon output goes to a per-host log on the auto-mounted
        # shared_dir (not /dev/null — a masked mount failure here previously
        # surfaced only as downstream ENOSPC).
        fuse_log = f'{self.shared_dir}/cte_fuse.$(hostname).log'
        bg_cmd = f'nohup {fuse_cmd} </dev/null >{fuse_log} 2>&1 &'
        Exec(bg_cmd, PsshExecInfo(
            env=self.mod_env, hostfile=self.hostfile,
            **container_kwargs(self))).run()
        time.sleep(self.config.get('sleep', 2))

    def stop(self):
        mp = self.config['mountpoint']
        self.log(f"Unmounting {mp}")
        # Both teardown steps must run inside the instance: the FUSE mount
        # exists only in the instance's mount namespace (host-side
        # fusermount3 sees "not found in /etc/mtab"), and the wrapped Kill
        # scopes pkill to this run's PID namespace.
        Exec(f'fusermount3 -u {mp}', PsshExecInfo(
            env=self.mod_env, hostfile=self.hostfile,
            **container_kwargs(self))).run()
        Kill(self.binary, PsshExecInfo(
            env=self.mod_env, hostfile=self.hostfile,
            **container_kwargs(self))).run()

    def clean(self):
        self.stop()
