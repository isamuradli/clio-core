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
