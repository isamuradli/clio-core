"""
clio_cte_reliability_bench as a pipeline package.

The binary talks to the filesystem chimod through CLIO_CFS_CLIENT -- open,
read, write -- so it measures the filesystem layer WITHOUT FUSE and without
interception. That is the middle term the other two benchmarks bracket:

    CTE core client, PutDefer   348.7 MiB/s   11.2 us/op   (clio_cte_bench)
    <this>                          ?             ?
    full FUSE path               19.1 MiB/s  ~150 us/op   (ior over the mount)

It MUST run as a package rather than a post_cmd: post_cmds execute after jarvis
tears the runtime down, so the benchmark cannot connect and exits.
"""
import os

from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo


class ClioCfsBench(Application):
    """Run clio_cte_reliability_bench against a composed runtime + CTE pool."""

    def _init(self):
        self.benchmark_executable = 'clio_cte_reliability_bench'

    def _configure_menu(self):
        return [
            {
                'name': 'mix',
                'msg': ('Operation mix, "<op>=<pct>[,...]". Unlisted ops get '
                        '0, so this is also how you restrict the run'),
                'type': str,
                'default': 'write_4k=50,read_4k=50',
            },
            {
                'name': 'threads',
                'msg': 'Benchmark threads (binary default is 8)',
                'type': int,
                'default': 1,
            },
            {
                'name': 'duration',
                'msg': 'Measured window (seconds)',
                'type': int,
                'default': 20,
            },
            {
                'name': 'warmup',
                'msg': 'Discarded warmup before measuring (seconds)',
                'type': int,
                'default': 5,
            },
            {
                'name': 'small_io',
                'msg': 'Size of the small-I/O operations',
                'type': str,
                'default': '4k',
            },
            {
                'name': 'max_data',
                'msg': 'Cap on dataset bytes (k/m/g suffix)',
                'type': str,
                'default': '1g',
            },
            {
                'name': 'root',
                'msg': 'Filesystem root for the dataset',
                'type': str,
                'default': '${HOME}/cfs_bench_root',
            },
            {
                'name': 'label',
                'msg': 'Label recorded in the CSV',
                'type': str,
                'default': 'cfs_direct',
            },
        ]

    def _configure(self, **kwargs):
        super()._configure(**kwargs)

    def start(self):
        out = os.path.join(self.shared_dir, 'cfs_bench_output.txt')
        csv_path = os.path.join(self.shared_dir, 'cfs_bench.csv')
        root = os.path.expandvars(self.config['root'])
        os.makedirs(root, exist_ok=True)

        cmd = [
            self.benchmark_executable,
            '--threads', str(self.config['threads']),
            '--duration', str(self.config['duration']),
            '--warmup', str(self.config['warmup']),
            '--small-io', str(self.config['small_io']),
            '--max-data', str(self.config['max_data']),
            '--mix', str(self.config['mix']),
            '--root', root,
            '--label', str(self.config['label']),
            '--csv', csv_path,
        ]
        cmd_str = ' '.join(cmd)
        self.log(f"Executing: {cmd_str}")
        Exec(cmd_str, LocalExecInfo(env=self.mod_env,
                                    pipe_stdout=out,
                                    pipe_stderr=out)).run()
        self.log(f"Benchmark output: {out}")
        return True

    def stop(self):
        pass

    def kill(self):
        pass

    def clean(self):
        pass
