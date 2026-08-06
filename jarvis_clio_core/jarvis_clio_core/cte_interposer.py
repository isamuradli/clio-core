"""
Shared base for the CTE interposer-chimod jarvis packages.

Every chimod deriving from ``clio::cte::core::CoreInterposer`` (issue #886)
is deployed the same way: write a one-entry ``clio_run compose`` YAML
naming the module, the pool it occupies, and the ``next_pool_id`` it
forwards to, then run ``clio_run compose start`` once across the hostfile.
Only the module name and its policy knobs differ, so the plumbing lives
here and each package supplies the two.

**The chain.** An interposer speaks the core's method ids and task
structs, overrides the data verbs (PutBlob/GetBlob/GetBlobSize/
MultiPutBlob), and forwards everything else down ``next_pool_id``
verbatim. A ``clio::cte::core::Client`` pointed at any link therefore
keeps working unchanged -- which is what lets a chain be assembled purely
from configuration. The canonical single-node stack, top to bottom, and
the pool ids the integration tests use:

    indexer 564.0 -> cache 563.0 -> replication 561.0 -> core 512.0

Order matters and is not arbitrary: cache sits above replication so a
read served from the node-local raw copy never pays the replication
layer, and both sit above core so core stays the single source of truth.

**Clients bind to the TOP, not to core.** A process talks to the chain by
setting ``CLIO_CTE_POOL=<major>.<minor>`` to the topmost link before its
CTE client initializes; unset, it binds to core at 512.0 and every
interposer in the chain is bypassed silently -- the pools exist, the
sweeps run, and no I/O traverses them. That failure is invisible in a
benchmark: it looks like the stack simply had no cost. Deploying a chain
therefore means configuring the *client* too (see clio_cte_libfuse's
``cte_pool``), not only composing the pools.
"""
import os

import yaml

from jarvis_cd.core.pkg import Service
from jarvis_cd.shell import Exec, PsshExecInfo


class CteInterposerService(Service):
    """Base for a single interposer chimod deployed via clio_run compose.

    Subclasses set :attr:`mod_name` / :attr:`default_pool_id` /
    :attr:`default_next_pool_id`, and override
    :meth:`_policy_menu` + :meth:`_policy_entry` for the knobs that are
    theirs.
    """

    #: chimod name as registered with the runtime (e.g. clio_cte_cache)
    mod_name = None
    #: pool this module occupies by default
    default_pool_id = None
    #: pool it forwards to by default
    default_next_pool_id = 512.0

    # ------------------------------------------------------------------
    # subclass hooks
    # ------------------------------------------------------------------

    def _policy_menu(self):
        """Module-specific config entries, appended to the common ones."""
        return []

    def _policy_entry(self):
        """Module-specific keys merged into the compose entry."""
        return {}

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------

    def _compose_basename(self):
        return f'{self.mod_name}_compose.yaml'

    def _init(self):
        self.compose_config_path = os.path.join(
            self.shared_dir, self._compose_basename())

    def _configure_menu(self):
        return [
            {
                'name': 'pool_name',
                'msg': f'Name of the {self.mod_name} pool',
                'type': str,
                'default': self.mod_name,
            },
            {
                'name': 'pool_id',
                'msg': (f'Pool ID for {self.mod_name}. Clients reach the '
                        'chain by pointing CLIO_CTE_POOL at its TOP link, '
                        'so this only needs to be unique and to match '
                        "whatever sits above it in the chain"),
                'type': float,
                'default': self.default_pool_id,
            },
            {
                'name': 'next_pool_id',
                'msg': ('Pool ID of the next link down the chain '
                        '(ultimately the clio_cte_core pool)'),
                'type': float,
                'default': self.default_next_pool_id,
            },
            {
                'name': 'pool_query',
                'msg': 'Pool query type (local or dynamic)',
                'type': str,
                'choices': ['local', 'dynamic'],
                'default': 'local',
            },
        ] + self._policy_menu()

    # Shares clio_runtime's image; nothing separate to build.
    def _build_deploy_phase(self) -> str:
        return None

    @staticmethod
    def _format_pool_id(pool_id) -> str:
        """clio_run compose wants "<major>.<minor>" as a STRING.

        A YAML-native 563.0 round-trips as a float and the compose parser
        reads pool ids textually (it splits on the dot), so coerce here.
        Integer-valued floats become "563.0" rather than "563".
        """
        if isinstance(pool_id, str):
            return pool_id
        as_float = float(pool_id)
        if as_float.is_integer():
            return f"{int(as_float)}.0"
        return repr(as_float)

    def _configure(self, **kwargs):
        super()._configure(**kwargs)

        self.compose_config_path = os.path.join(
            self.shared_dir, self._compose_basename())

        entry = {
            'mod_name': self.mod_name,
            'pool_name': self.config.get('pool_name', self.mod_name),
            'pool_query': self.config.get('pool_query', 'local'),
            'pool_id': self._format_pool_id(
                self.config.get('pool_id', self.default_pool_id)),
            'next_pool_id': self._format_pool_id(
                self.config.get('next_pool_id', self.default_next_pool_id)),
        }
        entry.update(self._policy_entry())

        with open(self.compose_config_path, 'w') as f:
            f.write(f'# {self.mod_name} clio_run-compose configuration\n\n')
            yaml.dump({'compose': [entry]}, f,
                      default_flow_style=False, indent=2)

        self.log(f"{self.mod_name}: compose written to "
                 f"{self.compose_config_path} "
                 f"(pool {entry['pool_id']} -> next {entry['next_pool_id']})")

    def start(self):
        self.log(f"Starting {self.mod_name} via clio_run compose...")

        if not os.path.exists(self.compose_config_path):
            self.log(f"Error: Compose config not found: "
                     f"{self.compose_config_path}")
            return False

        # Single-shot compose, deliberately NOT wrapped in a retry loop.
        # The jarvis-cd SSH layer prepends ``KEY=VAL`` env vars to the
        # command string, and bash only attaches those to a *simple*
        # command -- a wrapping `for ... do ... done` strips the env
        # (notably CLIO_SERVER_CONF), whereupon clio_run falls back to
        # ~/.clio/clio.yaml and picks up unrelated compose entries
        # squatting on our pool id.
        Exec(f'clio_run compose start {self.compose_config_path}',
             PsshExecInfo(
                 env=self.mod_env,
                 hostfile=self.jarvis.hostfile,
                 container=self._container_engine,
                 container_image=self.deploy_image_name(),
                 private_dir=self.private_dir,
                 bind_mounts=self.container_mounts,
             )).run()

        self.log(f"{self.mod_name} started successfully")
        return True

    # The pool dies with the runtime daemon; there is no per-module stop.
    def stop(self):
        pass

    def kill(self):
        pass

    def clean(self):
        if (self.compose_config_path
                and os.path.exists(self.compose_config_path)):
            os.remove(self.compose_config_path)
