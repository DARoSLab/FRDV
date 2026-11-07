"""Config composition for the CLI tools (Hydra's compose API: ``key=value`` / ``group=option``
overrides, config groups, interpolation), without Hydra's app runner."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .common import PACKAGE_DIR  # also registers the ${frdv_root:} resolver

CONF_DIR = PACKAGE_DIR / "conf"


def user_conf_dir() -> Path:
    return Path(os.environ.get("FRDV_CONFIG_DIR", "frdv_conf")).resolve()


def available(group: str) -> list[str]:
    names = set()
    if group == 'subject':
        for root in (CONF_DIR, user_conf_dir()):
            folder = root / 'calibration'
            if folder.is_dir():
                names |= {p.relative_to(folder).with_suffix('').as_posix()
                          for p in folder.glob('*/*/subjects/**/*.yaml')}
        return ['none'] + sorted(names)
    for root in (CONF_DIR, user_conf_dir()):
        d = root / group
        if d.is_dir():
            names |= {p.relative_to(d).with_suffix("").as_posix() for p in d.rglob("*.yaml")}
    return sorted(names)


def compose(overrides: list[str]):
    """Compose the config with Hydra; the result is cached on disk (~0.1 s of Hydra start-up),
    keyed by the overrides and every config file's path, size and mtime."""
    cached = _cache_get(overrides)
    if cached is not None:
        _validate_subject(cached)
        return cached
    cfg = _compose(overrides)
    _validate_subject(cfg)
    _cache_put(overrides, cfg)
    return cfg


def _validate_subject(cfg):
    """Subject profiles declare their dataset and robot; collection names remain arbitrary."""
    for field in ('robot', 'dataset'):
        expected = cfg.calibration.get(field)
        if expected and expected != cfg[field].name:
            raise ValueError(f'calibration requires {field} {expected!r}, got {cfg[field].name!r}')
    if cfg.dataset.scale.mode == 'manual' and cfg.dataset.scale.get('value') is None:
        raise ValueError(f'No manual scaling calibrated for robot {cfg.robot.name!r}, dataset {cfg.dataset.name!r}. '
                         'Use scaling=auto, provide dataset.scale.value=[sx,sy,sz], or save a robot-specific profile.')
    subject = cfg.get("subject") or {}
    if not subject.get("name"):
        return
    for field in ("dataset", "robot"):
        expected = subject.get(field)
        actual = cfg[field].get("name")
        if expected and expected != actual:
            raise ValueError(f"subject '{subject['name']}' requires {field} '{expected}', got '{actual}'; "
                             f"select the matching {field} preset")


def _cache_key(overrides):
    if os.environ.get("FRDV_NO_CONFIG_CACHE", "0") in ("1", "true", "True"):
        return None
    import hashlib
    h = hashlib.sha256(repr((list(overrides), os.environ.get("FRDV_CONFIG_DIR"), str(user_conf_dir()))).encode())
    try:
        for root in (CONF_DIR, user_conf_dir()):
            if not root.is_dir():
                continue
            for dirpath, dirs, files in os.walk(root):
                dirs.sort()
                for f in sorted(files):
                    st = os.stat(os.path.join(dirpath, f))
                    h.update(f"{dirpath}/{f}|{st.st_size}|{st.st_mtime_ns}".encode())
    except OSError:
        return None
    base = os.environ.get("FRDV_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "frdv")
    return Path(base) / "config" / f"{h.hexdigest()[:24]}.yaml"


def _cache_get(overrides):
    path = _cache_key(overrides)
    if path is None or not path.exists():
        return None
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(path)
        OmegaConf.set_struct(cfg, True)
        return cfg
    except Exception:
        return None


def _cache_put(overrides, cfg):
    path = _cache_key(overrides)
    if path is None:
        return
    try:
        import tempfile
        from omegaconf import OmegaConf
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".yaml", dir=path.parent)
        with os.fdopen(fd, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        os.replace(tmp, path)
        old = sorted(path.parent.glob("*.yaml"), key=lambda p: p.stat().st_mtime)[:-64]
        for p in old:          # keep the 64 most recent override sets
            p.unlink(missing_ok=True)
    except Exception:
        pass


def _compose(overrides: list[str]):
    try:
        from hydra.core.config_store import ConfigStore
    except ImportError:
        sys.exit("the frdv command-line tools need Hydra: pip install 'frdv[cli]'  (or uv sync --extra cli)")
    from omegaconf import OmegaConf
    from .profiles import load_profile
    selection = None
    remaining = []
    for override in overrides:
        if override.startswith('subject='):
            selection = override.partition('=')[2]
        else:
            remaining.append(override)
    # Resolve only the identities first; applying the final overrides after the calibration
    # layer lets scaling selections and explicit CLI values retain their normal precedence.
    store = ConfigStore.instance()
    store.store(group='calibration_profile', name='active', node={}, package='_global_')
    routing = [o for o in remaining if o.startswith(('robot=', 'dataset=', 'robot.name=', 'dataset.name='))]
    identity = _hydra_compose(routing)
    layer = load_profile(identity.robot.name, identity.dataset.name, selection)
    store.store(group='calibration_profile', name='active',
                node=OmegaConf.to_container(layer, resolve=False), package='_global_')
    return _hydra_compose(remaining)


def _hydra_compose(overrides: list[str]):
    try:
        from hydra import compose as _compose
        from hydra import initialize_config_dir
    except ImportError:
        sys.exit("the frdv command-line tools need Hydra: pip install 'frdv[cli]'  (or uv sync --extra cli)")
    user = user_conf_dir()
    extra = [f"hydra.searchpath=[file://{user}]"] if user.is_dir() else []
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        return _compose(config_name="config", overrides=extra + list(overrides))


def cli(doc: str, run, argv=None, extra_help: str = ""):
    """Parse ``[--help] [--cfg] key=value ...`` and call ``run(cfg)``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help") for a in argv):
        print(doc.strip())
        print(f"\nrobots:   {', '.join(available('robot'))}")
        print(f"datasets: {', '.join(available('dataset'))}")
        print(f"scaling:  {', '.join(available('scaling'))}")
        print(f"subjects: {', '.join(available('subject'))}")
        print(f"user configs: {user_conf_dir()} (set FRDV_CONFIG_DIR to change)")
        if extra_help:
            print(extra_help)
        return 0
    show = "--cfg" in argv
    argv = [a for a in argv if a != "--cfg"]
    bad = [a for a in argv if "=" not in a and not a.startswith("~")]   # ~key deletes a config entry
    if bad:
        print(f"unexpected argument(s): {' '.join(bad)}  (use key=value, ~key to delete, e.g. robot=g1 data=/path)")
        return 2
    cfg = compose(argv)
    if show:
        from omegaconf import OmegaConf
        cfg_print = OmegaConf.to_container(cfg, resolve=False)
        cfg_print.pop("hydra", None)
        print(OmegaConf.to_yaml(cfg_print))
        return 0
    return run(cfg)
