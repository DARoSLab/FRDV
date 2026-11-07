"""Robot-owned calibration bases and subject overrides, shared by all CLI tools."""
from pathlib import Path

from omegaconf import OmegaConf

from .common import PACKAGE_DIR


def _component(value):
    value = str(value)
    if not value or value in ('.', '..') or '/' in value or '\\' in value:
        raise ValueError(f'Invalid calibration identity: {value!r}')
    return value


def roots():
    from .config import user_conf_dir
    return [PACKAGE_DIR / 'conf' / 'calibration', user_conf_dir() / 'calibration']


def subject_path(robot, dataset, selection):
    """Accept an owned path, collection/name, or the previous dataset/collection/name CLI form."""
    robot, dataset = _component(robot), _component(dataset)
    parts = str(selection).split('/')
    if any(not p or p in ('.', '..') or '\\' in p for p in parts):
        raise ValueError(f'Invalid subject profile: {selection!r}')
    if len(parts) >= 4 and parts[2] == 'subjects':
        if parts[0] != robot:
            raise ValueError(f"subject '{selection}' requires robot '{parts[0]}', got '{robot}'")
        if parts[1] != dataset:
            raise ValueError(f"subject '{selection}' requires dataset '{parts[1]}', got '{dataset}'")
        path = Path(*parts)
        return path if path.suffix == '.yaml' else Path(str(path) + '.yaml')
    # Explicit legacy dataset prefix must match, rather than being mistaken for a collection.
    from .config import available
    datasets = set(available('dataset'))
    if parts[0] in datasets:
        if parts[0] != dataset:
            raise ValueError(f"subject '{selection}' requires dataset '{parts[0]}', got '{dataset}'")
        parts = parts[1:]
    if len(parts) < 2:
        raise ValueError('subject must be collection/name, e.g. accad/Female1Walking_c3d')
    path = Path(robot, dataset, 'subjects', *parts)
    return path if path.suffix == '.yaml' else Path(str(path) + '.yaml')


def load_profile(robot, dataset, selection=None):
    """Package base -> user base -> package subject -> user subject. Never cross robot roots."""
    robot, dataset = _component(robot), _component(dataset)
    layer = OmegaConf.create(dict(
        calibration=dict(robot=robot, dataset=dataset, markers={}, profile=None, sources=[]),
        subject=dict(name=None, collection=None, dataset=None, robot=None)))
    base = Path(robot, dataset, 'base.yaml')
    selected = subject_path(robot, dataset, selection) if selection and selection != 'none' else None
    sources = []
    for relative in [base] + ([selected] if selected else []):
        found = False
        for root in roots():
            path = root / relative
            if not path.is_file():
                continue
            found = True
            doc = OmegaConf.load(path)
            unknown = set(doc) - {'dataset', 'calibration', 'subject'}
            if unknown:
                raise ValueError(f'{path}: unsupported calibration sections: {sorted(unknown)}')
            layer = OmegaConf.merge(layer, doc)
            for key, expected in [('robot', robot), ('dataset', dataset)]:
                actual = layer.calibration.get(key)
                declared = (doc.get('subject') or {}).get(key)
                if actual != expected or (declared and declared != expected):
                    raise ValueError(f'{path}: calibration requires {key} {actual if actual != expected else declared!r}, got {expected!r}')
            sources.append(str(path))
        if relative == selected and not found:
            raise ValueError(f'No subject calibration for robot {robot!r}, dataset {dataset!r}: {selection!r}. '
                             'Create a profile under this robot; profiles from another robot are never reused.')
    if selected:
        layer.subject = dict(name=selected.stem,
                             collection=Path(*selected.parts[3:-1]).as_posix(),
                             robot=robot, dataset=dataset)
        layer.calibration.profile = selected.as_posix()
    layer.calibration.sources = sources
    return layer
