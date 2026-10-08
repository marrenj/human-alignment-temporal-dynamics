"""Read the `runs_root` Lightning `hparams.yaml` files.

Lightning serialized the `scratch_training` config dataclasses into
`hparams.yaml` as `!!python/object:config.supervised_config.*` tags, which
`yaml.safe_load` refuses and `FullLoader` can't import here. We only want the
plain scalars underneath, so any unknown tag becomes its bare mapping/sequence.
"""

from pathlib import Path

import yaml


class TolerantLoader(yaml.SafeLoader):
    pass


def _construct_unknown(loader, suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


TolerantLoader.add_multi_constructor("", _construct_unknown)
TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/object:", _construct_unknown)


def find_key(node, key):
    """First non-null value for `key` anywhere in the nested hparams mapping.

    Lightning nests the config dataclasses several levels deep and also emits a
    null-valued schema echo of the same keys alongside it, so the first *match*
    is often null while the real value sits elsewhere.
    """
    if isinstance(node, dict):
        if node.get(key) is not None:
            return node[key]
        for v in node.values():
            found = find_key(v, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for v in node:
            found = find_key(v, key)
            if found is not None:
                return found
    return None


def read_hparams(run_dir):
    hparams = sorted(Path(run_dir).glob("version_*/hparams.yaml"))
    if not hparams:
        raise FileNotFoundError(f"No version_*/hparams.yaml under {run_dir}")
    with open(hparams[0]) as f:
        return yaml.load(f, Loader=TolerantLoader)


def onset_epoch(run_dir):
    """`perturb_epoch` from the run's Lightning hparams.yaml, or None if unperturbed."""
    return find_key(read_hparams(run_dir), "perturb_epoch")
