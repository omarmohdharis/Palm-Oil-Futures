from pathlib import Path
import yaml

_ROOT = Path(__file__).parent.parent.parent
_CONFIG_DIR = _ROOT / "config"


def load_config(name: str) -> dict:
    path = _CONFIG_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_root() -> Path:
    return _ROOT
