from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config():
    config_path = PROJECT_ROOT / "configs" / "config.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config