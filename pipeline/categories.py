from pathlib import Path

DEFAULT_CATEGORIES: dict[str, str] = {
    "shipwreck": "Probable shipwreck or structural debris",
    "rock": "Geological feature \u2014 rock or reef",
    "shadow_artifact": "Sonar shadow / processing artifact",
    "biological": "Biological mass (e.g. fish school, kelp)",
    "unknown": "Cannot determine from sonar alone",
}

_YAML_PATH = Path(__file__).parent / "categories.yaml"


def load_categories() -> dict[str, str]:
    if _YAML_PATH.exists():
        import yaml  # transitive dep of streamlit -- no new dependency
        with open(_YAML_PATH, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    return DEFAULT_CATEGORIES
