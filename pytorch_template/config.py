"""Configuration functions."""
import toml


def get_config() -> dict:
    """Loads configuration file."""
    config = toml.load("./config.toml")
    return config
