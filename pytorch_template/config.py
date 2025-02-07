import toml


def get_config() -> dict:
    config = toml.load("./config.toml")
    return config