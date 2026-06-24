try:
    import nemotron_nemo_bridge  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "nemo_rl":
        raise
