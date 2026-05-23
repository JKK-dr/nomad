class _NoOpWandb:
    run = None

    class Settings:
        def __init__(self, *args, **kwargs):
            pass

    class Image:
        def __init__(self, *args, **kwargs):
            pass

    class _Config:
        def update(self, *args, **kwargs):
            pass

    config = _Config()

    def login(self, *args, **kwargs):
        pass

    def init(self, *args, **kwargs):
        return None

    def save(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass


try:
    import wandb as wandb
except ImportError:
    wandb = _NoOpWandb()
