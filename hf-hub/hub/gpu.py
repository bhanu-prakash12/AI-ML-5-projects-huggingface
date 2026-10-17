"""ZeroGPU decorator that is a no-op locally. `spaces` must be imported before torch (done in app.py)."""
try:
    import spaces
except ImportError:  # local run
    spaces = None


def gpu(duration: int = 60):
    def deco(fn):
        return spaces.GPU(duration=duration)(fn) if spaces is not None else fn
    return deco
