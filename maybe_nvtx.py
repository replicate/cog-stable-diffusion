try:
    from nvtx import annotate
except ImportError:
    class annotate:
        def __init__(self, _: str) -> None:
            pass

        def __enter__(self, *args: "Any", **kwargs: "Any") -> "Any":
            pass

        def __exit__(self, *args: "Any", **kwargs: "Any") -> "Any":
            pass

        def __call__(self, func: "Any") -> "Any":
            return func

