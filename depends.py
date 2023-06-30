import re
import toml

# poetry export --without-hashes > poetry-requirements.txt
requirements = open("poetry-requirements.txt").read().trim().split("\n")
# poetry lock
lock_data = toml.load("poetry.lock")

packages = {p["name"].lower(): p for p in lock_data["package"]}

torch_deps = set()


def add_subdeps(name: str) -> None:
    if name in torch_deps:
        return  # torch -> triton -> torch
    for dep in packages[name.lower()].get("dependencies", []):
        torch_deps.add(dep)
        add_subdeps(dep)


add_subdeps("torch")
torch_deps.add("torch")


# https://peps.python.org/pep-0508/#names
# '^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])'
pattern = re.compile("^[-\\w\\.]+")
require_lines = [(pattern.search(line).group(), line) for line in requirements if line]

torch = "\n".join(line for name, line in require_lines if name in torch_deps)
other = "\n".join(line for name, line in require_lines if name not in torch_deps)
open("torch-requirements.txt", "w").write(torch)
open("other-requirements.txt", "w").write(other)
