import re
import toml

# poetry export --without-hashes > poetry-requirements.txt
requirements = open("poetry-requirements.txt").read().strip().split("\n")
# poetry lock
lock_data = toml.load("poetry.lock")

packages = {p["name"].lower(): p for p in lock_data["package"]}


def add_subdeps(name: str, dep_set: set[str]) -> None:
    for dep in packages[name.lower()].get("dependencies", []):
        if dep in dep_set:
            continue  # torch -> triton -> torch
        dep_set.add(dep)
        add_subdeps(dep, dep_set)
    dep_set.add(name)


torch_deps = set()
add_subdeps("torch", torch_deps)

diffusers_deps = set()
add_subdeps("diffusers", diffusers_deps)
add_subdeps("safetensors", diffusers_deps)
diffusers_deps -= torch_deps

# https://peps.python.org/pep-0508/#names
# '^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])'
# must consist entirely of ASCII letters, numbers, ., -, and/or _
pattern = re.compile("^[-\\w\\.]+")
require_lines = [(pattern.search(line).group(), line) for line in requirements if line]

torch = "\n".join(line for name, line in require_lines if name in torch_deps)
other = "\n".join(
    line for name, line in require_lines if name not in torch_deps and name != "cog"
)
diffusers = "\n".join(line for name, line in require_lines if name in diffusers_deps)
open("torch-requirements.txt", "w").write(torch)
open("other-requirements.txt", "w").write(other)
open("diffusers-requirements.txt", "w").write(diffusers)
