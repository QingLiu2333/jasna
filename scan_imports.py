"""List every third-party module imported anywhere in jasna, for the freeze.

Static analysis (PyInstaller) follows top-level imports, but jasna reaches for
many heavy dependencies lazily inside functions. This walks the AST of the whole
package so the frozen build ships them all.
"""

import ast
import importlib.util
import pathlib
import sys

PKG = pathlib.Path(r"D:\jasna2\work\jasna")
OUT = pathlib.Path(r"D:\jasna2\hiddenimports.txt")

stdlib = set(sys.stdlib_module_names)
local = {"jasna"}

found: dict[str, set[str]] = {}


def record(name: str, where: str) -> None:
    top = name.split(".")[0]
    if not top or top in stdlib or top in local or top.startswith("_"):
        return
    found.setdefault(top, set()).add(where)


for path in sorted(PKG.rglob("*.py")):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError as exc:
        print("SKIP", path, exc)
        continue
    rel = path.relative_to(PKG.parent.parent)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                record(alias.name, str(rel))
        elif isinstance(node, ast.ImportFrom):
            if node.level and not node.module:
                continue
            if node.module:
                # ``from x.y import z`` may itself import submodule z
                record(node.module, str(rel))
                for alias in node.names:
                    if alias.name != "*":
                        record(f"{node.module}.{alias.name}", str(rel))

resolved: list[str] = []
unresolved: list[str] = []
for top in sorted(found):
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError, ModuleNotFoundError):
        spec = None
    (resolved if spec is not None else unresolved).append(top)

OUT.write_text("\n".join(resolved) + "\n", encoding="utf-8")
print(f"third-party top-level modules imported: {len(found)}")
print(f"  importable now (-> hiddenimports): {len(resolved)}")
print("   ", ", ".join(resolved))
print(f"  not importable (optional/absent): {len(unresolved)}")
print("   ", ", ".join(unresolved))
print("wrote", OUT)
