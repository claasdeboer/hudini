"""The import-chain invariant: the torch-free modules stay torch-free.

A subprocess imports every torch-free module and reports which banned
libraries ended up loaded; in-process checks would see torch from other
tests.
"""

import subprocess
import sys

TORCH_FREE_MODULES = (
    "schema",
    "storage",
    "views",
    "corrections",
    "catalog",
    "privacy",
    "cli",
    "detection",
    "hsv",
    "archive",
    "web.render",
    "web.server",
)


def test_torch_free_modules_never_import_torch_or_paddleocr():
    imports = "".join(f"import hudini.{name}\n" for name in TORCH_FREE_MODULES)
    code = f"{imports}import sys\nprint(sorted({{'torch', 'paddleocr'}} & set(sys.modules)))\n"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
