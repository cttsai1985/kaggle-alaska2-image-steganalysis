"""Generate an offline Kaggle entry notebook. Hosted execution is explicitly unverified."""

from pathlib import Path

import nbformat as nbf


def build_entry() -> nbf.NotebookNode:
    return nbf.v4.new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "alaska2": {"hosted_kaggle_execution": "unverified", "network_required": False},
        },
        cells=[
            nbf.v4.new_markdown_cell(
                """# ALASKA2 offline walkthrough — hosted execution UNVERIFIED

Attach the published source capsule and SHA256SUMS as a private Kaggle input. Turn Internet off.
The source contains aggregate receipts and synthetic fixture generation, no competition images or
model weights. This entry executes the maintained walkthrough; it does not submit to Kaggle.

Use Python 3.12 and the exact locked dependencies. Kaggle's default image is not assumed equivalent.
If the runtime check fails, supply a prebuilt offline wheelhouse/cache from `uv.lock` and install it
before rerunning. See `docs/kaggle.md`. Hosted execution remains unverified until saved outputs from
that environment exist. The locally executed teaching notebook is a separate verified artifact.""",
                id="scope",
            ),
            nbf.v4.new_code_cell(
                """from pathlib import Path
import hashlib
import zipfile

INPUT = Path("/kaggle/input/alaska2-source")  # change to the attached input's actual path
OUTPUT = Path("/kaggle/working/alaska2")
archive = INPUT / "alaska2-source.zip"
checksums = dict(line.split(maxsplit=1)[::-1] for line in (INPUT / "SHA256SUMS").read_text().splitlines())
assert hashlib.sha256(archive.read_bytes()).hexdigest() == checksums[archive.name], "source checksum mismatch"
SOURCE = OUTPUT / "source"
SOURCE.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as files:
    assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in files.namelist())
    files.extractall(SOURCE)
print({"source_checksum_verified": True, "hosted_kaggle_execution": "unverified"})""",
                id="verify-source",
            ),
            nbf.v4.new_code_cell(
                """import os
import subprocess
import sys

locked_python = SOURCE / ".venv/bin/python"
python = str(locked_python) if locked_python.is_file() else sys.executable
env = {**os.environ, "PYTHONPATH": str(SOURCE / "src") + os.pathsep + str(SOURCE),
       "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "POLARS_MAX_THREADS": "4"}
subprocess.run([python, "-m", "tools.check_locked_runtime", "--root", str(SOURCE)],
               cwd=SOURCE, env=env, check=True)
subprocess.run([python, "-m", "tools.notebooks.execute_walkthrough",
                "--out", str(OUTPUT / "executed.ipynb"), "--artifacts", str(OUTPUT / "artifacts"),
                "--receipt", str(OUTPUT / "execution.json")], cwd=SOURCE, env=env, check=True)
print({"fixture_execution": "complete", "outputs": str(OUTPUT), "submission_sent": False})""",
                id="execute-maintained-walkthrough",
            ),
        ],
    )


def main() -> None:
    output = Path("notebooks/kaggle/rerun.ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_entry(), output)


if __name__ == "__main__":
    main()
