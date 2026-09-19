"""Copy the example scripts into the current directory:

    python -m FuncyFlows.examples            # copies *.py (skips files that already exist)
    python -m FuncyFlows.examples --force    # overwrite
    python -m FuncyFlows.examples --list

Then, e.g.:  python bimodal_posterior.py
"""
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).parent
EXAMPLES = sorted(p for p in HERE.glob("*.py") if not p.name.startswith("__"))


def main(argv):
    if "--list" in argv:
        for path in EXAMPLES:
            doc = (path.read_text().split('"""')[1].strip().splitlines() or [""])[0]
            print(f"  {path.name:24s} {doc}")
        return
    force = "--force" in argv
    target = pathlib.Path.cwd()
    for path in EXAMPLES:
        dest = target / path.name
        if dest.exists() and not force:
            print(f"  exists, skipped: {dest.name}   (--force to overwrite)")
            continue
        shutil.copy(path, dest)
        print(f"  wrote {dest.name}")
    print("\nrun any of them with  python <name>.py  (they import _common.py from the same directory)")


if __name__ == "__main__":
    main(sys.argv[1:])
