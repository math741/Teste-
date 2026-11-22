"""Monta GENESIS_OMEGA.ipynb a partir de notebook_src/*.py (formato percent: '# %%' e '# %% [markdown]')."""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def cells_from(path):
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"^# %%(.*)$", text, flags=re.M)
    out = []
    for i in range(1, len(parts), 2):
        kind, body = parts[i].strip(), parts[i + 1].strip("\n")
        if not body.strip():
            continue
        if kind.startswith("[markdown]"):
            lines = [l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l) for l in body.splitlines()]
            out.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body})
    return out


def build(out_path):
    cells = []
    for f in sorted(glob.glob(os.path.join(HERE, "notebook_src", "*.py"))):
        cells.extend(cells_from(f))
    for i, c in enumerate(cells):
        c["id"] = f"genesis-{i:03d}"
        src = c["source"]
        c["source"] = [l + "\n" for l in src.split("\n")]
        c["source"][-1] = c["source"][-1].rstrip("\n")
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4", "name": "GENESIS_OMEGA.ipynb"},
                       "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"{out_path}: {len(cells)} células")


def script(out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for p in sorted(glob.glob(os.path.join(HERE, "notebook_src", "*.py"))):
            f.write(open(p, encoding="utf-8").read() + "\n\n")


if __name__ == "__main__":
    build(os.path.join(HERE, "GENESIS_OMEGA.ipynb"))
    if len(sys.argv) > 1:
        script(sys.argv[1])
