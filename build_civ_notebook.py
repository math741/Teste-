"""Monta GENESIS_CIVILIZATION_OMEGA.ipynb a partir de civ_src/*.py (formato percent), embutindo:
* o código-fonte exato do Genesis Ω original (notebook_src 00–14) para o LegacyBridge;
* os arquivos do observatório 3D (civ_src/viewer/*).
Uso: python build_civ_notebook.py [saida.ipynb] [--script saida.py]"""
import base64, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_notebook import cells_from, add_colab_titles  # noqa: E402


def legacy_b64() -> str:
    files = [f for f in sorted(glob.glob(os.path.join(HERE, "notebook_src", "*.py"))) if not os.path.basename(f).startswith("15_")]
    src = "\n\n".join(open(f, encoding="utf-8").read() for f in files)
    return base64.b64encode(src.encode("utf-8")).decode()


def viewer_b64() -> str:
    d = {os.path.basename(f): open(f, encoding="utf-8").read()
         for f in sorted(glob.glob(os.path.join(HERE, "civ_src", "viewer", "*"))) if f.endswith((".html", ".js", ".css"))}
    return base64.b64encode(json.dumps(d).encode("utf-8")).decode()


def sources():
    lb, vb = legacy_b64(), viewer_b64()
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, "civ_src", "*.py"))):
        text = open(f, encoding="utf-8").read().replace('"@@LEGACY_B64@@"', f'"{lb}"').replace('"@@VIEWER_FILES@@"', f'"{vb}"')
        out.append((f, text))
    return out


def build(out_path: str):
    cells = []
    tmp = os.path.join(HERE, ".build_tmp.py")
    for f, text in sources():
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        cells.extend(cells_from(tmp))
    os.remove(tmp)
    add_colab_titles(cells)
    for i, c in enumerate(cells):
        c["id"] = f"genesis-civ-{i:03d}"
        src = c["source"]
        c["source"] = [l + "\n" for l in src.split("\n")]
        c["source"][-1] = c["source"][-1].rstrip("\n")
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm",
                                                       "name": "GENESIS_CIVILIZATION_OMEGA.ipynb"},
                       "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
    print(f"{out_path}: {len(cells)} células, {os.path.getsize(out_path) / 1e6:.2f} MB")


def script(out_path: str):
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(t for _, t in sources()))
    print(f"{out_path}: script gerado")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--script" in args:
        i = args.index("--script"); script(args[i + 1]); args = args[:i] + args[i + 2:]
    build(args[0] if args else os.path.join(HERE, "GENESIS_CIVILIZATION_OMEGA.ipynb"))
