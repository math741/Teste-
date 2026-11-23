# %% [markdown]
# ## 1. Núcleo: imports, armazenamento, logging estruturado, utilidades e estatística
# Tudo aqui é infraestrutura **externa** ao universo (o simulador, o laboratório e seus instrumentos).

# %%
import sys, time, json, math, uuid, zlib, gzip, shutil, pickle, hashlib, random, platform
import traceback, warnings, copy, heapq, itertools, io, csv, glob, datetime, subprocess, contextlib, tempfile
import base64, collections
import html as _html
from dataclasses import dataclass, field, asdict, fields as dc_fields, replace as dc_replace
from typing import Any, Dict, List, Optional, Tuple, Sequence, Callable

import numpy as np
import torch
import torch.nn.functional as F

IN_NOTEBOOK = "ipykernel" in sys.modules
IN_COLAB = "google.colab" in sys.modules
if not IN_COLAB:
    try:
        import google.colab  # noqa: F401
        IN_COLAB = True
    except Exception:
        IN_COLAB = False

import matplotlib
if not IN_NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from IPython.display import display, clear_output, Markdown, HTML, Image as IPImage
except Exception:  # execução como script
    def display(*a, **k):
        for x in a:
            print(x)
    def clear_output(*a, **k):
        return None
    Markdown = str
    HTML = str
    IPImage = None

try:
    import scipy.stats as sstats
    import scipy.ndimage as sndi
    HAVE_SCIPY = True
except Exception:
    sstats, sndi, HAVE_SCIPY = None, None, False

try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    pd, HAVE_PANDAS = None, False

try:
    import psutil
    HAVE_PSUTIL = True
except Exception:
    psutil, HAVE_PSUTIL = None, False

warnings.filterwarnings("ignore", category=UserWarning)

GENESIS_VERSION = "1.0.0"
CHECKPOINT_VERSION = 1
VALIDATION_ENGINE_VERSION = "VE-1.0"
GENESIS_CHANGELOG = [
    ("1.0.0", "Primeira versão completa: física contínua multicanal evoluível, QD (MAP-Elites + novelty + especiação), "
              "Validation Engine congelado com controles, Scientist por valor de informação, checkpoints atômicos."),
]

STATUS_NOT_TESTED, STATUS_RUNNING, STATUS_PASS, STATUS_FAIL, STATUS_INCONCLUSIVE, STATUS_ERROR = (
    "NOT_TESTED", "RUNNING", "PASS", "FAIL", "INCONCLUSIVE", "ERROR")


class _LiveHook:
    """Ponto de acoplamento opcional do painel ao vivo (módulo 12b). Somente observação: o painel lê o estado,
    nunca participa da seleção, do fitness ou dos testes."""
    monitor = None


LIVE = _LiveHook()


def live_progress(phase: Optional[str] = None, done: Optional[float] = None, total: Optional[float] = None,
                  detail: Optional[str] = None):
    m = LIVE.monitor
    if m is not None:
        m.progress(phase, done, total, detail)


# ----------------------------------------------------------------------------
# Armazenamento
# ----------------------------------------------------------------------------
@dataclass
class GenesisPaths:
    root: str
    on_drive: bool

    def __post_init__(self):
        self.checkpoints = os.path.join(self.root, "checkpoints")
        self.backups = os.path.join(self.root, "checkpoints", "backups")
        self.logs = os.path.join(self.root, "logs")
        self.discoveries = os.path.join(self.root, "discoveries")
        self.reports = os.path.join(self.root, "reports")
        self.exports = os.path.join(self.root, "exports")
        self.plots = os.path.join(self.root, "plots")
        self.champion = os.path.join(self.root, "champion")
        self.crash = os.path.join(self.root, "crash_reports")
        for d in [self.root, self.checkpoints, self.backups, self.logs, self.discoveries, self.reports,
                  self.exports, self.plots, self.champion, self.crash]:
            os.makedirs(d, exist_ok=True)
        self.stop_file = os.path.join(self.root, "STOP")


def setup_storage(use_drive: bool, override: str = "") -> GenesisPaths:
    """Google Drive se disponível/montável; senão /content; fora do Colab, ./genesis_runs."""
    if override:
        return GenesisPaths(os.path.abspath(override), on_drive=override.startswith("/content/drive"))
    if IN_COLAB:
        drive_root = "/content/drive/MyDrive"
        if use_drive and not os.path.isdir(drive_root):
            try:
                from google.colab import drive
                drive.mount("/content/drive")
            except Exception as e:
                print(f"[storage] Drive não montado ({e}); usando /content (não persiste se a sessão cair).")
        if use_drive and os.path.isdir(drive_root):
            return GenesisPaths(os.path.join(drive_root, "GENESIS_OMEGA"), on_drive=True)
        return GenesisPaths("/content/GENESIS_OMEGA", on_drive=False)
    return GenesisPaths(os.path.abspath("./genesis_runs"), on_drive=False)


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def now_ts() -> float:
    return time.time()


def iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_id(prefix: str) -> str:
    """IDs únicos, seguros contra colisão (uuid4)."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def derive_seed(*parts) -> int:
    h = hashlib.sha256(("|".join(str(p) for p in parts)).encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, torch.Tensor):
        return to_jsonable(o.detach().cpu().numpy())
    if hasattr(o, "__dataclass_fields__"):
        return to_jsonable(asdict(o))
    return o


def hash_obj(obj) -> str:
    """Hash estável de objetos críticos (dicts/listas/arrays)."""
    hsh = hashlib.sha256()

    def feed(x):
        if isinstance(x, dict):
            hsh.update(b"{")
            for k in sorted(x.keys(), key=str):
                hsh.update(str(k).encode())
                feed(x[k])
            hsh.update(b"}")
        elif isinstance(x, (list, tuple)):
            hsh.update(b"[")
            for v in x:
                feed(v)
            hsh.update(b"]")
        elif isinstance(x, (np.ndarray, np.generic)):
            x = np.asarray(x)
            hsh.update(str(x.dtype).encode() + str(x.shape).encode())
            hsh.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, torch.Tensor):
            feed(x.detach().cpu().numpy())
        elif isinstance(x, float):
            hsh.update(repr(round(x, 12)).encode())
        else:
            hsh.update(repr(x).encode())
    feed(obj)
    return hsh.hexdigest()


def fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def human_bytes(n: float) -> str:
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def is_oom_error(e: BaseException) -> bool:
    if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


# ----------------------------------------------------------------------------
# Logging estruturado (events.jsonl + metrics.csv) com rotação
# ----------------------------------------------------------------------------
class EventLogger:
    MAX_LOG_BYTES = 64 * 1024 * 1024

    def __init__(self, paths: GenesisPaths, verbose: bool = True):
        self.paths = paths
        self.events_path = os.path.join(paths.logs, "events.jsonl")
        self.metrics_path = os.path.join(paths.logs, "metrics.csv")
        self.verbose = verbose
        self._metric_fields: Optional[List[str]] = None
        self.recent: List[dict] = []

    def _rotate(self, path):
        try:
            if os.path.exists(path) and os.path.getsize(path) > self.MAX_LOG_BYTES:
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(path, "rb") as fi, gzip.open(f"{path}.{stamp}.gz", "wb") as fo:
                    shutil.copyfileobj(fi, fo)
                os.remove(path)
        except Exception as e:
            print(f"[log] rotação falhou: {e}")

    def event(self, kind: str, echo: bool = False, **data):
        rec = {"ts": iso_now(), "kind": kind, **to_jsonable(data)}
        self.recent.append(rec)
        self.recent = self.recent[-200:]
        try:
            self._rotate(self.events_path)
            with open(self.events_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            print(f"[log] falha ao gravar evento: {e}")
        if echo and self.verbose:
            msg = ", ".join(f"{k}={v}" for k, v in data.items() if not isinstance(v, (dict, list)))
            print(f"[{kind}] {msg}")

    def metrics(self, row: Dict[str, Any]):
        row = {k: to_jsonable(v) for k, v in row.items()}
        try:
            self._rotate(self.metrics_path)
            exists = os.path.exists(self.metrics_path)
            if exists and self._metric_fields is None:
                with open(self.metrics_path) as f:
                    header = f.readline().strip()
                    self._metric_fields = header.split(",") if header else None
            if self._metric_fields is None:
                self._metric_fields = list(row.keys())
                exists = False
            with open(self.metrics_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._metric_fields, extrasaction="ignore")
                if not exists:
                    w.writeheader()
                w.writerow(row)
        except Exception as e:
            print(f"[log] falha ao gravar métricas: {e}")


# ----------------------------------------------------------------------------
# Estatística (instrumentos externos)
# ----------------------------------------------------------------------------
def summary_stats(x) -> Dict[str, float]:
    x = np.asarray([v for v in np.ravel(x) if v is not None and np.isfinite(v)], dtype=np.float64)
    if x.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    lo, hi = bootstrap_ci(x)
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "std": float(x.std(ddof=1)) if x.size > 1 else 0.0, "ci_lo": lo, "ci_hi": hi}


def bootstrap_ci(x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0, stat=np.mean) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    bs = stat(x[idx], axis=1)
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def cohens_d(a, b) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    s = math.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    if s < 1e-12:
        diff = a.mean() - b.mean()
        return float(np.sign(diff) * 10.0) if abs(diff) > 1e-12 else 0.0
    return float((a.mean() - b.mean()) / s)


def mann_whitney_greater(a, b) -> float:
    """p-valor unilateral (a > b)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    if HAVE_SCIPY:
        try:
            return float(sstats.mannwhitneyu(a, b, alternative="greater").pvalue)
        except Exception as e:
            warnings.warn(f"mannwhitneyu (scipy) falhou ({e}); usando aproximação normal")
    # aproximação normal
    allv = np.concatenate([a, b])
    ranks = np.argsort(np.argsort(allv)) + 1.0
    u = ranks[:a.size].sum() - a.size * (a.size + 1) / 2
    mu = a.size * b.size / 2
    sd = math.sqrt(a.size * b.size * (a.size + b.size + 1) / 12) + 1e-12
    z = (u - mu) / sd
    return float(0.5 * math.erfc(z / math.sqrt(2)))


def binom_p_greater(k: int, n: int, p0: float) -> float:
    if n <= 0:
        return 1.0
    if HAVE_SCIPY:
        try:
            return float(sstats.binomtest(int(k), int(n), p0, alternative="greater").pvalue)
        except Exception as e:
            warnings.warn(f"binomtest (scipy) falhou ({e}); usando soma exata")
    return float(sum(math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(int(k), n + 1)))


def holm_correction(pvals: Sequence[float]) -> List[float]:
    p = np.asarray([1.0 if (v is None or not np.isfinite(v)) else v for v in pvals], dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj.tolist()


print(f"PROJECT GENESIS Ω v{GENESIS_VERSION} | Python {platform.python_version()} | torch {torch.__version__} | "
      f"Colab={IN_COLAB} | notebook={IN_NOTEBOOK} | scipy={HAVE_SCIPY}")
