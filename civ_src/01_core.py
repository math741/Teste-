# %% [markdown]
# ## 01 — Núcleo: versões, armazenamento híbrido, logging com write-behind, estatística e instrumentos externos
# Infraestrutura **externa** ao universo. Nada aqui é visível para os agentes.

# %%
import sys, time, json, math, uuid, zlib, gzip, shutil, pickle, hashlib, random, platform, traceback, warnings, copy
import heapq, itertools, io, csv, glob, datetime, subprocess, contextlib, tempfile, base64, collections, threading
import asyncio, secrets, socket, struct, re, importlib, importlib.util, queue
import html as _html
from dataclasses import dataclass, field, asdict, fields as dc_fields, replace as dc_replace
from typing import Any, Dict, List, Optional, Tuple, Sequence, Callable, Generator, Iterable, Set

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
    from IPython.display import display, clear_output, HTML, Markdown
except Exception:
    def display(*a, **k):
        for x in a:
            print(x)
        return None
    def clear_output(*a, **k):
        return None
    HTML = str
    Markdown = str

try:
    import scipy.stats as sstats
    import scipy.ndimage as sndi
    HAVE_SCIPY = True
except Exception:
    sstats, sndi, HAVE_SCIPY = None, None, False
try:
    import psutil
    HAVE_PSUTIL = True
except Exception:
    psutil, HAVE_PSUTIL = None, False

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------- versões (toda descoberta carrega o hash)
GENESIS_VERSION = "2.0.0-civilization"
WORLD_ENGINE_VERSION = "WORLD-1.0"
MATTER_ENGINE_VERSION = "MATTER-1.0"
DEVELOPMENT_ENGINE_VERSION = "DEV-1.0"
BODY_ENGINE_VERSION = "BODY-1.0"
MIND_ENGINE_VERSION = "MIND-1.0"
AFFECT_ENGINE_VERSION = "AFFECT-1.0"
SOCIAL_ENGINE_VERSION = "SOCIAL-1.0"
CIVILIZATION_ENGINE_VERSION = "CIV-1.0"
VALIDATION_ENGINE_VERSION = "VE-2.0"
RENDER_PROTOCOL_VERSION = "RP-1.0"
TELEMETRY_SCHEMA_VERSION = "TEL-1.0"
EVENT_SCHEMA_VERSION = "EV-1.0"
CHECKPOINT_VERSION = 3          # 1–2 = Genesis Ω (2D); 3 = Genesis Civilization (importa o legado como arquivo separado)
ENGINE_VERSIONS = {"genesis": GENESIS_VERSION, "world": WORLD_ENGINE_VERSION, "matter": MATTER_ENGINE_VERSION,
                   "development": DEVELOPMENT_ENGINE_VERSION, "body": BODY_ENGINE_VERSION, "mind": MIND_ENGINE_VERSION,
                   "affect": AFFECT_ENGINE_VERSION, "social": SOCIAL_ENGINE_VERSION,
                   "civilization": CIVILIZATION_ENGINE_VERSION, "validation": VALIDATION_ENGINE_VERSION,
                   "render_protocol": RENDER_PROTOCOL_VERSION, "telemetry": TELEMETRY_SCHEMA_VERSION,
                   "events": EVENT_SCHEMA_VERSION, "checkpoint": CHECKPOINT_VERSION}
CHANGELOG = [
    ("2.0.0-civilization", "Mundo 3D multiescala (terreno, clima, água, dia/noite, estações, tempo), matéria de construção, "
                           "genoma de desenvolvimento → corpo massa-mola, substrato de sinais com plasticidade evoluível, "
                           "agentes F0–F4, probes H/A/C com controles, Scientist com DSL segura, checkpoint em shards, "
                           "servidor realtime + observatório Three.js com acesso remoto autenticado. Legado Genesis Ω "
                           "importado como arquivo separado."),
]

STATUS_NOT_TESTED, STATUS_RUNNING, STATUS_PASS, STATUS_FAIL, STATUS_INCONCLUSIVE, STATUS_ERROR = (
    "NOT_TESTED", "RUNNING", "PASS", "FAIL", "INCONCLUSIVE", "ERROR")
LAB_STATES = ["BOOTING", "TESTING", "RESTORING", "SEARCHING", "EXPERIMENTING", "CHECKPOINTING", "PAUSED", "CRASHED",
              "CHAMPION", "OFFLINE"]
CLAIM_TAGS = ["IMPLEMENTED", "TESTED", "BENCHMARKED", "SIMULATION HYPOTHESIS", "NOT YET VALIDATED"]


def _hash_versions() -> str:
    return hashlib.sha256(json.dumps(ENGINE_VERSIONS, sort_keys=True).encode()).hexdigest()[:16]


ENGINE_VERSIONS_HASH = _hash_versions()


class _LiveHook:
    """Acoplamento opcional com a telemetria. Somente observação: nunca participa de fitness, seleção ou RNG."""
    monitor = None


LIVE = _LiveHook()


def live_progress(phase: Optional[str] = None, done: Optional[float] = None, total: Optional[float] = None,
                  detail: Optional[str] = None):
    m = LIVE.monitor
    if m is not None:
        m.progress(phase, done, total, detail)


# ----------------------------------------------------------------------------- armazenamento híbrido GPU↔RAM↔SSD↔Drive
CIV_SUBDIRS = ["checkpoints", "checkpoints/backups", "worlds", "agents", "genomes", "bodies", "lineages", "discoveries",
               "affect", "culture", "settlements", "civilizations", "history", "replays", "snapshots", "reports", "plots",
               "logs", "exports", "crash_reports", "pantheon", "legacy", "viewer_user"]


class CivPaths:
    def __init__(self, root: str, work: str, on_drive: bool):
        self.root, self.work, self.on_drive = os.path.abspath(root), os.path.abspath(work), on_drive
        for d in CIV_SUBDIRS:
            os.makedirs(os.path.join(self.root, d), exist_ok=True)
        for d in ["tmp", "viewer", "viewer/static", "viewer/static/vendor", "bin", "legacy_code", "replays", "cache"]:
            os.makedirs(os.path.join(self.work, d), exist_ok=True)
        self.stop_file = os.path.join(self.root, "STOP")

    def __getattr__(self, name):
        if name in [d.split("/")[-1] for d in CIV_SUBDIRS]:
            sub = next(d for d in CIV_SUBDIRS if d.split("/")[-1] == name)
            return os.path.join(self.__dict__["root"], sub)
        raise AttributeError(name)

    def sub(self, *parts) -> str:
        p = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def tmp(self, *parts) -> str:
        p = os.path.join(self.work, "tmp", *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p


def setup_storage(use_drive: bool, override: str = "", work_override: str = "") -> CivPaths:
    """Drive = persistência; /content/genesis_work = scratch SSD (dados temporários pesados nunca vão para o Drive)."""
    work = work_override or ("/content/genesis_work" if IN_COLAB else os.path.abspath("./genesis_work"))
    if override:
        return CivPaths(override, work, on_drive=override.startswith("/content/drive"))
    if IN_COLAB:
        drive_root = "/content/drive/MyDrive"
        if use_drive and not os.path.isdir(drive_root):
            try:
                from google.colab import drive
                drive.mount("/content/drive")
            except Exception as e:
                print(f"[storage] Drive não montado ({e}); usando /content (não persiste se a sessão cair).")
        if use_drive and os.path.isdir(drive_root):
            return CivPaths(os.path.join(drive_root, "GENESIS_CIVILIZATION_OMEGA"), work, on_drive=True)
        return CivPaths("/content/GENESIS_CIVILIZATION_OMEGA", work, on_drive=False)
    return CivPaths(os.path.abspath("./genesis_civilization_runs"), work, on_drive=False)


# ----------------------------------------------------------------------------- utilidades
def now_ts() -> float:
    return time.time()


def iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def derive_seed(*parts) -> int:
    h = hashlib.sha256(("|".join(str(p) for p in parts)).encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, np.generic):
        return to_jsonable(o.item())
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, torch.Tensor):
        return to_jsonable(o.detach().cpu().numpy())
    if hasattr(o, "__dataclass_fields__"):
        return to_jsonable(asdict(o))
    return o


def hash_obj(obj) -> str:
    hsh = hashlib.sha256()

    def feed(x):
        if isinstance(x, dict):
            hsh.update(b"{")
            for k in sorted(x.keys(), key=str):
                hsh.update(str(k).encode()); feed(x[k])
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


def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class RingBuffer:
    """Buffer circular de tamanho fixo (sem listas Python sem limite no caminho quente)."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.items: "collections.deque" = collections.deque(maxlen=self.capacity)

    def push(self, x):
        self.items.append(x)

    def last(self, n: int) -> list:
        return list(self.items)[-n:]

    def __len__(self):
        return len(self.items)

    def state_dict(self):
        return {"capacity": self.capacity, "items": list(self.items)}

    def load_state_dict(self, s):
        self.capacity = s["capacity"]; self.items = collections.deque(s["items"], maxlen=self.capacity)


class EventLogger:
    """Eventos estruturados (schema versionado) com write-behind: lote em memória, gravação em blocos no disco."""
    MAX_LOG_BYTES = 64 * 1024 * 1024

    def __init__(self, paths: CivPaths, verbose: bool = True, flush_every: float = 20.0):
        self.paths = paths
        self.events_path = os.path.join(paths.logs, "events.jsonl")
        self.metrics_path = os.path.join(paths.logs, "metrics.csv")
        self.verbose, self.flush_every = verbose, flush_every
        self.recent = RingBuffer(400)
        self._buf: List[str] = []
        self._mbuf: List[Dict[str, Any]] = []
        self._metric_fields: Optional[List[str]] = None
        self._last_flush = time.time()
        self._lock = threading.Lock()

    def event(self, kind: str, echo: bool = False, **data):
        rec = {"schema": EVENT_SCHEMA_VERSION, "ts": iso_now(), "kind": kind, **to_jsonable(data)}
        with self._lock:
            self.recent.push(rec)
            self._buf.append(json.dumps(rec))
        if echo and self.verbose:
            msg = ", ".join(f"{k}={v}" for k, v in data.items() if not isinstance(v, (dict, list)))
            print(f"[{kind}] {msg}")
        if time.time() - self._last_flush > self.flush_every:
            self.flush()

    def metrics(self, row: Dict[str, Any]):
        with self._lock:
            self._mbuf.append({k: to_jsonable(v) for k, v in row.items()})
        if time.time() - self._last_flush > self.flush_every:
            self.flush()

    def _rotate(self, path):
        if os.path.exists(path) and os.path.getsize(path) > self.MAX_LOG_BYTES:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(path, "rb") as fi, gzip.open(f"{path}.{stamp}.gz", "wb") as fo:
                shutil.copyfileobj(fi, fo)
            os.remove(path)

    def flush(self):
        with self._lock:
            buf, self._buf = self._buf, []
            mbuf, self._mbuf = self._mbuf, []
        self._last_flush = time.time()
        try:
            if buf:
                self._rotate(self.events_path)
                with open(self.events_path, "a") as f:
                    f.write("\n".join(buf) + "\n")
            if mbuf:
                self._rotate(self.metrics_path)
                exists = os.path.exists(self.metrics_path)
                if exists and self._metric_fields is None:
                    with open(self.metrics_path) as f:
                        header = f.readline().strip()
                        self._metric_fields = header.split(",") if header else None
                if self._metric_fields is None:
                    self._metric_fields = list(mbuf[0].keys())
                    exists = False
                with open(self.metrics_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=self._metric_fields, extrasaction="ignore")
                    if not exists:
                        w.writeheader()
                    for r in mbuf:
                        w.writerow(r)
        except Exception as e:
            print(f"[log] falha ao gravar logs: {e}")


# ----------------------------------------------------------------------------- estatística (instrumentos externos)
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


# ----------------------------------------------------------------------------- decodificadores externos
def _standardize(Xtr, Xte):
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_predict(Xtr, ytr, Xte, n_classes, lam=1.0):
    Xtr, Xte = _standardize(Xtr, Xte)
    Xtr = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))]); Xte = np.hstack([Xte, np.ones((Xte.shape[0], 1))])
    Y = np.eye(n_classes)[ytr]
    n, d = Xtr.shape
    if n < d:   # forma dual
        Kt = Xtr @ Xtr.T
        alpha = np.linalg.solve(Kt + lam * np.eye(n), Y)
        scores = Xte @ (Xtr.T @ alpha)
    else:
        Wt = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ Y)
        scores = Xte @ Wt
    return scores.argmax(1)


def knn_predict(Xtr, ytr, Xte, n_classes, k=5):
    Xtr, Xte = _standardize(Xtr, Xte)
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    k = max(1, min(k, Xtr.shape[0]))
    nn = np.argsort(d, axis=1)[:, :k]
    votes = np.zeros((Xte.shape[0], n_classes))
    for j in range(k):
        votes[np.arange(Xte.shape[0]), ytr[nn[:, j]]] += 1.0 / (1 + j * 0.01)
    return votes.argmax(1)


def stratified_folds(y, k, rng):
    folds = np.zeros(len(y), int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % k
    return folds


def balanced_accuracy(y, yp, n_classes):
    accs = [np.mean(yp[y == c] == c) for c in range(n_classes) if np.any(y == c)]
    return float(np.mean(accs)) if accs else 0.0


def cv_predictions(X, y, n_classes, decoder, k, seed):
    rng = np.random.default_rng(seed)
    k = max(2, min(k, int(np.bincount(y).min()) if len(np.unique(y)) > 1 else 2))
    folds = stratified_folds(y, k, rng)
    yp = np.zeros_like(y)
    for f in range(k):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        yp[te] = decoder(X[tr], y[tr], X[te], n_classes)
    return yp


def mutual_info_bits(y, yp, n_classes) -> float:
    """MI plug-in da matriz de confusão com correção de Miller–Madow (viés reconhecido)."""
    conf = np.zeros((n_classes, n_classes))
    for a, b in zip(y, yp):
        conf[a, b] += 1
    n = conf.sum()
    if n == 0:
        return 0.0
    pj = conf / n
    px = pj.sum(1, keepdims=True); py = pj.sum(0, keepdims=True)
    nz = pj > 0
    mi = float((pj[nz] * np.log2(pj[nz] / (px @ py)[nz])).sum())
    k_joint, k_x, k_y = nz.sum(), (px > 0).sum(), (py > 0).sum()
    mm = (k_joint - k_x - k_y + 1) / (2 * n * math.log(2))
    return max(0.0, mi - mm)


def decode_report(X: np.ndarray, y: np.ndarray, n_classes: int, n_perm: int, folds: int, seed: int,
                  nonlinear: bool = True) -> Dict[str, Any]:
    """Acurácia balanceada CV (linear e kNN), p-valor por permutação, confusão, MI."""
    X = np.nan_to_num(np.asarray(X, np.float64)); y = np.asarray(y, int)
    chance = 1.0 / n_classes
    out: Dict[str, Any] = {"n": int(len(y)), "chance": chance, "n_classes": n_classes,
                           "method": "balanced acc, stratified CV (ridge linear + kNN), label-permutation null (empirical + Gaussian tail)"}
    if len(np.unique(y)) < 2 or X.shape[0] < 2 * n_classes:
        out.update({"linear_acc": chance, "linear_p": 1.0, "nonlinear_acc": chance, "nonlinear_p": 1.0, "mi_bits": 0.0})
        return out
    if np.all(X.std(0) < 1e-9):
        out.update({"linear_acc": chance, "linear_p": 1.0, "nonlinear_acc": chance, "nonlinear_p": 1.0, "mi_bits": 0.0,
                    "note": "features constantes"})
        return out
    rng = np.random.default_rng(seed)
    for name, dec in [("linear", ridge_predict)] + ([("nonlinear", knn_predict)] if nonlinear else []):
        yp = cv_predictions(X, y, n_classes, dec, folds, seed)
        acc = balanced_accuracy(y, yp, n_classes)
        null = []
        for i in range(n_perm):
            if i % 10 == 0:
                live_progress()
            yperm = rng.permutation(y)
            ypp = cv_predictions(X, yperm, n_classes, dec, folds, seed + 1 + i)
            null.append(balanced_accuracy(yperm, ypp, n_classes))
        null = np.asarray(null)
        p_emp = (np.sum(null >= acc - 1e-12) + 1) / (n_perm + 1)
        # cauda gaussiana da distribuição nula de permutação: permite p pequenos com poucas permutações;
        # só é usada quando a acurácia observada supera TODAS as permutações (senão vale o p empírico).
        sd = max(float(null.std()), 1e-3) if null.size else 1.0
        z = (acc - (float(null.mean()) if null.size else chance)) / sd
        p_par = 0.5 * math.erfc(z / math.sqrt(2))
        p = p_par if (null.size and acc > null.max()) else max(p_emp, p_par)
        out[f"{name}_acc"] = acc
        out[f"{name}_p"] = float(p)
        out[f"{name}_p_empirical"] = float(p_emp)
        out[f"{name}_null_mean"] = float(null.mean()) if null.size else chance
        if name == "linear":
            out["mi_bits"] = mutual_info_bits(y, yp, n_classes)
            conf = np.zeros((n_classes, n_classes), int)
            for a, b in zip(y, yp):
                conf[a, b] += 1
            out["confusion"] = conf.tolist()
            out["effect_size"] = float((acc - out[f"{name}_null_mean"]) / (null.std() + 1e-6)) if null.size else 0.0
    if not nonlinear:
        out["nonlinear_acc"], out["nonlinear_p"] = float("nan"), 1.0
    return out


def norm_acc(acc: float, chance: float) -> float:
    return float((acc - chance) / (1 - chance))


print(f"PROJECT GENESIS CIVILIZATION Ω∞ v{GENESIS_VERSION} | Python {platform.python_version()} | torch {torch.__version__} | "
      f"Colab={IN_COLAB} | notebook={IN_NOTEBOOK} | versões={ENGINE_VERSIONS_HASH}")
