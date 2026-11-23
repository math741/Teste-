# %% [markdown]
# ## 9. Checkpoint atômico, validação por hash, migração e backups
# `write checkpoint.tmp → validate → checkpoint_current → (anterior vira checkpoint_previous)`.
# Corrupção detectada por hash → fallback automático para o anterior, depois para backups.
# Carregamento sempre com `map_location="cpu"` (independente de hardware).

# %%
CHECKPOINT_REQUIRED_KEYS = ["version", "checkpoint_version", "genesis_version", "config", "counters", "population",
                            "store", "map_elites", "novelty", "zoo", "anomalies", "failures", "champions", "lineage",
                            "species", "scientist", "rng", "hashes", "history", "results", "tests"]


def _ckpt_hashes(state: dict) -> Dict[str, str]:
    return {"population": hash_obj(state["population"]),
            "store": hash_obj({k: v["packed"] for k, v in state["store"].items()}),
            "map_elites": hash_obj(state["map_elites"]["cells"]),
            "counters": hash_obj(state["counters"])}


def migrate_checkpoint(state: dict) -> dict:
    """Estrutura de migração: aplica conversores sucessivos v -> v+1."""
    def v1_to_v2(s: dict) -> dict:
        # Genesis 1.1: busca dirigida + canal com distância inicial de currículo 12 (RESEARCH/FAST/EXTREME).
        # A configuração científica muda por migração explícita e registrada (histórico de thresholds).
        s = dict(s)
        cfg = copy.deepcopy(s["config"])
        pr = cfg.setdefault("probe", {})
        old_d = pr.get("channel_distance")
        if cfg.get("profile") in ("RESEARCH", "FAST", "EXTREME") and isinstance(old_d, int) and old_d > 12:
            pr["channel_distance"] = 12
        cfg.setdefault("directed_search", True)
        cfg.setdefault("local_rate_limit", False)     # pesquisas antigas continuam com a mesma física (dt global)
        cfg.setdefault("focus_tests_per_generation", 3)
        s["config"] = cfg
        s["migrations"] = list(s.get("migrations", [])) + [{"from": 1, "to": 2, "ts": iso_now(),
                                                             "channel_distance": [old_d, pr.get("channel_distance")]}]
        s["checkpoint_version"] = 2
        return s

    MIGRATIONS: Dict[int, Callable[[dict], dict]] = {1: v1_to_v2}
    v = int(state.get("checkpoint_version", 1))
    while v < CHECKPOINT_VERSION:
        if v not in MIGRATIONS:
            raise RuntimeError(f"sem migração de checkpoint v{v} -> v{v + 1}")
        state = MIGRATIONS[v](state)
        v = int(state["checkpoint_version"])
    return state


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class CheckpointManager:
    def __init__(self, paths: GenesisPaths, logger: EventLogger, keep_backups: int = 3, backup_every: int = 5):
        self.paths, self.logger = paths, logger
        self.current = os.path.join(paths.checkpoints, "checkpoint_current.pt")
        self.previous = os.path.join(paths.checkpoints, "checkpoint_previous.pt")
        self.tmp = os.path.join(paths.checkpoints, "checkpoint.tmp")
        self.keep_backups, self.backup_every = keep_backups, backup_every
        self.saves = 0
        self.last_save_ts: Optional[float] = None
        self.last_path: Optional[str] = None

    def validate(self, path: str) -> Tuple[bool, Optional[dict], str]:
        try:
            st = _torch_load(path)
        except Exception as e:
            return False, None, f"load failed: {e}"
        missing = [k for k in CHECKPOINT_REQUIRED_KEYS if k not in st]
        if missing:
            return False, None, f"missing keys: {missing}"
        try:
            h = _ckpt_hashes(st)
        except Exception as e:
            return False, None, f"hash failed: {e}"
        if h != st["hashes"]:
            bad = [k for k in h if h[k] != st["hashes"].get(k)]
            return False, None, f"hash mismatch: {bad}"
        return True, st, "ok"

    def save(self, state: dict) -> str:
        state = dict(state)
        state["hashes"] = _ckpt_hashes(state)
        t0 = time.time()
        torch.save(state, self.tmp)
        ok, _, msg = self.validate(self.tmp)
        if not ok:
            self.logger.event("checkpoint_invalid", echo=True, reason=msg)
            raise RuntimeError(f"checkpoint temporário inválido: {msg}")
        if os.path.exists(self.current):
            os.replace(self.current, self.previous)
        os.replace(self.tmp, self.current)
        self.saves += 1
        if self.saves % self.backup_every == 0:
            bk = os.path.join(self.paths.backups, f"checkpoint_gen{state['counters']['generation']:07d}.pt")
            shutil.copy2(self.current, bk)
            bks = sorted(glob.glob(os.path.join(self.paths.backups, "checkpoint_gen*.pt")))
            for old in bks[:-self.keep_backups]:
                os.remove(old)
        self.last_save_ts = time.time()
        self.last_path = self.current
        size = os.path.getsize(self.current)
        self.logger.event("checkpoint_saved", generation=state["counters"]["generation"], size=size,
                          seconds=round(time.time() - t0, 2))
        return self.current

    def load_latest(self) -> Tuple[Optional[dict], str]:
        candidates = [self.current, self.previous] + sorted(glob.glob(os.path.join(self.paths.backups, "checkpoint_gen*.pt")), reverse=True)
        for p in candidates:
            if not os.path.exists(p):
                continue
            ok, st, msg = self.validate(p)
            if ok:
                try:
                    st = migrate_checkpoint(st)
                except Exception as e:
                    self.logger.event("checkpoint_migration_failed", echo=True, path=p, error=repr(e))
                    continue
                if p != self.current:
                    self.logger.event("checkpoint_fallback", echo=True, used=p)
                return st, p
            self.logger.event("checkpoint_corrupt", echo=True, path=p, reason=msg)
        return None, ""
