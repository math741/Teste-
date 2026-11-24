# %% [markdown]
# ## 16 — CHECKPOINT ENGINE: shards + manifesto + hashes, atômico, write-behind, recuperação sem reset silencioso
# Fluxo: snapshot do estado (cópias GPU→CPU) → serialização em **shards** no SSD local (`/content/genesis_work`) →
# manifesto com SHA-256 de cada shard → validação (rehash) → cópia para o Drive em `checkpoints/.tmp_*` → conferência →
# rotação atômica `current → previous → backups/` e `rename(.tmp → current)`. A escrita roda numa thread (a simulação
# continua); no fim de sessão é síncrona. Formato: JSON + NPZ (`allow_pickle=False`) — **nunca pickle**.
# Carga: current → previous → backups (mais novo primeiro), cada um com verificação de hashes. Se existirem checkpoints
# e nenhum for válido, eles são preservados em `checkpoints/corrupted_*` e o boot **avisa** (nunca reset silencioso).

# %%
class CheckpointEngine:
    MAX_PART_BYTES = 48 * 1024 * 1024
    MIGRATIONS: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    def __init__(self, paths: CivPaths, logger: Optional[EventLogger], keep_backups: int = 3):
        self.paths, self.logger, self.keep = paths, logger, keep_backups
        self.root = paths.sub("checkpoints")
        self.backups = paths.sub("checkpoints/backups")
        self._thread: Optional[threading.Thread] = None
        self.last_info: Dict[str, Any] = {}
        self.last_saved_tick = -1
        self.last_saved_ts = 0.0
        self.saves = 0
        self.failures = 0

    # ------------------------------------------------------------------ (de)serialização sem pickle
    def _pack(self, obj, arrays: Dict[str, np.ndarray], prefix: str):
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                raise TypeError(f"array de objetos em {prefix} não é serializável sem pickle")
            name = f"a{len(arrays):05d}"
            arrays[name] = obj
            return {"__nd__": name}
        if isinstance(obj, dict):
            return {str(k): self._pack(v, arrays, f"{prefix}.{k}") for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._pack(v, arrays, f"{prefix}[]") for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, set):
            return sorted(obj)
        if hasattr(obj, "__dataclass_fields__"):
            return self._pack(asdict(obj), arrays, prefix)
        return obj

    def _unpack(self, obj, arrays: Dict[str, np.ndarray]):
        if isinstance(obj, dict):
            if "__nd__" in obj and len(obj) == 1:
                return arrays[obj["__nd__"]]
            return {k: self._unpack(v, arrays) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._unpack(v, arrays) for v in obj]
        return obj

    @staticmethod
    def _sha(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()

    def _write_shards(self, state: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
        os.makedirs(out_dir, exist_ok=True)
        shards = []
        for key, val in state.items():
            arrays: Dict[str, np.ndarray] = {}
            js = self._pack(val, arrays, key)
            jp = os.path.join(out_dir, f"{key}.json")
            with open(jp, "w") as f:
                json.dump(js, f)
            shards.append({"file": f"{key}.json", "key": key, "kind": "json"})
            if arrays:
                part, cur, files = 0, {}, []
                size = 0
                for name, arr in arrays.items():
                    cur[name] = arr; size += arr.nbytes
                    if size >= self.MAX_PART_BYTES:
                        fn = f"{key}.part{part:03d}.npz"; np.savez(os.path.join(out_dir, fn), **cur)
                        files.append(fn); part += 1; cur, size = {}, 0
                if cur:
                    fn = f"{key}.part{part:03d}.npz"; np.savez(os.path.join(out_dir, fn), **cur); files.append(fn)
                for fn in files:
                    shards.append({"file": fn, "key": key, "kind": "npz"})
        for sh in shards:
            p = os.path.join(out_dir, sh["file"])
            sh["sha256"] = self._sha(p); sh["bytes"] = os.path.getsize(p)
        manifest = {"checkpoint_version": CHECKPOINT_VERSION, "genesis_version": GENESIS_VERSION,
                    "versions": ENGINE_VERSIONS, "versions_hash": ENGINE_VERSIONS_HASH, "created": iso_now(),
                    "tick": int(state["meta"]["tick"]), "shards": shards,
                    "total_bytes": int(sum(s["bytes"] for s in shards))}
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        return manifest

    def verify(self, d: str) -> Tuple[bool, str]:
        mp = os.path.join(d, "manifest.json")
        if not os.path.exists(mp):
            return False, "manifest ausente"
        try:
            man = json.load(open(mp))
        except Exception as e:
            return False, f"manifest ilegível: {e}"
        for sh in man.get("shards", []):
            p = os.path.join(d, sh["file"])
            if not os.path.exists(p):
                return False, f"shard ausente: {sh['file']}"
            if os.path.getsize(p) != sh["bytes"] or self._sha(p) != sh["sha256"]:
                return False, f"hash divergente: {sh['file']}"
        return True, "ok"

    def read(self, d: str) -> Dict[str, Any]:
        man = json.load(open(os.path.join(d, "manifest.json")))
        ver = int(man.get("checkpoint_version", 0))
        if ver > CHECKPOINT_VERSION:
            raise RuntimeError(f"checkpoint v{ver} é mais novo que este motor (v{CHECKPOINT_VERSION}) — não carregado")
        if ver < 3:
            raise RuntimeError(f"checkpoint v{ver} é do Genesis Ω (2D): importe como arquivo legado, não como mundo")
        by_key: Dict[str, Dict[str, Any]] = collections.defaultdict(dict)
        for sh in man["shards"]:
            by_key[sh["key"]].setdefault("files", []).append(sh)
        state = {}
        for key, info in by_key.items():
            arrays: Dict[str, np.ndarray] = {}
            js = None
            for sh in info["files"]:
                p = os.path.join(d, sh["file"])
                if sh["kind"] == "json":
                    js = json.load(open(p))
                else:
                    with np.load(p, allow_pickle=False) as z:
                        for k in z.files:
                            arrays[k] = z[k]
            state[key] = self._unpack(js, arrays)
        for v in sorted(self.MIGRATIONS):
            if ver < v:
                state = self.MIGRATIONS[v](state)
        state["_manifest"] = man
        return state

    # ------------------------------------------------------------------ escrita atômica
    def save(self, uni: "UniverseEngine", sync: bool = False, reason: str = "periodic") -> bool:
        if self._thread is not None and self._thread.is_alive():
            if not sync:
                return False
            self._thread.join()
        t0 = time.time()
        state = uni.state_dict()
        snap_t = time.time() - t0
        visual = self.visual_snapshot(uni)
        tick = uni.tick_count

        def work():
            try:
                self._write(state, tick, visual, reason, snap_t)
            except Exception as e:
                self.failures += 1
                tb = traceback.format_exc()
                if self.logger:
                    self.logger.event("checkpoint_failed", echo=True, error=repr(e), trace=tb[-2000:])
        if sync:
            work()
        else:
            self._thread = threading.Thread(target=work, name="genesis-checkpoint", daemon=True)
            self._thread.start()
        return True

    def _write(self, state, tick, visual, reason, snap_t):
        t0 = time.time()
        cid = f"ckpt_{tick:010d}_{int(time.time())}"
        local = self.paths.tmp(cid)
        man = self._write_shards(state, local)
        ok, why = self.verify(local)
        if not ok:
            raise RuntimeError(f"verificação local falhou: {why}")
        tmp = os.path.join(self.root, f".tmp_{cid}")
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        shutil.copytree(local, tmp)
        for sh in man["shards"]:                                     # conferência pós-cópia (tamanho + hash)
            p = os.path.join(tmp, sh["file"])
            if os.path.getsize(p) != sh["bytes"]:
                raise RuntimeError(f"cópia truncada: {sh['file']}")
        ok, why = self.verify(tmp)
        if not ok:
            raise RuntimeError(f"verificação no destino falhou: {why}")
        cur, prev = os.path.join(self.root, "current"), os.path.join(self.root, "previous")
        if os.path.exists(prev):
            man_prev = json.load(open(os.path.join(prev, "manifest.json"))) if os.path.exists(os.path.join(prev, "manifest.json")) else {}
            dst = os.path.join(self.backups, f"ckpt_{int(man_prev.get('tick', 0)):010d}_{int(time.time())}")
            os.replace(prev, dst)
        if os.path.exists(cur):
            os.replace(cur, prev)
        os.replace(tmp, cur)
        shutil.rmtree(local, ignore_errors=True)
        bk = sorted(glob.glob(os.path.join(self.backups, "ckpt_*")))
        for old in bk[:-self.keep]:
            shutil.rmtree(old, ignore_errors=True)
        if visual is not None:
            snapdir = self.paths.sub("snapshots")
            np.savez_compressed(os.path.join(snapdir, "last_world.npz"), **{k: v for k, v in visual.items() if isinstance(v, np.ndarray)})
            with open(os.path.join(snapdir, "last_world.json"), "w") as f:
                json.dump(to_jsonable({k: v for k, v in visual.items() if not isinstance(v, np.ndarray)}), f)
        self.saves += 1
        self.last_saved_tick, self.last_saved_ts = tick, time.time()
        self.last_info = {"tick": tick, "ts": iso_now(), "bytes": man["total_bytes"], "shards": len(man["shards"]),
                          "seconds": round(time.time() - t0, 2), "snapshot_seconds": round(snap_t, 2), "reason": reason}
        if self.logger:
            self.logger.event("checkpoint", **self.last_info)
        self._storage_pressure()

    def wait(self):
        if self._thread is not None:
            self._thread.join()

    def _storage_pressure(self):
        try:
            free = shutil.disk_usage(self.root).free
        except Exception:
            return
        if free < 2 * 1024 ** 3:
            bk = sorted(glob.glob(os.path.join(self.backups, "ckpt_*")))
            for old in bk[:-1]:
                shutil.rmtree(old, ignore_errors=True)
            for f in glob.glob(os.path.join(self.paths.work, "tmp", "*")):
                shutil.rmtree(f, ignore_errors=True) if os.path.isdir(f) else os.remove(f)
            if self.logger:
                self.logger.event("storage_pressure", echo=True, free_bytes=free,
                                  action="pruned old backups and temporary files; checkpoints, champions and discoveries preserved")

    # ------------------------------------------------------------------ carga com recuperação
    def candidates(self) -> List[str]:
        out = [os.path.join(self.root, "current"), os.path.join(self.root, "previous")]
        out += sorted(glob.glob(os.path.join(self.backups, "ckpt_*")), reverse=True)
        return [d for d in out if os.path.isdir(d)]

    def load_latest(self) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        info: Dict[str, Any] = {"tried": []}
        cands = self.candidates()
        for d in cands:
            ok, why = self.verify(d)
            info["tried"].append({"dir": os.path.basename(d), "ok": ok, "why": why})
            if not ok:
                if self.logger:
                    self.logger.event("checkpoint_corrupted", echo=True, dir=d, reason=why)
                continue
            try:
                st = self.read(d)
                info.update({"loaded": d, "tick": st["_manifest"]["tick"], "fallback": d != cands[0]})
                return st, info
            except Exception as e:
                info["tried"][-1].update({"ok": False, "why": repr(e)})
                if self.logger:
                    self.logger.event("checkpoint_unreadable", echo=True, dir=d, error=repr(e))
        if cands:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            keep = os.path.join(self.root, f"corrupted_{stamp}")
            os.makedirs(keep, exist_ok=True)
            for d in cands:
                shutil.move(d, os.path.join(keep, os.path.basename(d)))
            info["corrupted_preserved"] = keep
            info["warning"] = ("NENHUM checkpoint válido: arquivos preservados em " + keep +
                               " — um novo mundo será criado (aviso explícito, não é reset silencioso)")
        return None, info

    def visual_snapshot(self, uni: "UniverseEngine") -> Optional[Dict[str, Any]]:
        """'Last known world': estado visual compacto para o observatório quando a simulação estiver offline."""
        try:
            ws, pool = uni.ws, uni.pool
            step = max(1, ws.W // 256)
            idx = torch.nonzero(pool.alive).flatten()[:20000]
            return {"height": ws.height[::step, ::step].cpu().numpy().astype(np.float16),
                    "water": (ws.is_water & ~ws.filled)[::step, ::step].cpu().numpy(),
                    "struct": ws.struct_h[::step, ::step].cpu().numpy().astype(np.int8),
                    "agents": pool.pos[idx].cpu().numpy().astype(np.float32),
                    "colors": pool.color[idx].cpu().numpy().astype(np.float16),
                    "tick": int(uni.tick_count), "ts": iso_now(), "status": uni.status(), "step": step}
        except Exception as e:
            if self.logger:
                self.logger.event("visual_snapshot_failed", error=repr(e))
            return None
