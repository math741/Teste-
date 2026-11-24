# %% [markdown]
# ## 03 — Ponte com o Genesis Ω original (pesquisa preservada, nunca reinterpretada)
# O motor 2D original continua disponível como módulo `genesis_omega_legacy` (código embutido, versão exata).
# * **Importação**: o checkpoint antigo (`MyDrive/GENESIS_OMEGA`) é lido e resumido num arquivo separado
#   (`legacy/`), com descobertas, níveis e evidência. Nada disso vira "agente com corpo".
# * **Reprodução**: `reproduce_legacy_discovery(id)` reexecuta uma descoberta antiga com o motor antigo.
# * **Ponte de substrato** (Discovery Bridge): uma lei 2D antiga pode dirigir um campo ambiental de um *fork* do mundo
#   novo (experimento de substrato), para comparar leis físicas como base de desenvolvimento e ecologia.

# %%
LEGACY_GENESIS_B64 = "@@LEGACY_B64@@"


class LegacyBridge:
    MODULE_NAME = "genesis_omega_legacy"

    def __init__(self, paths: "CivPaths", logger: "EventLogger"):
        self.paths, self.logger = paths, logger
        self._module = None

    def source(self) -> str:
        if LEGACY_GENESIS_B64.startswith("@@"):
            raise RuntimeError("código legado não embutido (build incompleto)")
        return base64.b64decode(LEGACY_GENESIS_B64).decode("utf-8")

    def code_path(self) -> str:
        src = self.source()
        p = os.path.join(self.paths.work, "legacy_code", f"{self.MODULE_NAME}.py")
        if not os.path.exists(p) or hashlib.sha256(open(p, "rb").read()).hexdigest() != hashlib.sha256(src.encode()).hexdigest():
            with open(p, "w", encoding="utf-8") as f:
                f.write(src)
        return p

    def load_module(self):
        if self._module is None:
            spec = importlib.util.spec_from_file_location(self.MODULE_NAME, self.code_path())
            mod = importlib.util.module_from_spec(spec)
            with contextlib.redirect_stdout(io.StringIO()):
                spec.loader.exec_module(mod)
            sys.modules[self.MODULE_NAME] = mod
            self._module = mod
        return self._module

    def candidate_checkpoints(self) -> List[str]:
        roots = ["/content/drive/MyDrive/GENESIS_OMEGA", "/content/GENESIS_OMEGA", os.path.abspath("./genesis_runs")]
        extra = os.environ.get("GENESIS_LEGACY_ROOT", "")
        if extra:
            roots.insert(0, extra)
        out = []
        for r in roots:
            for name in ["checkpoint_current.pt", "checkpoint_previous.pt"]:
                p = os.path.join(r, "checkpoints", name)
                if os.path.exists(p):
                    out.append(p)
        return out

    @staticmethod
    def _load(path: str) -> dict:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def import_research(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """Resumo honesto da pesquisa 2D anterior. Idempotente (marcador com hash do checkpoint)."""
        cands = self.candidate_checkpoints()
        if not cands:
            return None
        path = cands[0]
        digest = hashlib.sha256(f"{path}|{os.path.getsize(path)}|{os.path.getmtime(path)}".encode()).hexdigest()[:16]
        marker = os.path.join(self.paths.legacy, "IMPORTED.json")
        if os.path.exists(marker) and not force:
            m = json.load(open(marker))
            if m.get("digest") == digest and os.path.exists(m.get("summary_path", "")):
                return json.load(open(m["summary_path"]))
        st = None
        for p in cands:
            try:
                st = self._load(p); path = p
                break
            except Exception as e:
                self.logger.event("legacy_checkpoint_unreadable", path=p, error=repr(e))
        if st is None:
            return None
        levels = st.get("levels", {})
        hist = collections.Counter(int(v) for v in levels.values())
        disc = []
        for did, rec in st.get("discoveries", {}).items():
            disc.append({"discovery_id": did, "candidate_id": rec.get("candidate_id"), "level": rec.get("new_level"),
                         "level_name": rec.get("level_name"), "probe": rec.get("probe"), "score": rec.get("score"),
                         "p_value": rec.get("p_value"), "generation": rec.get("generation"), "engine": rec.get("engine")})
        summary = {"source_checkpoint": path, "legacy_genesis_version": st.get("genesis_version"),
                   "legacy_checkpoint_version": st.get("checkpoint_version"), "counters": st.get("counters", {}),
                   "best_level_global": st.get("best_level_global"), "level_histogram": dict(hist),
                   "discoveries": disc, "champion": to_jsonable(st.get("champion")),
                   "pantheon_size": len(st.get("pantheon", [])), "config_profile": st.get("config", {}).get("profile"),
                   "imported_at": iso_now(), "digest": digest,
                   "note": "Pesquisa Genesis Ω (campos 2D). Descobertas computacionais de substrato; não possuem corpo, "
                           "agentes nem civilização. Preservadas como arquivo separado."}
        src_disc = os.path.join(os.path.dirname(os.path.dirname(path)), "discoveries")
        if os.path.isdir(src_disc):
            shutil.copytree(src_disc, os.path.join(self.paths.legacy, "discoveries"), dirs_exist_ok=True)
        spath = os.path.join(self.paths.legacy, "legacy_genesis_omega_summary.json")
        with open(spath, "w") as f:
            json.dump(to_jsonable(summary), f, indent=1)
        with open(marker, "w") as f:
            json.dump({"digest": digest, "summary_path": spath, "ts": iso_now()}, f)
        self.logger.event("legacy_imported", discoveries=len(disc), best_level=summary["best_level_global"])
        return summary

    def legacy_lab(self, path: Optional[str] = None):
        """Reconstrói o laboratório 2D antigo a partir do checkpoint (motor antigo, configuração travada)."""
        mod = self.load_module()
        path = path or (self.candidate_checkpoints() or [None])[0]
        if path is None:
            raise FileNotFoundError("nenhum checkpoint Genesis Ω encontrado")
        st = self._load(path)
        st = mod.migrate_checkpoint(st)
        cfg = mod.GenesisConfig.from_dict(st["config"])
        lpaths = mod.GenesisPaths(os.path.join(self.paths.work, "legacy_runtime"), on_drive=False)
        lab = mod.GenesisLab(cfg, mod.detect_hardware(), lpaths, mod.EventLogger(lpaths, verbose=False),
                             torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        lab.load_state_dict(st)
        return mod, lab

    def reproduce_legacy_discovery(self, discovery_id: str, path: Optional[str] = None) -> Dict[str, Any]:
        mod, lab = self.legacy_lab(path)
        with contextlib.redirect_stdout(io.StringIO()):
            out = mod.reproduce_discovery(discovery_id, lab)
        self.logger.event("legacy_reproduction", discovery_id=discovery_id, status_match=out.get("status_match"))
        return out


class LegacySubstrateField:
    """Discovery Bridge: uma lei 2D antiga evolui um campo sobre o mapa do mundo; a magnitude modula a regeneração de
    recursos de um FORK experimental. Instrumento de comparação de substratos, nunca usado no mundo principal sem marca."""

    def __init__(self, bridge: LegacyBridge, candidate_id: str, world_hw: Tuple[int, int], device: torch.device,
                 checkpoint: Optional[str] = None, field_size: int = 64):
        mod, lab = bridge.legacy_lab(checkpoint)
        if candidate_id not in lab.store:
            raise KeyError(f"candidato legado {candidate_id} não está no checkpoint")
        self.mod, self.genome = mod, lab.store[candidate_id]
        self.candidate_id = candidate_id
        self.sim = mod.Simulator(lab.cfg, device)
        self.gb = self.sim.prepare([self.genome], field_size)
        env = lab.envs[0]
        self.st = self.sim.init_state(self.gb, [env], [derive_seed("substrate", candidate_id)], run_seed=1)
        self.world_hw = world_hw

    def step(self, n: int = 1) -> torch.Tensor:
        for _ in range(n):
            self.st = self.sim.step(self.gb, self.st)
        mag = self.st.U[0].abs().sum(0, keepdim=True).unsqueeze(0)
        up = F.interpolate(mag, size=self.world_hw, mode="bilinear", align_corners=False)[0, 0]
        return (up / (up.max() + 1e-9)).clamp(0, 1)
