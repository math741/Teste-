# %% [markdown]
# ## 20 — TEST SUITE (boot): unidade, integração, reprodutibilidade, checkpoint, protocolo, controles + e −
# Todos os testes usam o código real (nada de mocks de ciência). Payloads sintéticos só aparecem nos testes de
# protocolo/pipeline, explicitamente marcados como TEST MODE, e nunca entram na pesquisa.

# %%
class BootTests:
    def __init__(self, cfg: CivConfig, paths: CivPaths, device: torch.device, logger: EventLogger):
        self.cfg = copy.deepcopy(cfg)
        self.cfg.world_w = self.cfg.world_h = 48
        self.cfg.initial_agents, self.cfg.max_agents, self.cfg.max_bodies = 60, 200, 6
        self.cfg.probe.trials, self.cfg.probe.permutations = 8, 30
        self.paths, self.device, self.logger = paths, device, logger
        self.results: List[Dict[str, Any]] = []

    def _mk(self):
        L = GenomeLayout(self.cfg).to(self.device)
        we = WorldEngine(self.cfg, self.device); matter = MatterEngine(self.cfg, self.device)
        dev = DevelopmentEngine(self.cfg, L, self.device); body = BodyEngine(self.cfg, dev, self.device)
        mind = SignalEngine(self.cfg, L, self.device)
        return L, we, matter, dev, body, mind

    def run(self) -> List[Dict[str, Any]]:
        tests = [m for m in dir(self) if m.startswith("test_")]
        for name in sorted(tests):
            t0 = time.time()
            try:
                info = getattr(self, name)() or {}
                self.results.append({"test": name, "ok": True, "seconds": round(time.time() - t0, 2), **to_jsonable(info)})
            except Exception as e:
                self.results.append({"test": name, "ok": False, "seconds": round(time.time() - t0, 2), "error": repr(e),
                                     "trace": traceback.format_exc()[-1200:]})
            live_progress("testes", len(self.results), len(tests))
        return self.results

    # ------------------------------------------------------------------ mundo e matéria
    def test_world_deterministic(self):
        L, we, *_ = self._mk()
        a, b = we.generate(11), we.generate(11)
        assert torch.equal(a.height, b.height) and torch.equal(a.is_water, b.is_water), "mundo não determinístico"
        assert not torch.equal(a.height, we.generate(12).height), "sementes diferentes geraram o mesmo mundo"
        return {"water_fraction": float(a.is_water.float().mean())}

    def test_material_conservation(self):
        L, we, matter, dev, body, mind = self._mk()
        ws = we.generate(3); eng = AgentEngine(self.cfg, L, we, matter, mind, self.device)
        pool = AgentPool(self.cfg.max_agents, L, self.cfg, self.device); ids = IdCounter(1)
        ledger = StructureLedger(self.cfg.max_blocks, self.device)
        g = torch.Generator(device=self.device); g.manual_seed(5)
        eng.spawn_founders(ws, pool, 60, g, ids, 0, None)
        m0 = matter.total_material(ws, pool.carried[pool.alive])
        for t in range(150):
            eng.step(ws, pool, t, g, None, ids, ledger)
            if t % 4 == 0:
                ws.tick = t; iy, ix = WorldEngine.tile_index(ws, pool.pos[pool.alive]); we.env_step(ws, g, torch.stack([iy, ix]))
        m1 = matter.total_material(ws, pool.carried[pool.alive])
        assert abs(m1 - m0) < 1e-3, f"matéria não conservada: {m0} → {m1}"
        return {"material": m0, "placed": int(ledger.total_placed)}

    # ------------------------------------------------------------------ desenvolvimento e física
    def test_development_deterministic_and_batch_independent(self):
        L, we, matter, dev, body, mind = self._mk()
        g = torch.Generator(device=self.device); g.manual_seed(7)
        G = L.random(5, g); sd = torch.arange(5, device=self.device) + 100
        full = dev.develop(G, sd).occ
        single = torch.stack([dev.develop(G[i:i + 1], sd[i:i + 1]).occ[0] for i in range(5)])
        assert torch.equal(full, single), "desenvolvimento depende da composição do lote"
        assert torch.equal(full, dev.develop(G, sd).occ), "desenvolvimento não determinístico"
        return {"nodes": full.sum(1).tolist()}

    def test_development_control_genomes(self):
        """Controles conhecidos: crescimento forte ocupa o orçamento máximo; limiar inatingível mantém só a semente."""
        L, we, matter, dev, body, mind = self._mk()
        g = torch.Generator(device=self.device); g.manual_seed(1)
        G = L.random(2, g)
        for name, val in [("dev_gw", 0.0), ("dev_noise", 0.0), ("dev_maxfrac", 0.5)]:
            a, b = L.slices[name]; G[:, a:b] = val
        a, b = L.slices["dev_gthr"]; G[0, a:b] = -1.0; G[1, a:b] = 1.0
        bp = dev.develop(L.clamp(G), torch.tensor([1, 1], device=self.device))
        n = bp.occ.sum(1).tolist()
        assert n[0] == math.ceil(0.5 * dev.N) or n[0] >= 0.45 * dev.N, f"controle positivo não cresceu: {n[0]}"
        assert n[1] == 1, f"controle negativo cresceu: {n[1]}"
        return {"positive_nodes": n[0], "negative_nodes": n[1]}

    def test_physics_stability(self):
        L, we, matter, dev, body, mind = self._mk()
        g = torch.Generator(device=self.device); g.manual_seed(3)
        G = passive_body(L, L.random(6, g))
        bb = body.make(dev.develop(G, torch.arange(6, device=self.device)))
        z0 = bb.com()[:, 2].clone()
        for _ in range(60):
            r = body.step(bb, torch.zeros(6, 2, device=self.device))
        assert bool(r["finite"].all()), "física divergiu"
        assert float(bb.vel.abs().max()) < 5.0, "corpo passivo não estabilizou"
        assert bool((bb.com()[:, 2] <= z0 + 0.05).all()), "gravidade/contato inconsistentes"
        return {"max_speed_after_settle": float(bb.vel.norm(dim=-1).max())}

    def test_spatial_hash_matches_bruteforce(self):
        g = torch.Generator(device=self.device); g.manual_seed(2)
        pos = torch.rand(300, 2, generator=g, device=self.device) * 40
        alive = torch.ones(300, dtype=torch.bool, device=self.device)
        nn, nd, _ = spatial_neighbors(pos, alive, 4.0, M=64)
        # referência EXATA (diferenças elemento a elemento; torch.cdist usa matmul e perde precisão em float32 na GPU)
        D = (pos[:, None, :] - pos[None, :, :]).norm(dim=-1); D.fill_diagonal_(1e9)
        bd, bi = D.min(1)
        ok = (bd > 4.0) | (bi == nn) | ((nd - bd).abs() <= 1e-4 * (1 + bd))
        assert bool(ok.all()), "hash espacial divergiu da força bruta"
        return {"pairs_checked": 300}

    # ------------------------------------------------------------------ controles positivos / negativos dos probes
    def test_memory_controls(self):
        """Controle positivo (unidade auto-recorrente lenta ligada à pista) memoriza; controle sem memória não."""
        L, we, matter, dev, body, mind = self._mk()
        runner = ArenaRunner(self.cfg, L, we, matter, mind, dev, body, self.device)
        ve = ValidationEngine(self.cfg, L, runner, AffectEngine(self.cfg, L, runner, EthicalSentinel(self.cfg), self.device), self.device)
        g = torch.Generator(device=self.device); g.manual_seed(4)
        G = L.random(1, g)
        N = self.cfg.n_units
        for name in ("ctl_Win", "ctl_W0", "ctl_b", "plast_eta", "babble_sigma"):
            a, b = L.slices[name]; G[:, a:b] = 0.0
        Win = L.view(G, "ctl_Win"); Win[0, 0, 12] = 2.0; Win[0, 0, 13] = -2.0
        W0 = L.view(G, "ctl_W0"); W0[0, 0, 0] = 1.6
        a, b = L.slices["ctl_logtau"]; G[:, a:b] = 2.3
        n = 32
        delays = [1, 8]
        pos_r = drive(ve._memory_decode(G.expand(n, -1).contiguous(), 99, delays, n, "test memory +"))
        neg_r = drive(ve._memory_decode(memoryless(L, G.expand(n, -1).contiguous()), 99, delays, n, "test memory −"))
        assert pos_r[8]["linear_p"] < 0.01 and pos_r[8]["linear_acc"] > 0.8, f"controle positivo falhou: {pos_r[8]['linear_acc']}"
        assert neg_r[8]["linear_acc"] < 0.75, f"controle negativo memorizou: {neg_r[8]['linear_acc']}"
        return {"positive_acc": pos_r[8]["linear_acc"], "negative_acc": neg_r[8]["linear_acc"]}

    def test_communication_mi_controls(self):
        rng = np.random.default_rng(0)
        state = rng.integers(0, 3, 600)
        sig = np.where(rng.random(600) < 0.85, state, rng.integers(0, 3, 600))          # TEST MODE: sinal sintético
        mi, p = mi_permutation(sig, state, 40, rng)
        mi0, p0 = mi_permutation(rng.permutation(sig), state, 40, rng)
        assert p < 0.05 and mi > 0.3, "controle positivo de comunicação não detectado"
        assert p0 > 0.05, "controle negativo de comunicação deu falso positivo"
        return {"mi_positive": mi, "p_negative": p0}

    def test_affect_pipeline_controls(self):
        """Controle sintético do pipeline de decodificação de valência (não entra na evolução)."""
        rng = np.random.default_rng(1)
        y = np.repeat([0, 1], 40)
        X = rng.normal(size=(80, 16)); X[:, 0] += (y * 2 - 1) * 1.5                          # TEST MODE
        r = decode_report(X, y, 2, 30, 4, 0, nonlinear=False)
        r0 = decode_report(rng.normal(size=(80, 16)), y, 2, 30, 4, 0, nonlinear=False)
        assert r["linear_acc"] > 0.8 and r["linear_p"] < 0.01, "pipeline não detectou sinal sintético"
        assert r0["linear_p"] > 0.01, "pipeline deu falso positivo"
        return {"acc_positive": r["linear_acc"], "acc_negative": r0["linear_acc"]}

    def test_dsl_security(self):
        bad = [{"name": "x", "target": {"kind": "candidate", "id": "C1"}, "steps": [{"op": "exec", "code": "import os"}]},
               {"name": "x", "target": {"kind": "candidate", "id": "C1"}, "steps": [{"op": "observe", "ticks": 10 ** 9}]},
               {"name": "__import__('os')", "target": {"kind": "candidate", "id": "C1"}, "steps": [{"op": "observe", "ticks": 5}]},
               {"name": "x", "target": {"kind": "host"}, "steps": [{"op": "observe", "ticks": 5}]},
               {"name": "x", "target": {"kind": "candidate", "id": "C1"}, "steps": [{"op": "ablate", "what": "action", "channel": "rm -rf"}]}]
        for b in bad:
            try:
                validate_dsl(b)
                raise AssertionError(f"DSL aceitou programa perigoso: {b}")
            except ValueError:
                pass
        ok = validate_dsl({"name": "ok", "target": {"kind": "candidate", "id": "C1"},
                           "steps": [{"op": "ablate", "what": "action", "channel": "place_material"}, {"op": "observe", "ticks": 50},
                                     {"op": "measure", "metric": "intake"}]})
        return {"accepted_steps": len(ok["steps"]), "rejected": len(bad)}

    # ------------------------------------------------------------------ integração: universo, checkpoint, observação
    def _small_universe(self, root: str, device: Optional[torch.device] = None):
        p = CivPaths(root, os.path.join(root, "_work"), on_drive=False)
        cfg = copy.deepcopy(self.cfg)
        cfg.n_islands, cfg.island_pop, cfg.qd_batch = 1, 6, 6
        u = UniverseEngine(cfg, p, device or self.device, None)
        u.create_world(bootstrap_seconds=0.5)
        return u, p

    def test_checkpoint_roundtrip_and_corruption_fallback(self):
        root = tempfile.mkdtemp(prefix="gtest_ck_", dir=self.paths.tmp(""))
        u, p = self._small_universe(root)
        ck = CheckpointEngine(p, None)
        for _ in range(20):
            u.tick()
        ck.save(u, sync=True)
        t_first = u.tick_count
        for _ in range(20):
            u.tick()
        ck.save(u, sync=True)
        st, info = ck.load_latest()
        assert st is not None and st["meta"]["tick"] == u.tick_count, "roundtrip falhou"
        u2 = UniverseEngine(u.cfg, p, self.device, None); u2.load_state_dict(st)
        assert u2.pool.n_alive() == u.pool.n_alive() and torch.allclose(u2.pool.pos[:u.pool.n_alive()].cpu(), u.pool.pos[u.pool.alive].cpu())
        shard = next(f for f in glob.glob(os.path.join(p.root, "checkpoints", "current", "*.npz")))
        with open(shard, "r+b") as f:
            f.seek(64); f.write(b"\x00CORRUPTED\x00")
        st2, info2 = ck.load_latest()
        assert st2 is not None and st2["meta"]["tick"] == t_first and info2.get("fallback"), "não recorreu ao previous"
        shutil.rmtree(root, ignore_errors=True)
        return {"fallback_to_tick": t_first, "tried": [t["dir"] for t in info2["tried"]]}

    def test_observation_independence(self):
        """O observador (telemetria + gêmeo visual + camadas) não altera o estado nem o RNG da simulação.
        Roda na CPU (bit-a-bit determinística): na GPU, somas atômicas (index_add_) já divergem sem observador nenhum,
        o que mascararia o efeito testado. A lógica do observador é a mesma em qualquer dispositivo."""
        cpu = torch.device("cpu")
        root = tempfile.mkdtemp(prefix="gtest_obs_", dir=self.paths.tmp(""))
        u, p = self._small_universe(root, cpu)
        st = u.state_dict()
        ua = UniverseEngine(u.cfg, p, cpu, None); ua.load_state_dict(copy.deepcopy(st))
        ub = UniverseEngine(u.cfg, p, cpu, None); ub.load_state_dict(copy.deepcopy(st))
        desk = ObserverDesk(); tel = TelemetryEngine(ub, desk)
        desk.layers["t"] = "population"
        aid = int(ub.pool.aid[ub.pool.alive][0]); desk.selected["t"] = aid
        for _ in range(30):
            ua.tick(); ub.tick(); tel.pump(force=True)
        assert torch.equal(ua.pool.pos, ub.pool.pos) and torch.equal(ua.gen.get_state(), ub.gen.get_state()), \
            "observação alterou a simulação"
        shutil.rmtree(root, ignore_errors=True)
        return {"ticks": 30, "frames": tel.frames_sent}

    def test_replay_determinism(self):
        """Replay checkpoint + semente → mesma trajetória (CPU, bit-a-bit). Na GPU, kernels com soma atômica
        (index_add_/scatter) são registrados como não determinísticos bit-a-bit (ver relatório/limitações)."""
        cpu = torch.device("cpu")
        root = tempfile.mkdtemp(prefix="gtest_det_", dir=self.paths.tmp(""))
        u, p = self._small_universe(root, cpu)
        st = u.state_dict()
        hashes = []
        for _ in range(2):
            v = UniverseEngine(u.cfg, p, cpu, None); v.load_state_dict(copy.deepcopy(st))
            for _ in range(25):
                v.tick()
            hashes.append(hash_obj(v.pool.pos[v.pool.alive].cpu().numpy()))
        shutil.rmtree(root, ignore_errors=True)
        assert hashes[0] == hashes[1], "replay a partir do mesmo checkpoint divergiu"
        note = "GPU: index_add_/scatter usam somas atômicas (não bit-a-bit determinísticas)" if self.device.type == "cuda" else ""
        return {"deterministic_cpu": True, "note": note}

    def test_multiscale_aggregation_bookkeeping(self):
        root = tempfile.mkdtemp(prefix="gtest_f0_", dir=self.paths.tmp(""))
        u, p = self._small_universe(root)
        n0 = u.pool.n_alive()
        idx = torch.nonzero(u.pool.alive).flatten()[:10]
        regs = u.f0.region_of(u.pool.pos[idx].cpu().numpy())
        u.f0.absorb(regs, u.pool.genome[idx].cpu().numpy(), u.pool.energy[idx].cpu().numpy(), u.pool.signature[idx].cpu().numpy(), u.inst_rng)
        u.agents.kill(u.pool, idx, u.bodies, u.ws)
        assert abs(u.f0.total() - 10) < 1e-6 and u.pool.n_alive() == n0 - 10, "agregação perdeu indivíduos"
        made = u._disaggregate(10)
        assert made == 10 and abs(u.f0.total()) < 1e-6 and u.pool.n_alive() == n0, "desagregação inconsistente"
        shutil.rmtree(root, ignore_errors=True)
        return {"aggregated": 10, "disaggregated": made}

    def test_oom_handler(self):
        e = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert is_oom_error(e)
        cfg = copy.deepcopy(self.cfg)
        before = (cfg.qd_batch, cfg.max_stream_agents, cfg.probe.trials)
        oom_backoff(cfg)
        after = (cfg.qd_batch, cfg.max_stream_agents, cfg.probe.trials)
        assert all(a <= b for a, b in zip(after, before)) and (after != before or before[0] <= 4), "backoff não reduziu carga"
        return {"before": before, "after": after}

    def test_server_protocol(self):
        if not HAVE_WEB:
            return {"skipped": "fastapi/uvicorn ausentes"}
        import urllib.request, urllib.error
        static = os.path.join(self.paths.work, "viewer", "static")
        if not os.path.exists(os.path.join(static, "index.html")):
            Genesis3DViewer(self.paths, None).install()
        desk = ObserverDesk()
        desk.base = pack_msg({"t": "base", "test_mode": True}); desk.base_seq = 1             # TEST MODE (payload marcado)
        desk.publish_frame(pack_msg({"t": "frame", "test_mode": True, "n": 0}))
        with socket.socket() as s_:
            s_.bind(("127.0.0.1", 0)); port = s_.getsockname()[1]
        srv = RealtimeServer(desk, lambda: None, self.paths, port, True, None, static)
        assert srv.start(), f"servidor não iniciou: {srv.error}"
        base = f"http://127.0.0.1:{port}"

        def code(path, headers=None, data=None):
            req = urllib.request.Request(base + path, headers=headers or {}, data=data, method="POST" if data is not None else "GET")
            try:
                return urllib.request.urlopen(req, timeout=5).status
            except urllib.error.HTTPError as e:
                return e.code
        try:
            assert code("/health") == 200
            assert code("/api/status") == 401, "API sem token deveria recusar"
            assert code("/api/status", {"X-Genesis-Token": srv.view_token}) == 200
            assert code("/api/admin/pause", {"X-Genesis-Token": srv.view_token}, b"{}") == 401, "admin com token de visualização"
            assert code("/api/admin/pause", {"X-Genesis-Admin": srv.admin_token, "cf-connecting-ip": "1.2.3.4"}, b"{}") == 403, \
                "admin pelo túnel deveria ser recusado em REMOTE_READ_ONLY"
            assert code("/api/admin/pause", {"X-Genesis-Admin": srv.admin_token}, b"{}") == 200
            assert code("/static/../../etc/passwd") in (404, 422), "path traversal"
            assert code("/api/agent/-5", {"X-Genesis-Token": srv.view_token}) == 422
            got = run_async_isolated(self._ws_probe(port, srv.view_token))
            assert got["hello"] and got["binary"] >= 1, f"WebSocket não entregou hello/base: {got}"
            bad = run_async_isolated(self._ws_probe(port, "wrong-token"))
            assert not bad["hello"], "WebSocket aceitou token errado"
        finally:
            srv.stop()
        return {"port": port, "ws": got}

    @staticmethod
    async def _ws_probe(port: int, tok: str) -> Dict[str, Any]:
        out = {"hello": False, "binary": 0}
        try:
            import websockets
        except ImportError:
            ensure_package("websockets", "websockets>=10,<16")
            import websockets
        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws/live?t={tok}", open_timeout=5) as w:
                for _ in range(4):
                    m = await asyncio.wait_for(w.recv(), timeout=3)
                    if isinstance(m, str):
                        out["hello"] = out["hello"] or json.loads(m).get("t") == "hello"
                    else:
                        msgpack.unpackb(m, raw=False); out["binary"] += 1
        except Exception as e:
            out["error"] = repr(e)
        return out


def run_async_isolated(coro, timeout: float = 30.0):
    """Executa uma corrotina num event loop próprio em outra thread (Jupyter/Colab já têm um loop rodando)."""
    box: Dict[str, Any] = {}

    def target():
        try:
            box["result"] = asyncio.run(coro)
        except Exception as e:
            box["error"] = e
    th = threading.Thread(target=target, daemon=True, name="genesis-async-test")
    th.start(); th.join(timeout)
    if "error" in box:
        raise box["error"]
    if "result" not in box:
        raise TimeoutError("corrotina não terminou a tempo")
    return box["result"]


def oom_backoff(cfg: CivConfig) -> CivConfig:
    """Recuperação de OOM: reduz lotes e streaming (o viewer perde LOD antes da ciência perder rigor)."""
    cfg.qd_batch = max(4, cfg.qd_batch // 2)
    cfg.max_stream_agents = max(300, cfg.max_stream_agents // 2)
    cfg.probe.trials = max(6, cfg.probe.trials - 2)
    return cfg


def run_boot_tests(cfg, paths, device, logger) -> Tuple[bool, List[Dict[str, Any]]]:
    bt = BootTests(cfg, paths, device, logger)
    res = bt.run()
    ok = all(r["ok"] for r in res)
    rows = "".join(f"<tr><td>{'✅' if r['ok'] else '❌'}</td><td>{r['test'][5:]}</td><td>{r['seconds']}s</td>"
                   f"<td style='color:#e88'>{_html.escape(r.get('error', '')[:160])}</td></tr>" for r in res)
    display(HTML(f"<details {'open' if not ok else ''}><summary style='font-family:system-ui'>Testes de boot: "
                 f"{sum(r['ok'] for r in res)}/{len(res)} OK</summary><table style='font-family:system-ui;font-size:12px'>{rows}</table></details>"))
    with open(os.path.join(paths.reports, "boot_tests.json"), "w") as f:
        json.dump(to_jsonable(res), f, indent=1)
    logger.event("boot_tests", ok=ok, passed=sum(r["ok"] for r in res), total=len(res))
    return ok, res
