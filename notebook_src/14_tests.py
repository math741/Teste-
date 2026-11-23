# %% [markdown]
# ## 14. Suíte de testes de boot (teste do teste)
# Roda **antes** da pesquisa. Cada instrumento científico tem controle positivo e negativo.
# Se qualquer teste falhar: `GENESIS BOOT FAILED` + stack trace, e a pesquisa não inicia.

# %%
def _boot_config(device: torch.device) -> GenesisConfig:
    cfg = profile_config("DEBUG", {"device": device.type}, 1234, 1.0, 0)
    cfg.world_size, cfg.population_size, cfg.batch_size = 32, 8, 16
    cfg.screen_steps, cfg.screen_burn_in, cfg.record_every = 48, 16, 4
    cfg.probe.trials, cfg.probe.permutations, cfg.probe.memory_delays = 8, 30, [1, 10, 20]
    cfg.probe.replication_seeds, cfg.probe.robust_trials = 2, 2
    cfg.max_experiments_per_generation = 2
    return cfg


class BootSuite:
    def __init__(self, device: torch.device, root: str):
        self.device, self.root = device, root
        self.results: List[Tuple[str, bool, float, str]] = []

    def run(self, name: str, fn: Callable[[], Optional[str]]):
        t0 = time.time()
        try:
            note = fn() or ""
            self.results.append((name, True, time.time() - t0, note))
        except Exception:
            self.results.append((name, False, time.time() - t0, traceback.format_exc()))

    def mini_lab(self, sub: str) -> "GenesisLab":
        cfg = _boot_config(self.device)
        paths = GenesisPaths(os.path.join(self.root, sub), on_drive=False)
        return GenesisLab(cfg, {"device": self.device.type, "gpu_name": "boot-test"}, paths,
                          EventLogger(paths, verbose=False), self.device)

    # ------------------------------------------------------------------
    def all(self) -> bool:
        cfg = _boot_config(self.device)
        space = ParamSpace(cfg)
        sim = Simulator(cfg, self.device)
        envs = make_environment_suite(cfg.master_seed)
        scr = Screener(sim, cfg)
        ve = ValidationEngine(sim, cfg, envs, scr, space)
        rng = np.random.default_rng(0)
        D = ve.channel_distance(cfg.world_size)
        ctrl = {k: control_genome(space, k, D, cfg.world_size) for k in ["identity", "noise_replace", "diffusion", "link_copy"]}

        def test_state_initialization():
            gs = [Genome(space.random_params(rng)) for _ in range(6)]
            gb = sim.prepare(gs)
            st = sim.init_state(gb, [envs[i % len(envs)] for i in range(6)], list(range(6)))
            assert st.U.shape == (6, cfg.channels_max, cfg.world_size, cfg.world_size)
            assert torch.isfinite(st.U).all()
            inactive = (gb.chm.view(6, -1) == 0)
            assert float(st.U.abs().sum((2, 3))[inactive].sum()) == 0.0, "canal inativo com estado não nulo"
            for e in envs:
                f = make_initial_field(e, 3, 16, 7, np.ones(3, bool)); f2 = make_initial_field(e, 3, 16, 7, np.ones(3, bool))
                assert np.array_equal(f, f2), "inicialização não reprodutível"

        def test_simulation_kernel():
            for shp in range(len(KERNEL_SHAPES)):
                k = build_kernel_np(shp, 3.0, 0.5, 0.15, 0.6, 0.7, cfg.radius_max)
                assert abs(k.sum() - 1) < 1e-5
            g = Genome(space.random_params(rng)); g.params["boundary"] = np.int64(0)
            g.params["k_active"][:] = False; g.params["k_active"][0] = True
            gb = sim.prepare([g]); U = torch.randn(1, cfg.channels_max, cfg.world_size, cfg.world_size, device=self.device)
            P = gb.P
            N = torch.fft.irfft2(torch.fft.rfft2(sim.pad(U, gb, P)).unsqueeze(1) * gb.Khat.unsqueeze(2), s=(gb.Hp, gb.Hp))[..., P:P + gb.S, P:P + gb.S]
            p = g.params
            kk = torch.as_tensor(build_kernel_np(int(p["k_shape"][0]), float(p["k_radius"][0]), float(p["k_mu"][0]), float(p["k_sigma"][0]),
                                                 float(p["k_aniso"][0]), float(p["k_angle"][0]), cfg.radius_max), device=self.device)
            ref = F.conv2d(F.pad(U.view(-1, 1, gb.S, gb.S), (P, P, P, P), mode="circular"), kk.flip(0, 1).view(1, 1, *kk.shape)).view_as(U)
            err = float((N[:, 0] - ref).abs().max())
            assert err < 1e-4, f"FFT conv != conv direta ({err})"
            lap = N[:, gb.K]
            ref_lap = (U.roll(1, 2) + U.roll(-1, 2) + U.roll(1, 3) + U.roll(-1, 3) - 4 * U)
            assert float((lap - ref_lap).abs().max()) < 1e-4, "laplaciano incorreto"
            return f"FFT-conv err={err:.1e}"

        def test_dynamics_shape():
            for b in range(len(BOUNDARIES)):
                gs = [Genome(space.random_params(rng)) for _ in range(4)]
                for g in gs:
                    g.params["boundary"] = np.int64(b); g.params["link_active"][:] = True
                gb = sim.prepare(gs)
                st = sim.init_state(gb, [envs[0]] * 4, [1, 2, 3, 4])
                st = sim.run(gb, st, 5)
                assert st.U.shape == (4, cfg.channels_max, cfg.world_size, cfg.world_size)
                assert st.G is None or st.G.shape[1] == cfg.links_max
            mixed = [Genome(space.random_params(rng)) for _ in range(3)]
            for i, g in enumerate(mixed):
                g.params["boundary"] = np.int64(i)
            gb = sim.prepare(mixed); st = sim.run(gb, sim.init_state(gb, [envs[0]] * 3, [5, 6, 7]), 5)
            assert torch.isfinite(st.U).all()

        def test_no_nan():
            gs = [Genome(space.random_params(rng)) for _ in range(16)]
            boom = Genome(space.random_params(rng))
            boom.params["W"][:] = 2.0; boom.params["g_gain"][:] = 2.0; boom.params["Q"][:] = 1.0; boom.params["sat"][:] = 4.0
            boom.params["max_rate"] = np.float32(1.0); boom.params["diss"][:] = 0
            gs.append(boom)
            gb = sim.prepare(gs)
            st = sim.run(gb, sim.init_state(gb, [envs[1]] * 17, list(range(17))), 60)
            assert torch.isfinite(st.U).all(), "NaN/Inf escapou"
            assert float(st.U.abs().max()) <= cfg.state_bound + 1e-5
            d = scr.screen([boom], envs[:1], 1, steps=40)[0]
            assert d["viability"] in ("EXPLOSIVE", "VIABLE", "NUMERICALLY_UNSTABLE", "PURE_NOISE", "FROZEN")
            return f"lei explosiva classificada como {d['viability']}"

        def test_mutation():
            g = Genome(space.random_params(rng))
            for i in range(200):
                op = MUTATION_OPERATORS[i % len(MUTATION_OPERATORS)]
                p, rec = space.mutate(g.params, rng, op, 1.0, space.random_params(rng))
                for n, s in space.specs.items():
                    v = np.asarray(p[n], np.float64)
                    if s.kind != "bool":
                        assert v.min() >= s.low - 1e-6 and v.max() <= s.high + 1e-6, f"{n} fora dos limites"
                assert p["ch_active"].any() and p["k_active"].any()
                assert rec
                g = Genome(p, parent_ids=[g.genome_id])
            p = space.random_params(rng); p["ch_active"][:] = False; p["ch_active"][0] = True
            p2 = {k: np.array(v) for k, v in p.items()}
            msg = space.duplicate_channel(p2, rng)
            assert p2["ch_active"].sum() == 2 and "dup_channel" in msg
            v = pack_params(space, p); p3 = unpack_params(space, v)
            assert hash_obj(space.clamp(p3)) == hash_obj(space.clamp({k: np.array(x) for k, x in p.items()})), "pack/unpack"

        def test_reproducibility():
            g = Genome(space.random_params(np.random.default_rng(5)))
            outs = []
            for _ in range(2):
                gb = sim.prepare([g, g])
                st = sim.run(gb, sim.init_state(gb, [envs[1]] * 2, [11, 11], groups=[0, 0], run_seed=3), 30)
                outs.append(st.U.clone())
            assert torch.allclose(outs[0], outs[1], atol=1e-6), "mesma semente != mesmo resultado"
            assert torch.allclose(outs[0][0], outs[0][1], atol=1e-6), "cópias idênticas divergiram"
            d1 = scr.screen([g], envs[:2], 9)[0]; d2 = scr.screen([g], envs[:2], 9)[0]
            assert abs(d1["persistence"] - d2["persistence"]) < 1e-5 and abs(d1["causal_spread"] - d2["causal_spread"]) < 1e-5

        def test_causal_probe():
            pos = ve.run_probe("causality", ctrl["diffusion"], 1)
            neg = ve.run_probe("causality", ctrl["identity"], 1)
            assert pos.status == STATUS_PASS, f"controle positivo (difusão) -> {pos.status} {pos.details.get('error', '')}"
            assert neg.status == STATUS_FAIL, f"controle negativo (identidade) -> {neg.status}"
            return f"+:{pos.details['spread_mean']:.2f} −:{neg.details['spread_mean']:.2f}"

        def test_memory_probe():
            pos = ve.run_probe("memory", ctrl["identity"], 2)
            neg = ve.run_probe("memory", ctrl["noise_replace"], 2)
            assert pos.status == STATUS_PASS, f"controle positivo (identidade) -> {pos.status} {pos.details.get('error', '')}"
            assert neg.status != STATUS_PASS, f"controle negativo (ruído) -> {neg.status}"
            assert not pos.details["leakage_detected"]
            return f"+:{pos.score:.2f} −:{neg.status}"

        def test_channel_probe():
            pos = ve.run_probe("channel", ctrl["link_copy"], 3)
            neg = ve.run_probe("channel", ctrl["identity"], 3)
            assert pos.status == STATUS_PASS, f"controle positivo (link) -> {pos.status} {pos.details.get('error', '')}"
            assert neg.status == STATUS_FAIL, f"controle negativo (isolado) -> {neg.status}"
            return f"+: latência={pos.details['latency']} MI={pos.details['mi_bits']:.2f} bits"

        def test_transformation_instrument():
            r_ = np.random.default_rng(1)
            a1, a2 = r_.integers(0, 2, 64), r_.integers(0, 2, 64)
            Xp = np.stack([a1, a2, a1 * a2], 1) + r_.normal(0, 0.1, (64, 3))
            Xn = np.stack([a1, a2], 1) + r_.normal(0, 0.1, (64, 2))
            pos = decode_report(Xp, a1 ^ a2, 2, 60, 4, 1); neg = decode_report(Xn, a1 ^ a2, 2, 60, 4, 1)
            assert pos["linear_p"] < 0.01 and pos["linear_acc"] > 0.85, "XOR com termo não linear deveria ser decodificável"
            assert neg["linear_p"] > 0.01, "XOR de features lineares não deveria ser linearmente decodificável"
            return f"+:{pos['linear_acc']:.2f} −:{neg['linear_acc']:.2f}"

        def test_recovery_probe():
            uni = [Environment("CTRL-U", "control", "uniform", 1, 0.5, 1, 0, 0, 0, 7)]
            pos = ve.run_probe("recovery", ctrl["diffusion"], 4, allow_fallback=True, envs=uni)
            neg = ve.run_probe("recovery", ctrl["identity"], 4, allow_fallback=True, envs=uni)
            assert pos.status == STATUS_PASS, f"controle positivo (difusão sobre campo uniforme) -> {pos.status}"
            assert neg.status == STATUS_FAIL, f"controle negativo (estático) -> {neg.status}"
            return f"+:{pos.score:.2f} −:{neg.score:.2f}"

        def test_archive():
            me = MapElites(["persistence", "causal_spread", "temporal_complexity"], [4, 4, 2])
            d = {"persistence": 0.5, "causal_spread": 0.2, "temporal_complexity": 0.9}
            assert me.try_insert("a", d, 0.3, 0) == (True, True)
            assert me.try_insert("b", d, 0.2, 0) == (False, False)
            assert me.try_insert("c", d, 0.5, 1) == (True, False)
            assert me.cells[me.index(d)]["genome_id"] == "c" and me.history[0]["genome_id"] == "a", "elite histórica perdida"
            na = NoveltyArchive(3, 100)
            V = np.random.default_rng(0).random((10, len(DESCRIPTOR_NAMES))).astype(np.float32)
            nv = na.novelty(V, V); assert nv.shape == (10,) and (nv > 0).all()
            na.maybe_add([f"g{i}" for i in range(10)], V, nv + 1, np.random.default_rng(0)); assert len(na.ids) == 10
            z = Zoo(); dd = {"viability": "VIABLE", "rel_activity": 0.1, "persistence": 0.1, "structures": 0, "periodicity": 0.1,
                             "causal_spread": 0.0, "lyapunov": 0.0, "mobility": 0.0}
            assert Zoo.classify(dd, {}, Thresholds(), False) == ["UNKNOWN"]
            assert "ANOMALOUS" in Zoo.classify(dd, {}, Thresholds(), True)
            lin = Lineage(); g0 = Genome(space.random_params(rng)); g1 = Genome(space.random_params(rng), parent_ids=[g0.genome_id])
            lin.add(g0); lin.add(g1); assert lin.ancestors(g1.genome_id) == [g0.genome_id] and g1.genome_id in lin.tree(g0.genome_id)
            sp = Speciation(0.3); s1 = sp.assign(np.zeros(5), np.zeros(5), 0); s2 = sp.assign(np.ones(5), np.ones(5), 0)
            assert s1 != s2 and sp.assign(np.zeros(5) + 0.01, np.zeros(5), 0) == s1

        def test_checkpoint_save():
            lab = self.mini_lab("ckpt")
            lab.initialize()
            p = lab.save_checkpoint("test")
            assert os.path.exists(p) and not os.path.exists(lab.ckpt.tmp)
            lab.counters["generation"] += 1
            lab.save_checkpoint("test2")
            assert os.path.exists(lab.ckpt.previous)
            return human_bytes(os.path.getsize(p))

        def test_checkpoint_restore():
            lab = self.mini_lab("ckpt")
            st, path = lab.ckpt.load_latest()
            assert st is not None and st["counters"]["generation"] == 1
            lab2 = self.mini_lab("ckpt"); lab2.load_state_dict(st)
            assert len(lab2.population) == len(st["population"]) and set(lab2.store) == set(st["store"])
            with open(lab.ckpt.current, "wb") as f:
                f.write(b"corrupted")
            st2, path2 = lab.ckpt.load_latest()
            assert st2 is not None and path2 == lab.ckpt.previous, "fallback para checkpoint_previous falhou"
            return "corrupção detectada → previous"

        def test_resume():
            lab = self.mini_lab("resume")
            lab.initialize()
            lab.generation_step()
            ev1, gen1 = lab.counters["evaluations"], lab.counters["generation"]
            pop1 = list(lab.population)
            lab.save_checkpoint("t")
            lab2 = self.mini_lab("resume")
            st, _ = lab2.ckpt.load_latest()
            lab2.load_state_dict(st)
            assert lab2.counters["generation"] == gen1 and lab2.counters["evaluations"] == ev1 and lab2.population == pop1
            lab2.generation_step()
            assert lab2.counters["generation"] == gen1 + 1 and lab2.counters["evaluations"] > ev1, "contadores reiniciaram"
            return f"gen {gen1}→{lab2.counters['generation']}, evals {ev1}→{lab2.counters['evaluations']}"

        def test_discovery_reproduction():
            lab = self.mini_lab("repro")
            g = ctrl["identity"]; lab.store[g.genome_id] = g
            res = lab.ve.run_probe("memory", g, 77)
            lab.best_level_global = 99          # teste silencioso: sem relatório/bundle
            lab.register_discovery(g.genome_id, 2, 3, res, ["test"])
            did = list(lab.discoveries)[-1]
            with contextlib.redirect_stdout(io.StringIO()):
                out = reproduce_discovery(did, lab)
            assert out["status_match"] and out["score_match"] and out["genome_hash_match"], f"reprodução divergiu: {out}"

        def test_device_consistency():
            """A física determinística deve coincidir entre CPU e CUDA (tolerância de float32).
            Os geradores aleatórios de CPU e CUDA produzem sequências DIFERENTES para a mesma semente, então ruído
            intrínseco, assincronia e choques ficam desligados aqui (eles são reprodutíveis apenas no mesmo tipo de
            dispositivo). Leis aleatórias são comparadas em 1 passo (caos amplificaria arredondamentos legítimos);
            leis conhecidas e estáveis, em 30 passos."""
            if not torch.cuda.is_available():
                return "sem GPU — pulado (CPU apenas)"
            r_ = np.random.default_rng(3)
            gs = []
            for i in range(6):
                g = Genome(space.random_params(r_))
                g.params["noise"] = np.array(0.0, np.float32)
                g.params["async_p"] = np.array(1.0, np.float32)
                g.params["boundary"] = np.array(i % len(BOUNDARIES), np.int64)
                gs.append(g)
            quiet = dc_replace(envs[0], ext_noise=0.0, shock_period=0, shock_amp=0.0)

            def run(dev, genomes, steps):
                s_ = Simulator(cfg, dev); gb = s_.prepare(genomes)
                st = s_.init_state(gb, [quiet] * len(genomes), list(range(len(genomes))), run_seed=1)
                return s_.run(gb, st, steps).U.cpu()

            def rel_err(a, b):
                return float(((a - b).abs().flatten(1).amax(1) / (a.abs().flatten(1).amax(1) + 1e-6)).max())
            e1 = rel_err(run(torch.device("cpu"), gs, 1), run(torch.device("cuda"), gs, 1))
            ctl = [control_genome(space, k, D, cfg.world_size) for k in ["diffusion", "reaction_diffusion", "link_copy"]]
            e30 = rel_err(run(torch.device("cpu"), ctl, 30), run(torch.device("cuda"), ctl, 30))
            assert e1 < 1e-3, f"1 passo, leis aleatórias sem ruído: CPU vs CUDA erro relativo {e1:.2e}"
            assert e30 < 1e-3, f"30 passos, leis de controle: CPU vs CUDA erro relativo {e30:.2e}"
            return f"rel.err 1 passo={e1:.1e}, 30 passos (controles)={e30:.1e}; ruído: fluxo por dispositivo"

        def test_signal_instruments():
            """Medidas graduadas da triagem (gradiente da busca dirigida) com controles: o link que copia A em B
            transmite; identidade e ruído não; identidade guarda traço local, ruído não. O instrumento de não
            linearidade é validado com sinais sintéticos: resposta só ímpar (linear) ≈ 0; com termo par > 0."""
            gs = [ctrl["identity"], ctrl["noise_replace"], ctrl["link_copy"]]
            ev = [e for e in envs if e.split == "evolution"]
            r = {k: x for k, x in zip(["identity", "noise", "link"], scr.screen(gs, ev, 11))}
            assert r["link"]["transmission"] > 0.3, f"controle positivo de transmissão fraco: {r['link']['transmission']:.3f}"
            assert r["identity"]["transmission"] < 0.02 and r["noise"]["transmission"] < 0.02, "transmissão sem canal"
            assert r["identity"]["memory_trace"] > 0.3 and r["noise"]["memory_trace"] < 0.1, "traço de memória"
            g_ = torch.Generator().manual_seed(0)
            T_, n_, nf_ = 6, 5, 12
            base = torch.randn(T_, n_, nf_, generator=g_)
            a = torch.randn(T_, n_, nf_, generator=g_)
            q = torch.randn(T_, n_, nf_, generator=g_)
            lin, _ = signal_metrics(torch.stack([base, base + a, base - a], 1), torch.ones(T_, n_), list(range(1, T_ + 1)))
            par, _ = signal_metrics(torch.stack([base, base + a + q, base - a + q], 1), torch.ones(T_, n_), list(range(1, T_ + 1)))
            assert lin["nonlinearity_raw"].max() < 0.05, "resposta linear acusada como não linear"
            assert par["nonlinearity_raw"].min() > 0.3, "termo par não detectado"
            return (f"tx link={r['link']['transmission']:.2f} id={r['identity']['transmission']:.2f} · "
                    f"traço id={r['identity']['memory_trace']:.2f} ruído={r['noise']['memory_trace']:.2f} · "
                    f"não-lin sintética {lin['nonlinearity_raw'].max():.2f}→{par['nonlinearity_raw'].min():.2f}")

        def test_live_panel():
            lab = self.mini_lab("live")
            lab.initialize()
            parts = {}

            class _H:
                def __init__(self, k): self.k = k
                def update(self, h): parts[self.k] = getattr(h, "data", str(h))

            class _M(LiveMonitor):
                def _open_handles(self, now):
                    best = self.lab.best_candidate()
                    parts.update(head=self._head_html(now), view=self._view_html(now, best), charts=self._charts_html())
                    self.handles = {k: _H(k) for k in ("head", "view", "charts")}
                    self.last_refresh = self.view_ts = self.charts_ts = now
            m = _M(lab, interval=0.0)
            m.mode = "html"
            LIVE.monitor = m
            try:
                m.start()
                lab.generation_step()
                m.render(force=True)
            finally:
                LIVE.monitor = None
            assert m.errors == 0, "erro ao desenhar o painel (veja events.jsonl: live_render_error)"
            assert "AO VIVO" in parts["head"] and "MELHOR CANDIDATO" in parts["view"] and "<svg" in parts["charts"]
            assert "data:image/jpeg;base64," in parts["view"], "mosaico ao vivo não foi gerado"
            jpg = mosaic_jpeg([np.random.default_rng(0).normal(size=(cfg.channels_max, 16, 16))])
            assert jpg[:2] == b"\xff\xd8", "JPEG inválido"
            return f"painel: {sum(len(v) for v in parts.values()) // 1024} KB de HTML por quadro"

        for name, fn in [("test_state_initialization", test_state_initialization), ("test_simulation_kernel", test_simulation_kernel),
                         ("test_dynamics_shape", test_dynamics_shape), ("test_no_nan", test_no_nan),
                         ("test_mutation", test_mutation), ("test_checkpoint_save", test_checkpoint_save),
                         ("test_checkpoint_restore", test_checkpoint_restore), ("test_reproducibility", test_reproducibility),
                         ("test_causal_probe", test_causal_probe), ("test_memory_probe", test_memory_probe),
                         ("test_channel_probe", test_channel_probe), ("test_transformation_instrument", test_transformation_instrument),
                         ("test_recovery_probe", test_recovery_probe), ("test_archive", test_archive),
                         ("test_resume", test_resume), ("test_discovery_reproduction", test_discovery_reproduction),
                         ("test_device_consistency", test_device_consistency), ("test_live_panel", test_live_panel),
                         ("test_signal_instruments", test_signal_instruments)]:
            self.run(name, fn)
        return all(ok for _, ok, _, _ in self.results)

    def report(self) -> str:
        lines = []
        for name, ok, dt, note in self.results:
            first = note.strip().splitlines()[-1] if note.strip() else ""
            lines.append(f"  {'PASS' if ok else 'FAIL'}  {name:<32} {dt:6.2f}s  {first[:90] if ok else ''}")
        return "\n".join(lines)


def run_boot_tests(device: torch.device, root: str) -> BootSuite:
    test_root = tempfile.mkdtemp(prefix="genesis_boot_")       # disco local (rápido; não suja o Drive)
    suite = BootSuite(device, test_root)
    t0 = time.time()
    ok = suite.all()
    print(suite.report())
    shutil.rmtree(test_root, ignore_errors=True)
    if not ok:
        print("\n╔══════════════════════════════════════╗\n║          GENESIS BOOT FAILED         ║\n╚══════════════════════════════════════╝")
        for name, passed, _, note in suite.results:
            if not passed:
                print(f"\n--- {name} ---\n{note}")
        raise RuntimeError("GENESIS BOOT FAILED — pesquisa não iniciada (veja os stack traces acima)")
    print(f"\nTodos os {len(suite.results)} testes de boot passaram em {time.time() - t0:.1f}s.")
    return suite
