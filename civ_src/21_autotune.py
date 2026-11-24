# %% [markdown]
# ## 21 — AUTOTUNE: benchmark real do hardware (física, desenvolvimento, agentes, streaming) e carga segura
# Nunca assume A100: sobe a carga progressivamente medindo tempo por tick e pico de VRAM; mantém margem (≤ 70% da
# VRAM total) e um orçamento de tempo por tick. Resultado vai para `reports/autotune.json` e para o checkpoint.

# %%
def benchmark_and_tune(cfg: CivConfig, device: torch.device, logger: Optional[EventLogger], enabled: bool = True) -> Tuple[CivConfig, Dict[str, Any]]:
    rep: Dict[str, Any] = {"device": device.type, "profile": cfg.profile, "ts": iso_now()}
    L = GenomeLayout(cfg).to(device)
    we = WorldEngine(cfg, device); matter = MatterEngine(cfg, device)
    dev = DevelopmentEngine(cfg, L, device); body = BodyEngine(cfg, dev, device); mind = SignalEngine(cfg, L, device)
    g = torch.Generator(device=device); g.manual_seed(1)

    def timed(fn, reps=3):
        sync_device(); t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        sync_device()
        return (time.perf_counter() - t0) / reps * 1000
    # física e desenvolvimento
    G = L.random(min(64, cfg.max_bodies), g)
    rep["development_ms_per_64"] = round(timed(lambda: dev.develop(G, torch.arange(G.shape[0], device=device)), 2), 2)
    bb = body.make(dev.develop(G, torch.arange(G.shape[0], device=device)))
    rep["physics_ms_per_step_64_bodies"] = round(timed(lambda: body.step(bb, torch.zeros(G.shape[0], 2, device=device)), 3), 3)
    ws = we.generate(5)
    eng = AgentEngine(cfg, L, we, matter, mind, device)
    ledger = StructureLedger(min(cfg.max_blocks, 100000), device)
    total = torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else 0
    budget_ms = 45.0 if device.type == "cuda" else 400.0
    chosen = cfg.max_agents
    sizes = sorted({max(100, cfg.max_agents // 8), max(100, cfg.max_agents // 4), max(100, cfg.max_agents // 2), cfg.max_agents})
    rows = []
    if enabled:
        for n in sizes:
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                pool = AgentPool(n, L, cfg, device); ids = IdCounter(1)
                eng.spawn_founders(ws, pool, n, g, ids, 0, None)
                pool.body_cap[:n] = torch.tensor([0.004, 1.0, 20.0, 0.02, 0.0, 0.01], device=device)
                ms = timed(lambda: eng.step(ws, pool, 0, g, None, ids, ledger, record_interactions=False), 3)
                peak = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
                rows.append({"agents": n, "ms_per_tick": round(ms, 2), "peak_vram": peak})
                del pool
                if ms > budget_ms or (total and peak > 0.55 * total):
                    chosen = max(sizes[0], n // 2) if ms > 1.5 * budget_ms else n
                    break
                chosen = n
            except Exception as e:
                if is_oom_error(e):
                    torch.cuda.empty_cache()
                    rows.append({"agents": n, "oom": True})
                    chosen = max(sizes[0], n // 2)
                    break
                raise
    rep["agent_benchmark"] = rows
    if enabled and chosen < cfg.max_agents:
        rep["max_agents_reduced"] = [cfg.max_agents, chosen]
        cfg.max_agents = int(chosen)
        cfg.initial_agents = int(min(cfg.initial_agents, chosen // 3))
    # streaming: custo de montar um frame (empacotamento msgpack)
    if HAVE_MSGPACK:
        arr = np.random.default_rng(0).integers(0, 65535, (cfg.max_stream_agents, 2)).astype(np.uint16)
        t0 = time.perf_counter()
        for _ in range(5):
            pack_msg({"t": "frame", "pos": arr.tobytes(), "ids": np.arange(cfg.max_stream_agents, dtype=np.int64).tobytes()})
        rep["stream_pack_ms"] = round((time.perf_counter() - t0) / 5 * 1000, 3)
    rep["final"] = {"max_agents": cfg.max_agents, "initial_agents": cfg.initial_agents, "max_bodies": cfg.max_bodies,
                    "world": [cfg.world_w, cfg.world_h], "precision": cfg.precision, "compile": cfg.use_compile,
                    "note": "fp32 = referência; bf16/compile não usados sem teste de erro aceitável"}
    if total:
        rep["vram_total"] = int(total)
    if logger:
        logger.event("autotune", **{k: v for k, v in rep.items() if k != "agent_benchmark"})
    return cfg, rep
