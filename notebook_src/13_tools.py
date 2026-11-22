# %% [markdown]
# ## 13. Ferramentas do pesquisador
# `inspect_candidate(id)`, `compare_candidates(a, b)`, `replay_candidate(id)`, `reproduce_discovery(discovery_id)`,
# `export_candidate(id)`, `import_candidate(path)`, `profile_genesis()`, `results_dataframe()`, `list_discoveries()`.

# %%
def _lab(lab=None) -> "GenesisLab":
    lab = lab or globals().get("LAB")
    if lab is None:
        raise RuntimeError("LAB ainda não inicializado — rode as células de boot.")
    return lab


def _resolve(lab, cid: Optional[str]) -> str:
    if cid is None:
        cid = lab.best_candidate()
    if cid not in lab.store:
        raise KeyError(f"candidato {cid} não está no store (talvez coletado; use um ID de arquivo/descoberta)")
    return cid


def inspect_candidate(candidate_id: Optional[str] = None, lab=None, show: bool = True, steps: int = 160):
    lab = _lab(lab); cid = _resolve(lab, candidate_id)
    g = lab.store[cid]
    print(g.summary(lab.space))
    print("\nScorecard:")
    for k, v in lab.scorecard(cid).items():
        print(f"  {k:>18}: {v}")
    print("\nDescritores:", {k: round(v, 4) for k, v in lab.desc.get(cid, {}).items() if isinstance(v, float)})
    print("\nLinhagem:\n" + lab.lineage.tree(cid, depth=2))
    print("\nExperimentos:")
    for k, rid in lab.tests.get(cid, {}).items():
        r = lab.results.get(rid)
        if r:
            print(f"  {k:>22}: {r.status:<12} score={r.score:.3f} p={r.p_value:.2e} ({r.version}, {r.duration_s:.1f}s)")
    if show:
        frames, _ = simulate_frames(lab, g, steps=steps, every=4)
        plot_frames(frames, f"{cid}", None, True)
        for k, rid in lab.tests.get(cid, {}).items():
            r = lab.results.get(rid)
            if r is None:
                continue
            if r.probe == "memory":
                plot_memory_curve(r, None, True)
            elif r.probe == "causality":
                plot_causal_map(r, None, True)
            elif r.probe == "recovery":
                plot_damage_curve(r, None, True)
            elif r.probe == "robustness":
                plot_robustness_curve(r, None, True)
    return lab.scorecard(cid)


def compare_candidates(a: str, b: str, lab=None) -> Dict[str, Any]:
    lab = _lab(lab)
    sa, sb = lab.scorecard(a), lab.scorecard(b)
    rows = [(k, sa.get(k), sb.get(k)) for k in sa.keys()]
    va, vb = lab.vec.get(a), lab.vec.get(b)
    bd = float(np.linalg.norm(va - vb)) if va is not None and vb is not None else float("nan")
    gd = float(np.sqrt(np.mean((lab.space.vector(lab.store[a].params) - lab.space.vector(lab.store[b].params)) ** 2))) \
        if a in lab.store and b in lab.store else float("nan")
    print(f"{'campo':>18} | {a:^24} | {b:^24}")
    for k, x, y in rows:
        print(f"{k:>18} | {str(x):^24} | {str(y):^24}")
    print(f"\ndistância comportamental={bd:.4f}  distância genética={gd:.4f}")
    stat = {}
    for key in ["replication_channel", "replication_memory"]:
        ra, rb = lab.result_of(a, key), lab.result_of(b, key)
        if ra and rb:
            x, y = np.asarray(ra.details["scores"]), np.asarray(rb.details["scores"])
            stat[key] = {"p_a_gt_b": mann_whitney_greater(x, y), "cohens_d": cohens_d(x, y)}
    if stat:
        print("comparação estatística:", stat)
    return {"rows": rows, "behavior_distance": bd, "genetic_distance": gd, "stats": stat}


def replay_candidate(candidate_id: Optional[str] = None, lab=None, env_id: Optional[str] = None, steps: int = 300,
                     gif: bool = True, seed: int = 0):
    lab = _lab(lab); cid = _resolve(lab, candidate_id)
    env = next((e for e in lab.envs if e.env_id == env_id), lab.envs[0])
    frames, _ = simulate_frames(lab, lab.store[cid], env, steps=steps, every=2, seed=seed)
    plot_frames(frames, f"replay {cid} @ {env.env_id}", None, True)
    if gif:
        p = save_animation(frames, os.path.join(lab.paths.plots, f"replay_{cid}_{env.env_id}.gif"))
        if p and IN_NOTEBOOK and IPImage is not None:
            display(IPImage(filename=p))
        return p
    return None


def reproduce_discovery(discovery_id: str, lab=None, deterministic: bool = True) -> Dict[str, Any]:
    """Restaura genoma, semente, protocolo, ambientes, duração e reexecuta o teste com o Validation Engine."""
    lab = _lab(lab)
    rec = lab.discoveries[discovery_id]
    g = genome_from_record(lab.space, rec["genome"])
    prot = rec["protocol"]
    probe = prot["probe"]
    kw = dict(prot.get("kwargs", {}))
    prev = torch.are_deterministic_algorithms_enabled()
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        res = lab.ve.run_probe(probe, g, prot["seed"], **kw)
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=True)
    same_status = res.status == rec["status"]
    a, b = res.score, rec["score"]
    same_score = (not np.isfinite(a) and not np.isfinite(b)) or (np.isfinite(a) and np.isfinite(b) and abs(a - b) < 1e-3)
    out = {"discovery_id": discovery_id, "probe": probe, "original_status": rec["status"], "reproduced_status": res.status,
           "original_score": b, "reproduced_score": a, "status_match": same_status, "score_match": bool(same_score),
           "genome_hash_match": g.hash() == prot.get("genome_hash"), "engine": res.version}
    print(json.dumps(to_jsonable(out), indent=1))
    return out


def export_candidate(candidate_id: Optional[str] = None, lab=None, path: Optional[str] = None) -> str:
    lab = _lab(lab); cid = _resolve(lab, candidate_id)
    path = path or os.path.join(lab.paths.exports, f"{cid}.genesis.pt")
    torch.save({"genesis_version": GENESIS_VERSION, "engine": VALIDATION_ENGINE_VERSION,
                "channels_max": lab.cfg.channels_max, "kernel_slots": lab.cfg.kernel_slots, "links_max": lab.cfg.links_max,
                "delay_max": lab.cfg.delay_max, "radius_max": lab.cfg.radius_max,
                "genome": genome_to_record(lab.space, lab.store[cid]), "descriptor": lab.desc.get(cid),
                "results": {k: lab.results[r].to_dict() for k, r in lab.tests.get(cid, {}).items() if r in lab.results},
                "lineage": {a: lab.lineage.records.get(a) for a in [cid] + lab.lineage.ancestors(cid)},
                "hash": lab.store[cid].hash()}, path)
    print(f"exportado: {path} ({human_bytes(os.path.getsize(path))})")
    return path


def import_candidate(path: str, lab=None, evaluate: bool = True) -> str:
    lab = _lab(lab)
    d = _torch_load(path)
    dims = ("channels_max", "kernel_slots", "links_max", "delay_max", "radius_max")
    if any(d.get(k) != getattr(lab.cfg, k) for k in dims):
        raise ValueError(f"dimensões do genoma incompatíveis com esta pesquisa: {[(k, d.get(k), getattr(lab.cfg, k)) for k in dims]}")
    g = genome_from_record(lab.space, d["genome"])
    if g.hash() != d["hash"]:
        raise ValueError("hash do genoma importado não confere (arquivo corrompido?)")
    g.genome_id = new_id("G"); g.origin = f"imported:{os.path.basename(path)}"; g.parent_ids = [d["genome"]["genome_id"]]
    g.generation = lab.counters["generation"]
    if evaluate:
        lab.evaluate_and_integrate([g], {})
        lab.zoo.add(g.genome_id, ["UNKNOWN"], lab.novelty_score.get(g.genome_id, 0.0), lab.counters["generation"])
        lab.schedule_experiments([g.genome_id])
    else:
        lab.store[g.genome_id] = g
    print(f"importado como {g.genome_id}")
    return g.genome_id


def profile_genesis(lab=None, steps: int = 20, batch: Optional[int] = None):
    lab = _lab(lab)
    B = batch or min(lab.cfg.batch_size, 64)
    gs = [Genome(lab.space.random_params(np.random.default_rng(i))) for i in range(B)]
    gb = lab.sim.prepare(gs)
    st = lab.sim.init_state(gb, [lab.envs[0]] * B, list(range(B)))
    try:
        from torch.profiler import profile, ProfilerActivity
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
        with profile(activities=acts) as prof:
            lab.sim.run(gb, st, steps)
        print(prof.key_averages().table(sort_by="self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total", row_limit=15))
    except Exception as e:
        t0 = time.time(); lab.sim.run(gb, st, steps); _sync()
        print(f"profiler indisponível ({e}); {B * steps / (time.time() - t0):.0f} universo·passos/s")


def results_dataframe(lab=None):
    lab = _lab(lab)
    rows = [{"result_id": k, "candidate": r.candidate_id, "probe": r.probe, "status": r.status, "score": r.score,
             "p": r.p_value, "effect": r.effect_size, "split": r.split, "seconds": r.duration_s, "version": r.version}
            for k, r in lab.results.items()]
    return pd.DataFrame(rows) if HAVE_PANDAS else rows


def list_discoveries(lab=None):
    lab = _lab(lab)
    rows = [{"discovery_id": k, "candidate": v["candidate_id"], "level": v["new_level"], "probe": v["probe"],
             "score": v["score"], "generation": v["generation"], "global_record": v.get("global_record")}
            for k, v in lab.discoveries.items()]
    return pd.DataFrame(rows) if HAVE_PANDAS else rows


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
