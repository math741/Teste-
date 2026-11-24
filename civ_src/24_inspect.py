# %% [markdown]
# ## 24 — INSPECTION: inspecionar, comparar, reexecutar, reproduzir, exportar e importar
# Funções preservadas do Genesis Ω, agora sobre o universo 3D. Exportações levam schema, versão e hash; importação
# valida os três (nunca carrega formato incompatível silenciosamente; nunca usa pickle).

# %%
EXPORT_SCHEMA = "genesis-civ-candidate/1"


def inspect_candidate(cid: str, uni: Optional[UniverseEngine] = None) -> Dict[str, Any]:
    uni = uni or GENESIS["uni"]
    c = uni.archive.zoo.get(cid)
    if c is None:
        raise KeyError(f"candidato {cid} não está no zoológico")
    G = torch.as_tensor(c.genome, device=uni.device).view(1, -1)
    bp = uni.dev.develop(G, torch.tensor([1], device=uni.device))
    cap = uni.body.measure_capability(G, torch.tensor([1], device=uni.device))[0].cpu().numpy()
    out = {"cid": cid, "origin": c.origin, "fitness": c.fitness, "evidence_score": c.evidence_score(), "passed": c.passed,
           "failed": c.failed, "attempts": c.attempts, "h_level": c.h_level(), "a_level": c.a_level(),
           "body": {"nodes": int(bp.occ.sum()), "speed": float(cap[0]), "cost": float(cap[1]), "turn": float(cap[3])},
           "genome_sections": {k: float(torch.as_tensor(c.genome)[uni.L.slices[k][0]:uni.L.slices[k][1]].abs().mean())
                               for k in ("ctl_W0", "ctl_Win", "plast_eta", "plast_abcd", "dev_Wm")},
           "note": "fitness ≠ evidência; níveis só por probes com controles"}
    return out


def compare_candidates(a: str, b: str, uni: Optional[UniverseEngine] = None) -> Dict[str, Any]:
    uni = uni or GENESIS["uni"]
    A, B = uni.archive.zoo[a], uni.archive.zoo[b]
    d = float(uni.L.distance(torch.as_tensor(A.genome).view(1, -1), torch.as_tensor(B.genome).view(1, -1)))
    return {"a": a, "b": b, "genome_distance": d, "levels_a": sorted(A.passed), "levels_b": sorted(B.passed),
            "only_a": sorted(set(A.passed) - set(B.passed)), "only_b": sorted(set(B.passed) - set(A.passed)),
            "evidence": [A.evidence_score(), B.evidence_score()], "fitness": [A.fitness, B.fitness]}


def replay_candidate(cid: str, ticks: int = 160, uni: Optional[UniverseEngine] = None, seed: int = 7) -> Dict[str, Any]:
    """Reexecuta o candidato numa arena padronizada (mesma física) e devolve trajetórias + métricas."""
    uni = uni or GENESIS["uni"]
    c = uni.archive.zoo[cid]
    n = uni.cfg.probe.trials
    G = torch.as_tensor(c.genome, device=uni.device).expand(n, -1).contiguous()
    tr = uni.runner.build(G, ArenaSpec(task="replay", n_patches=3, resource_floor=0.05), seed)
    rec = drive(uni.runner.run_gen(tr, ticks, record=("gain", "pos", "alive")))
    return {"cid": cid, "intake": summary_stats(rec["gain"].sum(0).cpu().numpy()), "survival": float(rec["alive"].float().mean()),
            "replay": make_replay(tr, rec)}


def reproduce_discovery(did: str, uni: Optional[UniverseEngine] = None) -> Dict[str, Any]:
    """Reexecuta o probe da descoberta com o genoma salvo e as MESMAS sementes; compara com o registro."""
    uni = uni or GENESIS["uni"]
    meta = uni.archive.load_discovery(did)
    if meta is None:
        raise KeyError(f"descoberta {did} não encontrada")
    ver = uni.archive.verify_discovery(did)
    if meta.get("kind") == "civilization":
        return {"id": did, "integrity": ver, "note": "descoberta civilizacional: reprodução = carregar o checkpoint/snapshot "
                "do tick e reavaliar os critérios", "evidence": meta.get("evidence")}
    G = torch.as_tensor(meta["genome"], device=uni.device)
    code = meta["level"]
    key = meta["candidate"]
    res = drive(uni.validation.run_gen(code, G, key))
    first = meta["evidence"][0] if meta.get("evidence") else {}
    return {"id": did, "integrity": ver, "level": code, "reproduced_passed": res.passed, "p": res.p, "effect": res.effect,
            "original_p": first.get("p"), "original_effect": first.get("effect"),
            "status_match": bool(res.passed == bool(first.get("passed", True))),
            "versions_match": meta.get("versions_hash") == ENGINE_VERSIONS_HASH}


def export_candidate(cid: str, path: Optional[str] = None, uni: Optional[UniverseEngine] = None) -> str:
    uni = uni or GENESIS["uni"]
    c = uni.archive.zoo[cid]
    path = path or os.path.join(uni.paths.exports, f"{cid}.npz")
    meta = {"schema": EXPORT_SCHEMA, "cid": cid, "origin": c.origin, "passed": c.passed, "versions": ENGINE_VERSIONS,
            "versions_hash": ENGINE_VERSIONS_HASH, "genome_P": int(c.genome.size), "layout": {k: list(v) for k, v in uni.L.slices.items()},
            "genome_sha256": hashlib.sha256(c.genome.astype(np.float32).tobytes()).hexdigest(), "exported": iso_now()}
    np.savez(path, genome=c.genome.astype(np.float32), meta=np.frombuffer(json.dumps(to_jsonable(meta)).encode(), dtype=np.uint8))
    return path


def import_candidate(path: str, uni: Optional[UniverseEngine] = None) -> str:
    uni = uni or GENESIS["uni"]
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta"].astype(np.uint8)).decode())
        g = np.asarray(z["genome"], np.float32)
    if meta.get("schema") != EXPORT_SCHEMA:
        raise ValueError(f"schema incompatível: {meta.get('schema')}")
    if hashlib.sha256(g.tobytes()).hexdigest() != meta.get("genome_sha256"):
        raise ValueError("hash do genoma não confere (arquivo corrompido ou alterado)")
    if int(meta.get("genome_P", -1)) != uni.L.P or meta.get("layout") != {k: list(v) for k, v in uni.L.slices.items()}:
        raise ValueError("layout de genoma incompatível com este motor (exige migração explícita)")
    c = uni.scientist.add_candidate(g, f"import:{meta.get('cid')}", uni.tick_count)
    return c.cid


def inspect_agent(aid: int, uni: Optional[UniverseEngine] = None) -> Dict[str, Any]:
    uni = uni or GENESIS["uni"]
    return agent_card(uni, int(aid))
