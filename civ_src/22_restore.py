# %% [markdown]
# ## 22 — RESTORE / CREATE: retoma o MESMO mundo (idade, agentes, cultura, assentamentos, fila científica)
# Ordem: checkpoint válido mais recente (current → previous → backups) → se não houver nenhum, cria o planeta.
# `FORCE_NEW=True` arquiva os checkpoints existentes em `checkpoints/archived_*` (nunca apaga) e cria um mundo novo.
# A pesquisa 2D do Genesis Ω é importada como arquivo separado (`legacy/`), sem fingir que tinha corpos.

# %%
def boot_universe(cfg: CivConfig, paths: CivPaths, device: torch.device, logger: EventLogger, hw: Dict[str, Any],
                  force_new: bool = False, import_legacy: bool = True, bootstrap_seconds: Optional[float] = None):
    ck = CheckpointEngine(paths, logger)
    info: Dict[str, Any] = {}
    if force_new and ck.candidates():
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(ck.root, f"archived_{stamp}")
        os.makedirs(dst, exist_ok=True)
        for d in ck.candidates():
            shutil.move(d, os.path.join(dst, os.path.basename(d)))
        info["archived"] = dst
        logger.event("force_new_archived", echo=True, dir=dst)
    state, linfo = (None, {}) if force_new else ck.load_latest()
    info.update(linfo)
    uni: Optional[UniverseEngine] = None
    if state is not None:
        saved = CivConfig.from_dict(state["meta"]["config"])
        if saved.thresholds_hash() != cfg.thresholds_hash():
            logger.event("thresholds_changed_warning", echo=True, saved=saved.thresholds_hash(), now=cfg.thresholds_hash(),
                         note="resultados antigos continuam vinculados aos thresholds com que foram obtidos")
        if state["meta"].get("versions_hash") != ENGINE_VERSIONS_HASH:
            logger.event("migration_warning", echo=True, saved=state["meta"].get("versions_hash"), now=ENGINE_VERSIONS_HASH,
                         note="motor mudou: campeões antigos serão revalidados antes de comparação direta")
        for k in ("world_w", "world_h", "n_units", "dev_lattice", "n_morph", "n_sound"):
            setattr(cfg, k, getattr(saved, k))                 # forma do mundo/genoma vem do checkpoint (compatibilidade)
        cfg.max_agents = max(cfg.max_agents, min(int(state["agents"]["n"]) + 64, 4 * saved.max_agents))
        uni = UniverseEngine(cfg, paths, device, logger, hw)
        uni.load_state_dict(state)
        uni.sessions += 1
        uni.history.add("session_resume", uni.tick_count, session=uni.sessions, checkpoint=info.get("loaded"),
                        fallback=info.get("fallback", False))
        info["mode"] = "resumed"
    else:
        uni = UniverseEngine(cfg, paths, device, logger, hw)
        if info.get("warning"):
            print("⚠️ " + info["warning"])
        bs = bootstrap_seconds if bootstrap_seconds is not None else (90.0 if device.type == "cuda" else 12.0)
        uni.create_world(bootstrap_seconds=bs)
        uni.sessions = 1
        info["mode"] = "created"
    if import_legacy:
        try:
            lb = LegacyBridge(paths, logger)
            summary = lb.import_research()
            if summary:
                uni.legacy_summary = {"discoveries": len(summary.get("discoveries", [])),
                                      "best_level": summary.get("best_level_global"), "source": summary.get("source_checkpoint"),
                                      "note": summary.get("note")}
                info["legacy"] = uni.legacy_summary
        except Exception as e:
            info["legacy_error"] = repr(e)
    return uni, ck, info
