# %% [markdown]
# ## 12. Relatórios, evidence bundles, saída de campeão e dashboard
# Linguagem obrigatoriamente contida: nada é chamado de vida, organismo, inteligência, consciência ou AGI.
# Todo relatório lista limitações.

# %%
LIMITATIONS = [
    "Universo artificial: nada aqui simula a física real; resultados valem dentro destas regras.",
    "Decodificadores externos (ridge/kNN) medem informação acessível a um observador; não provam que o universo 'usa' essa informação.",
    "Transformação/composição são definidas operacionalmente (XOR/paridade linearmente decodificáveis); outras formas de computação podem passar despercebidas.",
    "Estimativas de MI/TE são plug-in com correção simples de viés; amostras pequenas.",
    "Tempo de GPU não é energia física; nenhum resultado implica viabilidade de hardware.",
    "Adaptação = mudança de comportamento com manutenção de função; não implica aprendizado, vida ou agência.",
    "Um único notebook/Colab limita tamanho de mundo, horizonte temporal e número de sementes.",
    "Conservação global (lei opcional do genoma) é não local: permite influência instantânea fraca à distância; "
    "pesquisas criadas antes da v1.1 usam ainda o dt adaptativo global (física v1), que também acopla o universo inteiro.",
    "Busca dirigida: medidas graduadas de triagem (transmissão, não linearidade, traço) orientam a evolução; só os probes "
    "com controles concedem níveis.",
]


def discovery_report_text(rec: Dict[str, Any]) -> str:
    ev = "\n".join(f"  - {e}" for e in rec.get("evidence", []))
    return (f"\n╔══════════════════════════════════════╗\n║           DISCOVERY EVENT            ║\n╚══════════════════════════════════════╝\n"
            f"Discovery: {rec['discovery_id']}\nCandidate: {rec['candidate_id']}\n"
            f"Previous level: {rec['previous_level']}\nNew level: {rec['new_level']} ({rec['level_name']})\n"
            f"Probe: {rec['probe']}  status={rec['status']}  score={rec['score']:.3f}  p={rec['p_value']:.2e}\n"
            f"Evidence:\n{ev}\nPrimeira reação: não comemorar — tentar quebrar (replicação, ablação, holdout).\n")


def write_evidence_bundle(lab: "GenesisLab", rec: Dict[str, Any]) -> str:
    d = os.path.join(lab.paths.discoveries, f"{rec['discovery_id']}_L{rec['new_level']}_{rec['candidate_id']}")
    os.makedirs(d, exist_ok=True)
    g = lab.store[rec["candidate_id"]]
    np.savez_compressed(os.path.join(d, "genome.npz"), **{k: np.asarray(v) for k, v in g.params.items()})
    env = lab.envs[0]
    frames, init = simulate_frames(lab, g, env, steps=160, every=2, seed=derive_seed(rec["discovery_id"], "viz"))
    np.savez_compressed(os.path.join(d, "initial_state.npz"), U0=init, env=json.dumps(env.to_dict()),
                        seed=derive_seed(rec["discovery_id"], "viz"))
    res = lab.results.get(rec["result_id"])
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(to_jsonable({"discovery": {k: v for k, v in rec.items() if k != "genome"},
                               "result": res.to_dict() if res else None,
                               "scorecard": lab.scorecard(rec["candidate_id"]),
                               "descriptor": {k: v for k, v in lab.desc.get(rec["candidate_id"], {}).items()}}), f, indent=1)
    with open(os.path.join(d, "protocol.json"), "w") as f:
        json.dump(to_jsonable(rec["protocol"]), f, indent=1)
    plot_frames(frames, f"{rec['candidate_id']} — L{rec['new_level']}", os.path.join(d, "frames.png"), show=False)
    if res is not None:
        if res.probe == "memory":
            plot_memory_curve(res, os.path.join(d, "memory_curve.png"), show=False)
        if res.probe == "causality":
            plot_causal_map(res, os.path.join(d, "causal_map.png"), show=False)
        if res.probe == "recovery":
            plot_damage_curve(res, os.path.join(d, "damage_curve.png"), show=False)
    if rec["new_level"] >= 2:
        save_animation(frames, os.path.join(d, "animation.gif"))
    with open(os.path.join(d, "REPORT.md"), "w") as f:
        f.write(f"# DISCOVERY {rec['discovery_id']}\n\n```\n{discovery_report_text(rec)}\n```\n\n## Scorecard\n\n")
        for k, v in lab.scorecard(rec["candidate_id"]).items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Linhagem\n\n```\n" + lab.lineage.tree(rec["candidate_id"], depth=2) + "\n```\n\n## Limitações\n\n")
        f.writelines(f"- {l}\n" for l in LIMITATIONS)
    return d


def write_champion_outputs(lab: "GenesisLab", cid: str, battery: Dict[str, TestResult], entry: Dict[str, Any]):
    d = lab.paths.champion
    g = lab.store[cid]
    lab.say("╔══════════════════════════════════════╗\n║       GENESIS CHAMPION DETECTED      ║\n╚══════════════════════════════════════╝")
    torch.save({"genome": genome_to_record(lab.space, g), "config": lab.cfg.to_dict(), "entry": entry,
                "version": GENESIS_VERSION}, os.path.join(d, "GENESIS_CHAMPION.pt"))
    with open(os.path.join(d, "GENESIS_CHAMPION_EXPERIMENTS.json"), "w") as f:
        json.dump(to_jsonable({k: r.to_dict() for k, r in battery.items()} |
                              {"all_results": {k: lab.results[r].to_dict() for k, r in lab.tests.get(cid, {}).items() if r in lab.results}}), f, indent=1)
    anc = [cid] + lab.lineage.ancestors(cid, 500)
    with open(os.path.join(d, "GENESIS_CHAMPION_LINEAGE.json"), "w") as f:
        json.dump(to_jsonable({a: lab.lineage.records.get(a) for a in anc}), f, indent=1)
    with open(os.path.join(d, "GENESIS_CHAMPION_METRICS.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["field", "value"])
        for k, v in lab.scorecard(cid).items():
            w.writerow([k, v])
        for k, v in lab.desc.get(cid, {}).items():
            if not isinstance(v, list):
                w.writerow([f"descriptor.{k}", v])
    runner = entry["dominance"].get("rival")
    with open(os.path.join(d, "GENESIS_CHAMPION_REPORT.md"), "w") as f:
        f.write(f"# GENESIS CHAMPION — {cid}\n\nGerado em {iso_now()} · Genesis {GENESIS_VERSION} · {VALIDATION_ENGINE_VERSION}\n\n")
        f.write("## Gate\n\n" + "\n".join(f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in entry["gate"].items()))
        f.write(f"\n\n## Dominância\n\n```\n{json.dumps(to_jsonable(entry['dominance']), indent=1)}\n```\n")
        f.write(f"\nRunner-up preservado: {runner}\n\n## Scorecard\n\n")
        f.writelines(f"- **{k}**: {v}\n" for k, v in lab.scorecard(cid).items())
        f.write("\n## Genoma\n\n```\n" + g.summary(lab.space) + "\n```\n\n## Limitações obrigatórias\n\n")
        f.writelines(f"- {l}\n" for l in LIMITATIONS)
    lab.logger.event("GENESIS_CHAMPION", echo=True, candidate=cid, runner_up=runner)


def dashboard_text(lab: "GenesisLab", status: str = "SEARCHING") -> str:
    best = lab.best_candidate()
    sc = lab.scorecard(best) if best else {}
    gpu = lab.hw.get("gpu_name", "CPU")
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9; tot = lab.hw.get("vram_total", 0) / 1e9
        vram = f"{used:.2f} / {tot:.1f} GB"
    else:
        vram = "n/a"
    nxt = max(0.0, lab.cfg.checkpoint_every_minutes * 60 - (time.time() - lab.last_checkpoint_ts))
    last = lab.ckpt.last_save_ts
    val = lambda k: sc.get(k, STATUS_NOT_TESTED)
    lines = [
        "╔══════════════════════════════════════╗",
        "║          PROJECT GENESIS Ω           ║",
        "║ UNIVERSAL DISCOVERY ENGINE — ONLINE  ║",
        "╚══════════════════════════════════════╝",
        f"Session Runtime:      {fmt_hms(time.time() - lab.session_start)}   (sessão #{lab.counters['sessions']})",
        f"Total Runtime:        {fmt_hms(lab.runtime_total())}",
        f"Generation:           {lab.counters['generation']}",
        f"Universes Evaluated:  {lab.counters['evaluations']:,}",
        f"Simulation Steps:     {lab.counters['sim_steps']:,} universo·passos   (~{lab.throughput['universe_steps_per_s']:,.0f}/s, ~{lab.throughput['evals_per_hour_est']:,.0f} avaliações/h — estimativa)",
        f"GPU:                  {gpu}",
        f"VRAM:                 {vram}",
        f"Population:           {len(lab.population)}   batch={lab.cfg.batch_size}  world={lab.cfg.world_size}²×{lab.cfg.channels_max}",
        f"Species:              {len(lab.species.reps)} vivas / {len(lab.species.extinct)} extintas",
        f"Novelty Zoo:          {len(lab.novelty.ids)} | zoo: " + ", ".join(f"{k[:5]}={v}" for k, v in lab.zoo.counts().items() if v),
        f"MAP-Elites Coverage:  {lab.map.coverage():.1%} ({len(lab.map.cells)} células)",
        f"Best Discovery Level: {sc.get('level', -1)} ({LEVEL_NAMES.get(sc.get('level', -1), '-')})",
        f"Current Best Candidate: {best}",
        f"Best Fitness:         {max([lab.quality.get(c, -1) for c in lab.population], default=-1):.4f}   (fitness ≠ evidência; evidência={sc.get('evidence', 0)})",
        f"Persistence:          {val('persistence')}",
        f"Complexity:           {val('complexity')}",
        f"Causality:            {val('causality')}",
        f"Memory:               {val('memory')}",
        f"Channel:              {val('channel')}",
        f"Transformation:       {val('transformation')}",
        f"Composition:          {val('composition')}",
        f"Generalization:       {val('generalization')}",
        f"Recovery:             {val('recovery')}",
        f"Robustness:           {val('robustness')}",
        f"Novel Discoveries:    {lab.counters['discovery_events']}   Experiments: {lab.counters['experiments']}  (fila: {len(lab.scientist.queue)})",
        f"Anomalies:            {len(lab.anomalies.entries)}   Failures archived: {len(lab.failures.entries)}",
        f"Provisional Champions: {len(lab.champions.entries)}   Pantheon: {len(lab.pantheon)}",
        f"Checkpoint:           {lab.paths.checkpoints}  ({'Drive' if lab.paths.on_drive else 'local'})",
        f"Last Checkpoint:      {datetime.datetime.fromtimestamp(last).strftime('%H:%M:%S') if last else '-'}   Next: ~{fmt_hms(nxt)}",
        f"Stagnation stage:     {lab.stagnation['stage']}  sigma×{lab.stagnation['sigma_mult']:.2f}  novelty×{lab.stagnation['novelty_weight']:.2f}",
        f"STATUS:               {status}",
    ]
    return "\n".join(lines)


def show_dashboard(lab: "GenesisLab", status: str = "SEARCHING", plots: bool = False):
    if IN_NOTEBOOK:
        clear_output(wait=True)
    print(dashboard_text(lab, status))
    if plots and IN_NOTEBOOK:
        plot_map_elites(lab.map, None, True)
        plot_convergence(lab.history, None, True)


def session_report(lab: "GenesisLab", reason: str) -> str:
    path = os.path.join(lab.paths.reports, f"session_{lab.counters['sessions']:04d}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    best = lab.best_candidate()
    lvl_counts: Dict[int, int] = {}
    for l in lab.levels.values():
        lvl_counts[l] = lvl_counts.get(l, 0) + 1
    with open(path, "w") as f:
        f.write(f"# Genesis Ω — relatório de sessão #{lab.counters['sessions']}\n\nMotivo do encerramento: **{reason}**\n\n")
        f.write("```\n" + dashboard_text(lab, "SESSION END") + "\n```\n\n")
        f.write("## Distribuição de DiscoveryLevel (candidatos retidos)\n\n")
        f.writelines(f"- L{k} {LEVEL_NAMES.get(k, '')}: {v}\n" for k, v in sorted(lvl_counts.items()))
        f.write("\n## Melhor candidato\n\n")
        if best:
            f.writelines(f"- **{k}**: {v}\n" for k, v in lab.scorecard(best).items())
            f.write("\n```\n" + lab.lineage.tree(best, depth=2) + "\n```\n")
        f.write("\n## Promoções (auditoria)\n\n")
        f.writelines(f"- gen {a['generation']}: {a['candidate_id']} L{a['from']}→L{a['to']} (result {a['result_id']})\n" for a in lab.audit[-50:])
        f.write("\n## Failure archive (últimos)\n\n")
        f.writelines(f"- {e['genome_id']}: {e['reason']}\n" for e in lab.failures.entries[-20:])
        f.write("\n## Bandit de operadores\n\n" + json.dumps(lab.bandit.probs()) + "\n")
        f.write("\n## Sinais de METRIC EXPLOIT\n\n" + (json.dumps(to_jsonable(lab.metric_flags[-10:])) or "nenhum") + "\n")
        f.write("\n## Limitações\n\n")
        f.writelines(f"- {l}\n" for l in LIMITATIONS)
        if lab.champion is None:
            f.write("\n**Nenhum GENESIS CHAMPION nesta sessão. Isso não é erro: a pesquisa continua na próxima sessão.**\n")
    return path
