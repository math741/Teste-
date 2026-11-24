# %% [markdown]
# ## 25 — EXPORT / REPORTS: relatório de sessão (com limitações científicas explícitas)

# %%
LIMITATIONS = ["universo artificial (não é simulação atomística do Universo real)",
               "abstrações físicas: massa-mola simplificada, clima procedural, recurso abstrato",
               "coarse-graining entre escalas (F0–F4) com erro medido no probe MS, não nulo",
               "escala temporal artificial (ticks); 'dia'/'estação' são ciclos artificiais",
               "fidelidade limitada pelo hardware; corpos completos só para um subconjunto",
               "nenhuma inferência direta de consciência ou experiência subjetiva (afeto = funcional)",
               "nenhuma equivalência automática a humanos reais; níveis C são hipóteses de simulação",
               "sem regeneração estrutural de corpos após dano (só robustez funcional)",
               "kernels CUDA com somas atômicas podem não ser bit-a-bit determinísticos (registrado nos testes)"]


def generate_report(uni: UniverseEngine, hw: Dict[str, Any], boot: Dict[str, Any], stop_reason: str = "") -> str:
    p = uni.paths
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dem = uni.demography.last(400)
    figs = []
    try:
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        if dem:
            ax[0].plot([d["tick"] for d in dem], [d["population"] for d in dem], color="#3d7bff"); ax[0].set_title("população")
            ax[0].plot([d["tick"] for d in dem], [d.get("f0", 0) for d in dem], color="#999", lw=1)
        for isl in uni.qd.islands:
            ax[1].plot(isl.history, lw=1)
        ax[1].set_title("QD: melhor fitness por ilha (≠ evidência)")
        pr = uni.prof.report()
        ax[2].barh(list(pr.keys())[:8], [v["share"] for v in list(pr.values())[:8]], color="#46c3a0"); ax[2].set_title("tempo por subsistema")
        fig.tight_layout()
        fp = os.path.join(p.plots, f"session_{ts}.png"); fig.savefig(fp, dpi=90); plt.close(fig)
        figs.append(fp)
    except Exception as e:
        figs.append(f"plot error: {e}")
    civ = uni.last.get("civ", {})
    lad = ladder_status(uni)
    lines = [f"# GENESIS CIVILIZATION Ω∞ — relatório de sessão {ts}", "",
             "> universo artificial; comportamento observado ≠ mecanismo demonstrado ≠ interpretação ≠ experiência subjetiva", "",
             "## hardware e execução",
             f"- device: {hw.get('device')} · GPU: {hw.get('gpu_name')} · VRAM: {human_bytes(hw.get('vram_total', 0))} · torch {hw.get('torch')}",
             f"- perfil: {uni.cfg.profile} · mundo {uni.ws.W}×{uni.ws.H} · tick {uni.tick_count:,} · sessões {uni.sessions}",
             f"- runtime total: {fmt_hms(uni.total_runtime)} · motivo do fim: {stop_reason}",
             f"- testes de boot: {boot.get('tests_passed')}/{boot.get('tests_total')} · autotune: {to_jsonable(boot.get('autotune', {}).get('final'))}",
             "", "## evolução e população",
             f"- avaliações QD: {uni.qd.evaluations:,} · gerações QD: {uni.qd.generations} · cobertura MAP-Elites {uni.qd.map.coverage():.3f}",
             f"- população viva {uni.pool.n_alive():,} · F0 {uni.f0.total() if uni.f0 else 0:.0f} · nascimentos totais {uni.lineage.total:,}",
             f"- fidelidade: {dict(uni.fidelity_log)}", "", "## escadas (evidência)"]
    for grp in ("H", "A"):
        for r in lad[grp]:
            tag = "TESTED" if r.get("validated") else ("NOT YET VALIDATED" if r["tested"] else "não testado")
            lines.append(f"- {r['name']}: {tag}" + (f" · efeito {r.get('effect'):.2f} · {r.get('discovery')}" if r.get("validated") and r.get("effect") is not None else ""))
    for r in lad["C"]:
        lines.append(f"- {r['name']}: {'SIMULATION HYPOTHESIS (critérios atingidos)' if r['achieved'] else 'NOT YET VALIDATED'}")
    lines += ["", "## descobertas"] + [f"- {d['id']} · {d['level_name']} · {d['claim']} · réplicas {d['passed_replications']}/{d['replications']}"
                                       for d in uni.archive.discoveries[-40:]]
    lines += ["", "## mundo, cultura, comunicação, sociedade",
              f"- assentamentos {civ.get('settlements', 0)} · cidades {civ.get('cities', 0)} · blocos {civ.get('blocks', 0):.0f} · estradas {civ.get('roads', 0)}",
              f"- casa-like (função de abrigo demonstrada): {uni.settlements.house_like}",
              f"- comunicação (MI sinal×mundo): {to_jsonable(uni.last.get('comm', {}))}",
              f"- cultura: {to_jsonable({k: v for k, v in uni.last.get('culture', {}).items() if k != 'techniques'})}",
              f"- social: {to_jsonable({k: v for k, v in uni.last.get('social', {}).items() if k != 'dissolved'})}",
              f"- eventos históricos: {uni.history.total} · ETHICAL_REVIEW_TRIGGER: {len(uni.sentinel.triggers)}",
              f"- falhas arquivadas: {len(uni.archive.failures)} · anomalias: {len(uni.archive.weird)} · convergências: {len(uni.archive.convergence)}",
              f"- forks contrafactuais: {list(uni.forks.keys())[-10:]}", "", "## limitações atuais"] + [f"- {x}" for x in LIMITATIONS]
    lines += ["", "## status das alegações", "- IMPLEMENTED: todo módulo listado no notebook roda neste build",
              "- TESTED: só o que passou nos testes de boot e nos probes com controles (acima)",
              "- BENCHMARKED: autotune + probe MS (erro multiescala) + profiler", "- SIMULATION HYPOTHESIS: níveis C, A9, H12",
              "- NOT YET VALIDATED: todo nível sem descoberta registrada"]
    md = "\n".join(lines)
    out = os.path.join(p.reports, f"report_{ts}.md")
    with open(out, "w") as f:
        f.write(md)
    with open(os.path.join(p.reports, "latest_report.md"), "w") as f:
        f.write(md)
    return out
