# PROJECT GENESIS Ω — Universal Computation Discovery Laboratory

Um único notebook para Google Colab: **[`GENESIS_OMEGA.ipynb`](GENESIS_OMEGA.ipynb)**.

> Não construir um computador dentro de um universo. Construir um universo no qual alguma forma de computação
> tenha a oportunidade de ser descoberta — e não acreditar nela até conseguirmos destruí-la, reproduzi-la e fazê-la funcionar novamente.

## Uso

1. Abra o notebook no Colab → `Ambiente de execução → Alterar tipo → GPU`.
2. `Ambiente de execução → Executar tudo`. Autorize o Google Drive quando pedido (é o que guarda a pesquisa).
3. Na seção **18. ▶ PESQUISA AO VIVO**, a saída da célula é um **painel que se atualiza sozinho a cada segundo**:
   universos sendo simulados naquele instante, fase com barra de progresso, contadores e velocidade, melhor candidato
   com scorecard, anúncios de descoberta, eventos recentes e gráficos por geração (convergência, MAP-Elites, escada,
   zoológico). Não é preciso parar nada para ver resultados.
4. A pesquisa roda até o Colab encerrar a sessão, com checkpoint a cada poucos minutos. Pausar: botão ■ da célula
   (salva tudo). Continuar: rodar a célula de novo (mesma sessão) ou `Executar tudo` (sessão nova).
5. Parada limpa alternativa: criar o arquivo `STOP` em `MyDrive/GENESIS_OMEGA/`.

No Colab o código fica escondido atrás dos títulos das seções (duplo clique mostra) e as configurações aparecem
como formulário na primeira célula.

Sem GPU o notebook roda em modo `DEBUG` (mundo 32², população 24).

## O que tem dentro

| Módulo | Conteúdo |
|---|---|
| Física | campo `U[B,C,S,S]` sem semântica; kernels evoluídos via FFT (anel, gaussiano, Moore, von Neumann, anisotrópico); interação, não linearidades mistas, reação, difusão, advecção, inércia (ondas), atraso, histerese, dissipação, conservação parcial, saturação, recurso abstrato, links distantes com plasticidade local, bordas periódica/refletora/absorvente, assincronia, dt adaptativo (numérico) |
| Genoma | ~45 genes de leis + parâmetros de mutação auto-adaptativos; mutações gaussianas, fortes, estruturais (duplicar/remover canal, kernel), temporais, topológicas, crossover; bandit UCB de operadores |
| Quality-Diversity | MAP-Elites, novelty search, especiação genótipo+fenótipo, zoo (STATIC … UNKNOWN, ANOMALOUS), arquivos de anomalias/falhas/campeões, linhagem, surpresa |
| Validation Engine (VE-1.0) | causalidade contrafactual, memória M(T), canal (MI, latência, capacidade), transformação (XOR), composição (paridade), generalização (não vistos), dano/recuperação, robustez (8 perturbações), adaptação, replicação, ablação, análise avançada (TE, Granger, lei efetiva, PCA, atratores, objetos, fase, histerese) |
| Scientist | regras + valor de informação, sem p-hacking, cache, exploração/aprofundamento, ChampionLab em holdout com dominância estatística |
| Painel ao vivo | atualização a cada 1 s sem piscar (display handles), mosaico dos universos do lote atual, progresso por fase, melhor candidato, eventos, gráficos SVG; recriado periodicamente para sobreviver a recarregamentos da página; linha de status fora de notebook |
| Infra | auto-tuning de VRAM, teste de precisão e `torch.compile`, recuperação de OOM, checkpoint atômico com hash e fallback, migração, Drive, dashboard, relatórios, evidence bundles, 18 testes de boot com controles +/− |

## Desenvolvimento

O notebook é gerado a partir de `notebook_src/*.py` (formato percent `# %%`):

```bash
python build_notebook.py            # regenera GENESIS_OMEGA.ipynb
python build_notebook.py genesis.py # também escreve um script único equivalente
GENESIS_ROOT=./run GENESIS_MAX_GENERATIONS=3 python genesis.py   # execução local (DEBUG em CPU)
```

Variáveis de ambiente úteis: `GENESIS_PROFILE` (AUTO|DEBUG|FAST|RESEARCH|EXTREME), `GENESIS_MAX_HOURS`,
`GENESIS_MAX_GENERATIONS`, `GENESIS_LIVE_SECONDS`, `GENESIS_USE_DRIVE`, `GENESIS_FORCE_NEW`, `GENESIS_RUN_TESTS`, `GENESIS_START_RESEARCH`.
