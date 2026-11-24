# PROJECT GENESIS — laboratórios de universos artificiais

## NOVO: GENESIS CIVILIZATION Ω∞ — **[`GENESIS_CIVILIZATION_OMEGA.ipynb`](GENESIS_CIVILIZATION_OMEGA.ipynb)**

Universo artificial 3D multiescala + observatório 3D ao vivo (Three.js) + laboratório científico com controles.
**É um universo artificial, não uma simulação atomística do Universo real; nada aqui afirma vida real, consciência,
sentimentos subjetivos, AGI ou humanidade real.**

1. Colab → `Ambiente de execução → Alterar tipo → GPU` (A100 recomendado; sem GPU roda em DEBUG na CPU).
2. `Ambiente de execução → Executar tudo`. Autorize o Google Drive (persistência em `MyDrive/GENESIS_CIVILIZATION_OMEGA`).
3. No fim do boot aparece **🌍 OPEN GENESIS WORLD** (viewer embutido no Colab) e um **link remoto HTTPS** (Cloudflare
   quick tunnel; ngrok se `NGROK_AUTHTOKEN` estiver em Colab Secrets) para abrir no celular ou em outro computador.
   O link remoto é somente leitura; o token é aleatório por sessão e fica só no fragmento `#` da URL.
4. A pesquisa roda sozinha até o Colab encerrar (checkpoint atômico periódico). Próxima sessão: `Executar tudo` →
   o **mesmo** mundo é retomado (idade, agentes, cultura, assentamentos, fila científica, história).

O que existe (código em `civ_src/`, montado por `build_civ_notebook.py`): mundo procedural com relevo, rios, clima,
dia/noite, estações e tempestades · matéria manipulável e construção (sem `build_house()`) · DevelopmentGenome
(semente → corpo em Matter Nodes) · física massa-mola na GPU · substrato de sinais com plasticidade evoluível ·
agentes F0–F4 com promoção/rebaixamento e teste de consistência multiescala · afeto funcional A0–A9 com ablação causal
e Ethical Sentinel · grafo social, comunicação (MI com nulos), cultura e grafo de conhecimento · detecção de
assentamentos, cidades, estradas, trocas e normas · escadas H0–H12 e C0–C9 · ilhas QD (MAP-Elites, novidade, Pareto,
bandit, estagnação) com busca dirigida ao próximo nível · Scientist com fila limitada, replicação em sementes de
holdout, Pantheon revogável e DSL segura · forks contrafactuais A/B · checkpoints com shards, manifesto e SHA-256 ·
servidor FastAPI/WebSocket (msgpack) com tokens e rate limit · observatório 3D (câmeras planeta/cidade/rua/agente/
corpo/sinal/desenvolvimento/social/cultura/afeto/laboratório, timeline, replays, camadas, WebXR, modo celular).

---

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
4. **Busca dirigida pela escada** (padrão, `DIRECTED_SEARCH`): o laboratório mira o degrau logo acima do melhor
   nível alcançado. A triagem aplica +P e −P na região A e mede, na região B, transmissão reprodutível,
   não linearidade e traço de memória; o fitness pesa a medida do degrau-alvo, metade dos pais vem da fronteira
   (validados mais altos + mais próximos do próximo degrau) e os melhores são testados direto no probe do degrau
   (fail-fast). Os níveis continuam sendo concedidos só pelos probes com controles, na ordem da escada.
5. A pesquisa roda até o Colab encerrar a sessão, com checkpoint a cada poucos minutos. Pausar: botão ■ da célula
   (salva tudo). Continuar: rodar a célula de novo (mesma sessão) ou `Executar tudo` (sessão nova).
6. Parada limpa alternativa: criar o arquivo `STOP` em `MyDrive/GENESIS_OMEGA/`.

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
| Busca dirigida | foco no próximo degrau, medidas graduadas de sinal na triagem (trio contrafactual base/+P/−P), pais da fronteira, teste direto do degrau-alvo, pré-requisitos em seguida |
| Scientist | regras + valor de informação, sem p-hacking, cache, exploração/aprofundamento, ChampionLab em holdout com dominância estatística |
| Painel ao vivo | atualização a cada 1 s sem piscar (display handles), mosaico dos universos do lote atual, progresso por fase, melhor candidato, eventos, gráficos SVG; recriado periodicamente para sobreviver a recarregamentos da página; linha de status fora de notebook |
| Infra | auto-tuning de VRAM, teste de precisão e `torch.compile`, recuperação de OOM, checkpoint atômico com hash e fallback, migração, Drive, dashboard, relatórios, evidence bundles, 19 testes de boot com controles +/− |

## Desenvolvimento

O notebook é gerado a partir de `notebook_src/*.py` (formato percent `# %%`):

```bash
python build_notebook.py            # regenera GENESIS_OMEGA.ipynb
python build_notebook.py genesis.py # também escreve um script único equivalente
GENESIS_ROOT=./run GENESIS_MAX_GENERATIONS=3 python genesis.py   # execução local (DEBUG em CPU)
```

Variáveis de ambiente úteis: `GENESIS_PROFILE` (AUTO|DEBUG|FAST|RESEARCH|EXTREME), `GENESIS_MAX_HOURS`,
`GENESIS_MAX_GENERATIONS`, `GENESIS_LIVE_SECONDS`, `GENESIS_DIRECTED`, `GENESIS_USE_DRIVE`, `GENESIS_FORCE_NEW`, `GENESIS_RUN_TESTS`, `GENESIS_START_RESEARCH`.
