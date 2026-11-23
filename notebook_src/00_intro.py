# %% [markdown]
# # PROJECT GENESIS Ω
# ## Universal Computation Discovery Laboratory — Continuous Colab Research Engine
#
# > **Não construir um computador dentro de um universo.**
# > **Construir um universo no qual alguma forma de computação tenha a oportunidade de ser descoberta.**
# > E depois: **não acreditar nela até conseguirmos destruí-la, reproduzi-la e fazê-la funcionar novamente.**
#
# ### Como usar
# 1. `Ambiente de execução → Alterar o tipo de ambiente de execução → GPU` (funciona em CPU também, em modo DEBUG).
# 2. `Ambiente de execução → Executar tudo`. Autorize o Google Drive quando pedido (é o que guarda a pesquisa).
# 3. Desça até a seção **18. PESQUISA AO VIVO**: a saída dela é um **painel que se atualiza sozinho a cada segundo**
#    (universos sendo simulados agora, fase e progresso, melhor candidato, descobertas, gráficos). Não é preciso
#    parar nada nem rodar outras células para ver resultados.
# 4. A pesquisa continua sozinha até o Colab encerrar a sessão (checkpoint a cada poucos minutos).
#    Para **pausar**: botão ■ da célula (salva tudo). Para **continuar**: rode a célula de novo, ou `Executar tudo`
#    numa sessão nova — o último checkpoint válido é restaurado e a **mesma** pesquisa continua.
#
# ### Honestidade fundamental
# Isto **não** simula o Universo físico real. O Genesis cria **universos artificiais completos dentro de suas próprias regras**:
# campos contínuos `U(x,y,t) ∈ R^C` sem semântica pré-atribuída, com espaço, tempo, interações locais e distantes, topologia mutável,
# conservação parcial, dissipação, recurso abstrato limitado, ruído/perturbação e evolução das próprias leis.
# Nenhum conceito de computação (bit, CPU, memória, instrução, neurônio...) existe **dentro** do universo.
# Os experimentos (sondas causais, de memória, de canal, de transformação...) são **instrumentos externos**,
# cada um com controles positivo e negativo testados no boot.
#
# Nada aqui é chamado de vida, organismo, inteligência, consciência ou AGI. LEVEL 10 permanece reservado.

# %%
# @title ⚙️ Configuração (defaults inteligentes — nada precisa ser mudado) { display-mode: "form" }
import os

PROFILE = "AUTO"  # @param ["AUTO", "DEBUG", "FAST", "RESEARCH", "EXTREME"]
USE_GOOGLE_DRIVE = True  # @param {type:"boolean"}
LIVE_UPDATE_SECONDS = 1.0  # @param {type:"number"}
MAX_SESSION_HOURS = 24.0  # @param {type:"number"}
MAX_GENERATIONS_THIS_SESSION = 0  # @param {type:"integer"}
RUN_BOOT_TESTS = True  # @param {type:"boolean"}
START_RESEARCH = True  # @param {type:"boolean"}
FORCE_NEW_GENESIS = False  # @param {type:"boolean"}
MASTER_SEED = 20261122  # @param {type:"integer"}
GENESIS_ROOT_OVERRIDE = ""  # @param {type:"string"}

# variáveis de ambiente (execução local/automatizada) têm precedência sobre o formulário
_env = os.environ.get
PROFILE = _env("GENESIS_PROFILE", PROFILE)
USE_GOOGLE_DRIVE = _env("GENESIS_USE_DRIVE", "1" if USE_GOOGLE_DRIVE else "0") == "1"
LIVE_UPDATE_SECONDS = float(_env("GENESIS_LIVE_SECONDS", LIVE_UPDATE_SECONDS))
MAX_SESSION_HOURS = float(_env("GENESIS_MAX_HOURS", MAX_SESSION_HOURS))      # 0–24 h; checkpoints frequentes de qualquer forma
MAX_GENERATIONS_THIS_SESSION = int(_env("GENESIS_MAX_GENERATIONS", MAX_GENERATIONS_THIS_SESSION))  # 0 = sem limite
RUN_BOOT_TESTS = _env("GENESIS_RUN_TESTS", "1" if RUN_BOOT_TESTS else "0") == "1"
START_RESEARCH = _env("GENESIS_START_RESEARCH", "1" if START_RESEARCH else "0") == "1"
FORCE_NEW_GENESIS = _env("GENESIS_FORCE_NEW", "1" if FORCE_NEW_GENESIS else "0") == "1"
MASTER_SEED = int(_env("GENESIS_SEED", MASTER_SEED))
GENESIS_ROOT_OVERRIDE = _env("GENESIS_ROOT", GENESIS_ROOT_OVERRIDE)
