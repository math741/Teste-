# %% [markdown]
# # PROJECT GENESIS Ω
# ## Universal Computation Discovery Laboratory — Continuous Colab Research Engine
#
# > **Não construir um computador dentro de um universo.**
# > **Construir um universo no qual alguma forma de computação tenha a oportunidade de ser descoberta.**
# > E depois: **não acreditar nela até conseguirmos destruí-la, reproduzi-la e fazê-la funcionar novamente.**
#
# ### Como usar
# 1. `Runtime → Change runtime type → GPU` (funciona em CPU também, em modo DEBUG).
# 2. `Runtime → Run all`.
# 3. O notebook faz: **BOOT → HARDWARE → TEST SUITE → CHECKPOINT SEARCH → NEW GENESIS / RESUME → EVOLUÇÃO → TRIAGEM → NOVELTY → SCIENTIST → PROMOÇÕES → VALIDAÇÃO → CHECKPOINT → REPETE**.
# 4. Se a sessão cair, abra de novo e rode `Run all`: o estado é restaurado do último checkpoint válido (no Google Drive, se montado) e a **mesma** pesquisa continua.
# 5. Para parar limpo: botão *stop* da célula (o Genesis captura a interrupção, salva checkpoint e relatório) ou crie o arquivo `STOP` na pasta do projeto.
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
# ============================================================================
# CÉLULA 1 — CONFIGURAÇÃO DO USUÁRIO (defaults inteligentes; nada precisa ser mudado)
# ============================================================================
import os

PROFILE = os.environ.get("GENESIS_PROFILE", "AUTO")          # AUTO | DEBUG | FAST | RESEARCH | EXTREME
USE_GOOGLE_DRIVE = os.environ.get("GENESIS_USE_DRIVE", "1") == "1"   # recomendado: a continuidade entre sessões depende disso
GENESIS_ROOT_OVERRIDE = os.environ.get("GENESIS_ROOT", "")   # pasta alternativa (opcional)
MAX_SESSION_HOURS = float(os.environ.get("GENESIS_MAX_HOURS", "11.0"))   # encerramento limpo antes do limite do Colab
MAX_GENERATIONS_THIS_SESSION = int(os.environ.get("GENESIS_MAX_GENERATIONS", "0"))  # 0 = sem limite
RUN_BOOT_TESTS = os.environ.get("GENESIS_RUN_TESTS", "1") == "1"
START_RESEARCH = os.environ.get("GENESIS_START_RESEARCH", "1") == "1"
FORCE_NEW_GENESIS = os.environ.get("GENESIS_FORCE_NEW", "0") == "1"  # True = ignora checkpoints e cria nova pesquisa
MASTER_SEED = int(os.environ.get("GENESIS_SEED", "20261122"))
