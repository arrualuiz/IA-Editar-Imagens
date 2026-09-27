# Continuar o projeto em outro computador

Passo a passo para montar o robô de fotos do zero numa máquina nova e continuar de onde
parou. Escrito em 27/09/2026, com o que está instalado e funcionando no PC atual.

## Resumo em uma frase

Instale **Git** e **uv**, clone o repositório, rode `uv venv` + `uv pip install`, e copie
à mão as pastas de fotos e de dados — o Git leva só o código, nunca as fotos.

## 1. O que instalar

| Programa | Para quê | Como instalar (Windows) |
|---|---|---|
| **Git** | baixar e versionar o código | https://git-scm.com/download/win |
| **uv** | instala o Python certo e cria o ambiente (`.venv`) | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| **Python 3.11 ou mais novo** | roda o robô | o `uv` instala sozinho no passo 2 (o PC atual usa **3.14.6**) |
| Navegador | página de revisão (Etapa 4, `http://127.0.0.1:8765`) | qualquer um |
| VS Code (opcional) | editar o código | https://code.visualstudio.com |

**Banco de dados: nenhum.** O projeto não usa MySQL, PostgreSQL, SQLite nem nada que
precise ser instalado ou ficar rodando. Tudo que o robô "lembra" fica em arquivos comuns
dentro do projeto:

| Arquivo | O que guarda |
|---|---|
| `dados/correcoes.jsonl` | suas correções, uma linha JSON por foto — é o "treino humano" (aparece a partir da Etapa 4) |
| `dados/sugestoes/<lote>.json` | o que o robô sugeriu para cada lote (pode ser refeito) |
| `modelos/orientacao_vNNN.pt` | os modelos treinados do PyTorch (a partir da Etapa 3) |
| `config.yaml` | todos os parâmetros (vai no Git) |

**GPU: não é necessária.** O PC atual roda em CPU (`torch 2.14.0+cpu`). Se a máquina
nova tiver placa NVIDIA, dá para instalar o PyTorch com CUDA depois e o robô usa sozinho
(`orientacao.dispositivo: auto` no `config.yaml`).

**Internet:** só para instalar. Quando a Etapa 3 existir, o primeiro treino vai baixar
uma vez os pesos pré-treinados do MobileNetV3 (uns 10 MB, em `%USERPROFILE%\.cache\torch`).
Nenhuma foto sai do computador.

## 2. Baixar o código e instalar

No PowerShell:

```powershell
mkdir C:\DEV\iA -Force
cd C:\DEV\iA
git clone https://github.com/arrualuiz/IA-Editar-Imagens.git "google photos old"
cd "google photos old"

uv python install 3.14
uv venv --python 3.14
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev]"
```

O nome da pasta pode ser qualquer um: os caminhos do `config.yaml` são relativos. Neste
PC ela se chama `C:\DEV\iA\google photos old`.

A linha do `--index-url` instala o PyTorch só para CPU (menor e igual ao do PC atual).
Sem ela também funciona, mas baixa mais.

### Conferir

```powershell
.\.venv\Scripts\Activate.ps1
python -m robo_fotos --help
python -m robo_fotos info
python -m pytest                 # hoje: 58 testes passando
```

Se o PowerShell bloquear o `Activate.ps1`, rode uma vez
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. Sem ativar, também dá para chamar
direto: `.\.venv\Scripts\python -m robo_fotos info`.

Evite `uv run` neste projeto: como ele tem `pyproject.toml`, o `uv run` cria um `uv.lock`
e sincroniza o ambiente por conta própria, podendo trocar o PyTorch só-CPU por outro.

### Versões que estão funcionando no PC atual (27/09/2026)

```
python 3.14.6        uv 0.11.23
pillow 12.3.0        pillow-heif 1.8.0
opencv-python-headless 5.0.0.93   numpy 2.5.3
torch 2.14.0+cpu     torchvision 0.29.0
fastapi 0.141.1      uvicorn 0.54.0     python-multipart 0.0.32
PyYAML 6.0.3         pytest 9.1.1
```

Se alguma versão mais nova quebrar algo, instale exatamente estas, por exemplo
`uv pip install "numpy==2.5.3"`.

## 3. Pastas e arquivos que o Git NÃO leva

As fotos e os dados gerados estão no `.gitignore` de propósito (são grandes e pessoais).
O clone traz essas pastas vazias (com um `.gitkeep`). Copie do PC antigo o que interessa:

| Pasta | O que tem | No PC novo | Hoje no PC atual |
|---|---|---|---|
| `entrada/` | fotos originais do Google Fotos / Takeout (somente leitura) | **copiar** (ou baixar de novo do Google Fotos) | 127 fotos, ~60 MB |
| `treino_base/` | fotos já na orientação certa, para o treino inicial | **copiar** se tiver | vazia |
| `dados/correcoes.jsonl` | suas correções — o mais valioso, não dá para refazer | **copiar sempre** que existir | ainda não existe |
| `modelos/*.pt` | modelos treinados | **copiar** para não treinar de novo | vazia |
| `saida/` | fotos corrigidas | copiar se ainda não subiu no Google Fotos | vazia |
| `lotes/` | lote_001, lote_002… | copiar, ou refazer com `dividir` (mesma `entrada/` gera os mesmos lotes) | 3 lotes |
| `dados/sugestoes/`, `dados/debug_corte/`, `dados/demo_antes_depois/` | sugestões e imagens de conferência | não precisa: são refeitas pelos comandos | |
| `.venv/` | ambiente Python | **nunca copiar**: recriar com o passo 2 | |

Jeito rápido de levar tudo (no PC antigo, dentro da pasta do projeto):

```powershell
Compress-Archive -Path entrada, treino_base, dados, modelos, saida, lotes -DestinationPath "$HOME\Desktop\robo-fotos-dados.zip"
```

No PC novo, extraia o zip dentro da pasta do projeto, por cima das pastas vazias. Esse zip
tem fotos pessoais: leve por pendrive ou pelo seu Drive, **nunca pelo Git**.

## 4. Onde o projeto parou (27/09/2026)

- [x] Etapa 0 — Setup (CLI, `config.yaml`, estrutura)
- [x] Etapa 1 — Lotes (`dividir`: 127 fotos → `lote_001` a `lote_003`)
- [x] Etapa 2 — Corte (`sugerir-corte`, métodos `bordas` e `fundo`)
- [ ] **Etapa 3 — Orientação v1 ← próxima.** `orientacao.py` e `treino.py` ainda são
  esqueletos. Antes de começar, coloque algumas centenas de fotos certas em `treino_base/`.
- [ ] Etapas 4 a 8 — revisão no navegador, exportar, treino com correções, evolução.

O plano completo de cada etapa, as regras (nunca mexer em `entrada/`, preservar EXIF,
Orientation = 1…) e a seção **Progresso** estão no [`CLAUDE.md`](CLAUDE.md). Para
continuar com o Claude Code, abra a pasta do projeto e peça: *"continue da Etapa 3
conforme o CLAUDE.md"*.

## 5. Checklist do PC novo

- [ ] `git --version` e `uv --version` respondem
- [ ] `python -m pytest` passa (com o `.venv` ativado)
- [ ] `python -m robo_fotos info` mostra as fotos em `entrada/` e os lotes
- [ ] `dados/correcoes.jsonl` e `modelos/` copiados (quando existirem)
- [ ] Antes de trabalhar: `git pull`. Ao terminar: `git add`, `git commit`, `git push`
