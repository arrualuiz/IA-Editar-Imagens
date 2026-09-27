# Continuar o projeto em outro computador

Passo a passo para montar o robô de fotos do zero numa máquina nova — Linux, Mac ou
Windows — e continuar de onde parou.

Última atualização: **27/09/2026, 16h25**, no PC Windows original, imediatamente antes
de pausar o trabalho.

## Resumo em uma frase

Instale **Git** e **uv**, clone o repositório, rode `uv venv` + `uv pip install`, e copie
à mão as pastas de fotos e de dados — o Git leva só o código, nunca as fotos.

---

## 1. O que instalar

Só três coisas: Git, uv e um navegador. O `uv` instala o Python sozinho.

### Linux (Ubuntu / Debian / Fedora)

```bash
# Git
sudo apt install git          # Debian, Ubuntu, Mint
# sudo dnf install git        # Fedora
# sudo pacman -S git          # Arch

# uv (instala o Python e cria o ambiente)
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL                   # recarrega o terminal para achar o uv
```

**Nenhuma biblioteca de sistema é necessária.** O projeto usa
`opencv-python-headless` justamente para isso: a versão normal do OpenCV pede
`libGL`, `libglib` e companhia, que muita instalação de Linux não tem. A versão
headless não abre janela nenhuma e dispensa tudo isso. O HEIC também não precisa
de pacote do sistema — o `pillow-heif` já traz o `libheif` embutido.

### Mac

```bash
xcode-select --install                                  # traz o Git
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Em Mac com chip Apple (M1 em diante), o PyTorch usa a GPU por MPS
automaticamente (`orientacao.dispositivo: auto` no `config.yaml`).

### Windows (PowerShell)

```powershell
# Git: https://git-scm.com/download/win
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### O que NÃO precisa instalar

**Banco de dados: nenhum.** Nada de MySQL, PostgreSQL ou SQLite. Tudo que o robô
"lembra" fica em arquivos comuns dentro do projeto:

| Arquivo | O que guarda |
|---|---|
| `dados/correcoes.jsonl` | suas correções, uma linha JSON por foto — é o "treino humano" (aparece a partir da Etapa 4) |
| `dados/sugestoes/<lote>.json` | o que o robô sugeriu para cada lote (pode ser refeito) |
| `modelos/orientacao_vNNN.pt` | os modelos treinados do PyTorch (a partir da Etapa 3) |
| `config.yaml` | todos os parâmetros (vai no Git) |

**GPU: não é necessária.** O PC antigo rodava em CPU (`torch 2.14.0+cpu`) e dava conta.

**Internet:** só para instalar. Quando a Etapa 3 existir, o primeiro treino vai baixar
uma vez os pesos pré-treinados do MobileNetV3 (uns 10 MB, em `~/.cache/torch`).
Nenhuma foto sai do computador.

---

## 2. Baixar o código e instalar

### Linux / Mac

```bash
mkdir -p ~/dev/ia && cd ~/dev/ia
git clone https://github.com/arrualuiz/IA-Editar-Imagens.git robo-fotos
cd robo-fotos

uv python install 3.14
uv venv --python 3.14
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev]"
```

### Windows (PowerShell)

```powershell
mkdir C:\DEV\iA -Force
cd C:\DEV\iA
git clone https://github.com/arrualuiz/IA-Editar-Imagens.git robo-fotos
cd robo-fotos

uv python install 3.14
uv venv --python 3.14
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev]"
```

O nome da pasta pode ser qualquer um: os caminhos do `config.yaml` são relativos à raiz
do projeto. No PC antigo ela se chamava `C:\DEV\iA\google photos old` — com espaços, o
que funciona, mas num Linux vale usar um nome sem espaço para facilitar a vida no
terminal.

A linha do `--index-url` instala o PyTorch só para CPU: é bem menor (uns 200 MB em vez de
2,5 GB) e é o que estava rodando. Se a máquina nova tiver placa NVIDIA e você quiser
usar a GPU, troque essa linha por `uv pip install torch torchvision` e o robô passa a
usar CUDA sozinho.

### Conferir que deu certo

```bash
source .venv/bin/activate        # Linux e Mac
# .\.venv\Scripts\Activate.ps1   # Windows

python -m robo_fotos --help
python -m robo_fotos info
python -m pytest                 # em 27/09/2026: 58 testes passando
```

Se os 58 testes passarem, o ambiente está igual ao do PC antigo.

**Windows:** se o PowerShell bloquear o `Activate.ps1`, rode uma vez
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. Sem ativar, também dá para chamar
direto: `.\.venv\Scripts\python -m robo_fotos info`.

**Evite `uv run` neste projeto.** Como ele tem `pyproject.toml`, o `uv run` cria um
`uv.lock` e sincroniza o ambiente por conta própria, podendo trocar o PyTorch só-CPU
por outro.

### Versões que estavam funcionando (27/09/2026)

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

---

## 3. Levar as fotos e os dados (o Git não leva)

As fotos e os dados gerados estão no `.gitignore` de propósito: são grandes, são pessoais
e o repositório é **público**. O clone traz essas pastas vazias (com um `.gitkeep`).

| Pasta | O que tem | No PC novo | Em 27/09/2026 |
|---|---|---|---|
| `entrada/` | fotos originais do Google Fotos / Takeout (somente leitura) | **copiar** — ou baixar de novo do Google Fotos | 127 fotos + 3 vídeos, ~60 MB |
| `dados/correcoes.jsonl` | suas correções — **o mais valioso, não dá para refazer** | **copiar sempre** que existir | ainda não existe |
| `modelos/*.pt` | modelos treinados | **copiar** para não treinar de novo | vazia |
| `treino_base/` | fotos já na orientação certa, para o treino inicial | **copiar** se tiver | vazia |
| `saida/` | fotos corrigidas | copiar se ainda não subiu no Google Fotos | vazia |
| `lotes/` | lote_001, lote_002… | copiar, ou refazer com `dividir` (a mesma `entrada/` gera exatamente os mesmos lotes) | 3 lotes |
| `dados/sugestoes/`, `dados/debug_corte/`, `dados/demo_antes_depois/` | sugestões e imagens de conferência | não precisa: os comandos refazem | |
| `.venv/` | ambiente Python | **nunca copiar**: recriar com o passo 2 | |

### Empacotar no PC antigo

```powershell
# Windows
Compress-Archive -Path entrada, treino_base, dados, modelos, saida, lotes -DestinationPath "$HOME\Desktop\robo-fotos-dados.zip"
```

```bash
# Linux / Mac
tar -czf ~/robo-fotos-dados.tar.gz entrada treino_base dados modelos saida lotes
```

No PC novo, extraia dentro da pasta do projeto, por cima das pastas vazias. Esse arquivo
tem fotos pessoais: leve por pendrive ou pelo seu Drive, **nunca pelo Git**.

### Se você formatar sem copiar

Não é o fim do mundo hoje, porque as 127 fotos vieram do Google Fotos e continuam lá.
Baixe de novo e rode `dividir`. O que **não** dá para recuperar é `dados/correcoes.jsonl`
e os modelos em `modelos/` — mas em 27/09/2026 nenhum dos dois existe ainda, então neste
momento específico não há nada insubstituível fora do Git.

### Detalhe do Linux: maiúsculas importam

No Linux, `FOTO.JPG` e `foto.jpg` são arquivos diferentes; no Windows, não. O código já
trata isso (compara sempre a extensão em minúsculas) e o `.gitignore` lista as duas
formas. Só fique atento se copiar as fotos entre os dois sistemas.

---

## 4. Onde o projeto parou (27/09/2026, 16h25)

- [x] **Etapa 0 — Setup**: CLI, `config.yaml`, estrutura de pastas, 7 testes
- [x] **Etapa 1 — Lotes**: `dividir` (127 fotos → `lote_001` 50, `lote_002` 50,
      `lote_003` 27), data vinda do EXIF, 15 testes
- [x] **Etapa 2 — Corte**: `sugerir-corte`, métodos `bordas` (padrão) e `fundo`,
      36 testes
- [ ] **Etapa 3 — Orientação v1 ← é aqui que se recomeça.** `orientacao.py` e
      `treino.py` ainda são esqueletos, só com docstring.
- [ ] Etapas 4 a 8 — revisão no navegador, exportar, treino com correções, evolução

### Antes de começar a Etapa 3

Coloque algumas centenas de fotos **já na orientação certa** em `treino_base/`. Elas
viram o treino inicial sem você precisar rotular nada: o robô gira cada uma nas 4
posições e aprende a reconhecer qual é a certa.

Um detalhe que já foi medido e muda a expectativa: das 127 fotos em `entrada/`,
**125 já trazem a tag EXIF Orientation** (95 com valor 6, que significa girar 90°,
3 com valor 3 = 180°, 27 já certas). Como a regra do projeto é obedecer ao EXIF
quando ele existe, a rede neural só vai decidir sobre as 2 fotos restantes deste
acervo. Ela é importante para o resto da sua biblioteca, mas não espere uma
acurácia muito informativa medida só nestas 127.

### Como pedir ao Claude Code para continuar

Abra a pasta do projeto e peça:

> Continue da Etapa 3 conforme o CLAUDE.md.

O `CLAUDE.md` é lido automaticamente e tem o plano completo, as regras do projeto e a
seção **Decisões tomadas durante a implementação**, que registra o que foi descoberto
no caminho e que não está no plano original.

---

## 5. Checklist do PC novo

- [ ] `git --version` e `uv --version` respondem
- [ ] `python -m pytest` passa — 58 testes
- [ ] `python -m robo_fotos info` mostra as fotos em `entrada/` e os lotes
- [ ] `dados/correcoes.jsonl` e `modelos/` copiados (quando existirem)
- [ ] `git config user.name` e `user.email` configurados, para os commits saírem no seu nome
- [ ] Antes de trabalhar: `git pull`. Ao terminar: `git add`, `git commit`, `git push`
