# Robô de fotos

App **local** que corrige a orientação (fotos de lado ou de cabeça para baixo) e
corta as bordas grandes (fundo de scanner, faixas pretas/brancas) das fotos
baixadas do Google Fotos — e que **aprende com as suas correções**, errando menos
a cada lote.

Nenhuma foto sai do seu computador.

> **Estado em 27/09/2026:** Etapas 0, 1 e 2 prontas (setup, divisão em lotes e corte
> de bordas), 58 testes passando. A próxima é a **Etapa 3 — Orientação**, e os
> módulos `orientacao.py` e `treino.py` ainda são esqueletos. Os comandos `sugerir`,
> `revisar`, `exportar` e `treinar` avisam em qual etapa chegam.
>
> Montando numa máquina nova (Linux, Mac ou Windows)? Vá direto para o
> [`NOVO-PC.md`](NOVO-PC.md).

## Instalação

Só é preciso fazer isso uma vez. **Montando num computador novo?** Siga o
[`NOVO-PC.md`](NOVO-PC.md): o que instalar, versões testadas, quais pastas copiar à mão
(as fotos não vão para o Git) e onde o projeto parou.

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### Mac / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Depois de ativar a venv, confira que deu tudo certo:

```bash
python -m robo_fotos --help
python -m robo_fotos info
```

> Se você usa [`uv`](https://docs.astral.sh/uv/), o atalho é
> `uv venv && uv pip install -e .`.

## Como usar

1. **Coloque as fotos em `entrada/`** (o conteúdo do .zip do álbum ou do Google
   Takeout, inclusive subpastas). Essa pasta é **somente leitura** para o app:
   nada nela é alterado ou apagado.

2. **Coloque em `treino_base/`** algumas centenas de fotos que **já estão na
   orientação certa**. Elas ensinam o modelo, sem você precisar marcar nada. Se
   as fotos a corrigir são scans antigos, o ideal é que o `treino_base/` também
   tenha scans antigos.

3. Rode os comandos, na ordem:

```bash
python -m robo_fotos dividir              # cria lotes/lote_001, lote_002... (50 fotos cada)
python -m robo_fotos treinar-base         # treino inicial da orientação (uma vez)
python -m robo_fotos sugerir lote_001     # sugere rotação + corte, com confiança
python -m robo_fotos revisar lote_001     # abre o navegador para você aprovar/corrigir
python -m robo_fotos exportar lote_001    # grava as fotos corrigidas em saida/lote_001/
python -m robo_fotos treinar              # re-treina o modelo com as suas correções
```

Repita `sugerir → revisar → exportar` para cada lote. De vez em quando rode
`treinar` para o robô ficar melhor.

## Comandos

| Comando | O que faz |
|---|---|
| `info` | mostra o estado do projeto: pastas, lotes criados, correções, modelos |
| `dividir` | divide `entrada/` em lotes de 50 fotos, ordenadas por data |
| `treinar-base` | treino inicial da orientação a partir de `treino_base/` |
| `sugerir-corte <lote>` | só o corte, com imagens de debug para conferir os parâmetros |
| `sugerir <lote>` | gera `dados/sugestoes/<lote>.json` com rotação, corte e confiança |
| `revisar <lote>` | página de revisão no navegador (atalhos de teclado) |
| `exportar <lote>` | aplica rotação e corte, preserva o EXIF, grava em `saida/<lote>/` |
| `treinar` | treina uma nova versão do modelo e compara com a anterior |

Use `--help` em qualquer comando, ex.: `python -m robo_fotos dividir --help`.

## Atalhos da página de revisão

| Tecla | Ação |
|---|---|
| `←` `→` | foto anterior / próxima |
| `Q` / `E` | girar 90° anti-horário / horário |
| `R` | girar 180° |
| `C` | resetar o corte |
| `X` | marcar "sem corte" |
| `Espaço` | aprovar e ir para a próxima |

## Pastas

```
entrada/       suas fotos originais — o app NUNCA escreve aqui
treino_base/   fotos já na orientação certa, para o treino inicial
lotes/         lote_001, lote_002... (50 fotos por lote)
saida/         fotos corrigidas, prontas para subir no Google Fotos
dados/         sugestões do robô e correcoes.jsonl (o seu "treino humano")
modelos/       orientacao_v001.pt, orientacao_v002.pt...
config.yaml    todos os parâmetros ajustáveis
```

## Configuração

Tudo que vale ajustar está em [`config.yaml`](config.yaml), com um comentário
explicando cada parâmetro. Os mais úteis:

- `lotes.tamanho` — fotos por lote (padrão 50)
- `corte.metodo` — **`bordas`** (padrão) para foto de foto, quando você
  fotografou com o celular uma foto impressa no álbum; **`fundo`** para scanner
  de mesa, com a foto sobre um fundo liso branco ou preto
- `corte.forca_borda` — no método `bordas`, quanto da altura (ou largura) a
  beirada do papel precisa atravessar. Maior corta menos.
- `corte.area_minima_removida` — abaixo disso o robô sugere "sem corte"
- `orientacao.confianca_baixa` — abaixo disso a foto aparece destacada na revisão

### Qual método de corte usar

O método `bordas` procura a beirada do papel — o risco que atravessa a imagem
inteira. Ele funciona bem quando você enquadrou a foto impressa ao fotografar,
que é o caso normal. Quando o papel preenche o quadro todo, ele corretamente não
sugere corte nenhum.

A limitação conhecida: se a foto impressa ficou pequena no meio de um entorno
grande, ele pode não achar todos os quatro lados e sugerir um corte que deixa um
pedaço de fundo. Ele erra sobrando fundo, nunca decepando a foto — e você ajusta
na revisão.

## Testes

```bash
python -m pytest
```

## Depois de exportar (manual)

1. Suba as fotos de `saida/` para um **álbum novo** no Google Fotos.
2. Confira se está tudo certo.
3. Apague as originais tortas no Google Fotos, para não ficarem duplicadas.

## Requisitos

Python 3.11 ou mais novo. Roda em CPU; se houver GPU (CUDA no Windows ou MPS no
Mac), é usada automaticamente. Formatos aceitos: JPG, PNG e HEIC. Vídeos e os
`.json` do Takeout são ignorados.

## Estado do projeto

Veja a seção **Progresso** em [CLAUDE.md](CLAUDE.md).
