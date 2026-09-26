# Projeto: Robô de orientação e corte de fotos (com treino humano)

## Contexto

Tenho muitas fotos no Google Fotos com orientação errada (de lado, de cabeça para baixo) e com bordas grandes (fundo de scanner, faixas pretas/brancas). Quero um app local que corrija isso em lotes de 50 fotos e que **aprenda com as minhas correções**, errando menos a cada lote.

Este também é um projeto para eu **aprender IA na prática**. Explique as decisões de forma simples enquanto constrói.

## O que o app deve fazer

1. Ler as fotos baixadas do Google Fotos (álbum em .zip ou Google Takeout) da pasta `entrada/`.
2. Dividir em lotes de 50 fotos.
3. Para cada foto, sugerir a **rotação correta** e o **corte das bordas grandes**, com um nível de confiança.
4. Mostrar uma **página de revisão** no navegador onde eu aprovo ou corrijo rápido (de preferência só pelo teclado).
5. Salvar as fotos corrigidas em `saida/`, prontas para eu subir de volta no Google Fotos.
6. Guardar minhas correções e usá-las para **re-treinar o modelo**, mostrando a evolução do acerto por lote.

## Regras importantes

- **Nunca alterar nem apagar nada em `entrada/`.** Tudo que for gerado vai para `saida/`, `dados/` ou `modelos/`.
- **Preservar metadados EXIF** (principalmente data/hora e GPS), porque o Google Fotos usa a data para organizar a linha do tempo.
- Depois de girar os pixels, **definir a tag EXIF Orientation = 1**, senão a foto aparece girada duas vezes.
- Arquivos do Takeout vêm com um `.json` ao lado (ex.: `IMG_0123.jpg.json` ou `IMG_0123.jpg.supplemental-metadata.json`). Ignorar esses JSONs no processamento. Porém, se a foto não tiver data no EXIF, usar o `photoTakenTime` do JSON para gravar a data na foto de saída.
- Formatos: JPG, PNG e HEIC (via `pillow-heif`). Vídeos são ignorados.
- Ao salvar JPEG, usar qualidade alta (95). Se for possível fazer só rotação sem perda de qualidade de forma simples, melhor ainda.
- **Tudo local**: nenhuma foto é enviada para serviços externos.
- Precisa funcionar em **Windows e Mac** (usar `pathlib`, nada de caminhos fixos).
- Rodar em CPU. Se houver GPU (CUDA ou Apple MPS), usar automaticamente.

## Stack sugerida

- Python 3.11+ com `venv`
- Pillow + pillow-heif (abrir/salvar imagens, EXIF)
- OpenCV (`opencv-python-headless`) para o corte de bordas
- PyTorch + torchvision para o modelo de orientação
- FastAPI + uvicorn + HTML/JS simples (sem framework pesado) para a página de revisão
- pytest para testes
- Configurações em `config.yaml`

Se achar que outra escolha é melhor, **me pergunte antes de trocar**.

## Estrutura de pastas

```
robo-fotos/
├── CLAUDE.md
├── README.md
├── config.yaml
├── requirements.txt
├── entrada/                 # fotos baixadas do Google Fotos (somente leitura)
├── lotes/lote_001/ ...      # 50 fotos por lote (cópias ou links)
├── saida/lote_001/ ...      # fotos corrigidas, prontas para subir
├── treino_base/             # fotos JÁ na orientação certa, para o treino inicial
├── dados/
│   ├── sugestoes/lote_001.json
│   └── correcoes.jsonl      # todas as minhas correções (o "treino humano")
├── modelos/                 # orientacao_v001.pt, orientacao_v002.pt ...
├── src/robo_fotos/
│   ├── __main__.py          # CLI: python -m robo_fotos <comando>
│   ├── lotes.py
│   ├── corte.py
│   ├── orientacao.py
│   ├── treino.py
│   ├── exportar.py
│   └── app/                 # servidor + página de revisão
└── tests/
```

## Como funciona a parte de IA

### Orientação (classificador de 4 classes: 0°, 90°, 180°, 270°)

1. **Primeiro olhar o EXIF.** Se a tag Orientation já indicar a rotação, aplicar e considerar confiança 1.0.
2. Caso contrário, usar um **modelo pequeno pré-treinado** (ex.: MobileNetV3 ou EfficientNet-B0 do torchvision) ajustado (fine-tuning) para prever a rotação.
3. **Treino inicial auto-supervisionado:** pegar as fotos de `treino_base/` (que já estão certas) e gerar as 4 rotações de cada uma. Os rótulos saem de graça, sem eu marcar nada. Algumas centenas de fotos já bastam para começar. Se as fotos a corrigir forem scans antigos, o ideal é o `treino_base/` ter fotos parecidas.
4. Cada previsão tem uma **confiança** (softmax). Fotos com confiança baixa aparecem primeiro na revisão.
5. Depois de cada lote revisado, **minhas correções entram no treino**, com peso maior, porque são justamente os casos difíceis.

### Corte de bordas (visão computacional clássica, sem rede neural no começo)

1. O corte é sempre calculado **na imagem já girada**.
2. Estimar a cor do fundo a partir de faixas nas margens da imagem (mediana).
3. Criar uma máscara dos pixels "diferentes do fundo" (limiar) e limpar o ruído com operações morfológicas.
4. Pegar o maior contorno e obter o retângulo da foto.
5. Aplicar uma margem de segurança configurável (ex.: alguns pixels para dentro).
6. Só sugerir corte se a borda for "grande" (ex.: remove mais de 3% da área). Caso contrário, sugerir "sem corte".
7. Parâmetros (limiar, margem, % mínimo) ficam em `config.yaml`.
8. **"Aprender" o corte:** com as correções acumuladas, testar combinações de parâmetros e escolher a que mais se aproxima dos meus cortes (maior IoU médio). Uma rede que prevê o retângulo fica para o futuro.

## Página de revisão

- Comando: `python -m robo_fotos revisar lote_001`, que abre o navegador em `localhost`.
- Grade com as 50 fotos já giradas e com o retângulo de corte desenhado por cima. Fotos de baixa confiança aparecem destacadas e primeiro.
- Para cada foto: girar ↺ / ↻ / 180°, marcar "sem corte", arrastar os cantos do retângulo e aprovar ✓.
- Atalhos de teclado: setas para navegar, `Q`/`E` para girar, `C` para resetar o corte, `Espaço` para aprovar.
- Botão "Aprovar o resto do lote" (aprova tudo que eu não mexi).
- Ao finalizar: grava as correções em `dados/correcoes.jsonl` e mostra o placar do lote, ex.: "Orientação: 46/50 certas · Corte: 41/50 sem ajuste".

### Formato de cada correção (uma linha por foto em `correcoes.jsonl`)

```json
{"arquivo": "IMG_0123.jpg", "lote": "lote_001", "modelo": "orientacao_v001",
 "rotacao_sugerida": 90, "confianca": 0.62, "rotacao_final": 0,
 "corte_sugerido": [40, 32, 1800, 1200], "corte_final": [52, 30, 1790, 1205],
 "revisado_em": "2026-09-25T14:03:00"}
```

## Ciclo de treino

- Comando: `python -m robo_fotos treinar`.
- Separar ~20% das correções como **validação fixa** (nunca usada para treinar).
- Treinar uma nova versão (`orientacao_v002.pt`) e compará-la com a anterior na validação. **Só promover se não piorar.** Manter as versões antigas.
- Relatório no terminal + página `/evolucao` com gráfico do acerto por lote (orientação e corte).

## Etapas (seguir em ordem e parar ao final de cada uma para eu testar)

- **Etapa 0 — Setup:** venv, `requirements.txt`, estrutura de pastas, `README.md` com os comandos.
  *Aceite:* `python -m robo_fotos --help` funciona.
- **Etapa 1 — Lotes:** comando `dividir`, que pega `entrada/` e cria `lotes/lote_001`, `lote_002`... com 50 fotos cada, ordenadas por data, ignorando vídeos e `.json`.
  *Aceite:* 120 fotos geram 3 lotes (50, 50, 20).
- **Etapa 2 — Corte:** `corte.py` + comando `sugerir-corte lote_001`, que gera imagens de debug com o retângulo desenhado.
  *Aceite:* testes com imagens sintéticas (uma foto colada sobre fundo branco e sobre fundo preto) acertam o retângulo com IoU > 0.95.
- **Etapa 3 — Orientação v1:** leitura de EXIF + treino auto-supervisionado com `treino_base/` + comando `sugerir lote_001`, que gera `dados/sugestoes/lote_001.json` (rotação, corte e confiança).
  *Aceite:* relatório de acurácia na validação.
- **Etapa 4 — Página de revisão:** conforme a seção acima.
  *Aceite:* consigo revisar 50 fotos só com o teclado em poucos minutos.
- **Etapa 5 — Exportar:** aplicar rotação e corte, preservar o EXIF e gerar `saida/lote_xxx/`.
  *Aceite:* teste automático confirma que a data da foto foi preservada e que Orientation = 1.
- **Etapa 6 — Treino com correções:** validação fixa, versionamento de modelos e ajuste dos parâmetros de corte.
  *Aceite:* `treinar` gera a v002 e mostra a comparação com a v001.
- **Etapa 7 — Evolução:** página com o gráfico de acerto por lote.
- **Etapa 8 — Futuro (opcional):** auto-aprovar fotos de alta confiança, endireitar fotos levemente tortas (deskew), separar várias fotos num mesmo scan.

## Como quero que você trabalhe

- Explicações e comentários em **português**. Nomes no código podem ser em português ou inglês, desde que consistentes.
- **Uma etapa por vez.** No fim de cada uma, me diga: o que foi feito, como eu testo e qual conceito de IA/visão computacional apareceu ali.
- Pergunte antes de decisões grandes (trocar a stack, adicionar dependência pesada, mudar a estrutura).
- Escreva testes, principalmente para o corte e para a preservação do EXIF.
- Código simples e legível é melhor que código esperto.
- Ao concluir uma etapa, **atualize a seção "Progresso" abaixo**.

## Progresso

- [x] Etapa 0 — Setup
- [x] Etapa 1 — Lotes
- [x] Etapa 2 — Corte
- [ ] Etapa 3 — Orientação v1
- [ ] Etapa 4 — Página de revisão
- [ ] Etapa 5 — Exportar
- [ ] Etapa 6 — Treino com correções
- [ ] Etapa 7 — Evolução
- [ ] Etapa 8 — Futuro

## Depois de processar (manual, fora do código)

1. Subir as fotos de `saida/` para um álbum novo no Google Fotos (pelo site ou pelo backup do Google Drive para computador).
2. Conferir se está tudo certo.
3. Apagar as fotos originais tortas no Google Fotos para não ficarem duplicadas.
