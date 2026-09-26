"""Encontra o retângulo da foto para cortar as bordas.

Visão computacional clássica, sem rede neural. Há dois métodos, porque há dois
tipos de material — e o certo depende de como a foto foi digitalizada.

**"bordas"** (padrão) — para foto de foto: você fotografou com o celular uma
foto impressa, apoiada num álbum ou numa mesa. Não existe fundo uniforme, mas
existe a beirada do papel, que é um risco atravessando a imagem inteira.
Medimos, coluna por coluna e linha por linha, o quanto ali existe borda, e
ficamos com os quatro picos mais internos.

**"fundo"** — para scanner de mesa: a foto está sobre um fundo liso branco ou
preto. Estimamos a cor desse fundo pelas margens, marcamos tudo que é diferente
dela e pegamos a maior mancha.

Por que sem rede neural? Porque o problema tem uma regra clara e visível. Rede
neural serve quando você não sabe escrever a regra. Usar uma aqui seria mais
lento, mais difícil de depurar e precisaria de exemplos rotulados que você
ainda não tem.

Todas as funções recebem e devolvem imagens como array NumPy em BGR, que é a
ordem de canais que o OpenCV usa.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import Config


@dataclass(frozen=True)
class Retangulo:
    """Um retângulo em pixels: canto superior esquerdo, largura e altura.

    É o mesmo formato [x, y, largura, altura] que vai para o correcoes.jsonl.
    """

    x: int
    y: int
    largura: int
    altura: int

    @property
    def direita(self) -> int:
        return self.x + self.largura

    @property
    def baixo(self) -> int:
        return self.y + self.altura

    @property
    def area(self) -> int:
        return max(0, self.largura) * max(0, self.altura)

    def como_lista(self) -> list[int]:
        return [self.x, self.y, self.largura, self.altura]

    @classmethod
    def da_lista(cls, valores) -> Retangulo:
        x, y, largura, altura = (int(v) for v in valores)
        return cls(x, y, largura, altura)

    @classmethod
    def imagem_inteira(cls, imagem: np.ndarray) -> Retangulo:
        altura, largura = imagem.shape[:2]
        return cls(0, 0, largura, altura)

    def encolher(self, pixels: int, limite: Retangulo) -> Retangulo:
        """Puxa as quatro bordas para dentro, sem sair do `limite`.

        É a margem de segurança: melhor perder alguns pixels da foto do que
        deixar uma listra do fundo do scanner aparecendo na borda.
        """
        if pixels <= 0:
            return self
        x = max(limite.x, self.x + pixels)
        y = max(limite.y, self.y + pixels)
        direita = min(limite.direita, self.direita - pixels)
        baixo = min(limite.baixo, self.baixo - pixels)
        # Se a margem comeu o retângulo todo, desistimos dela.
        if direita - x < 1 or baixo - y < 1:
            return self
        return Retangulo(x, y, direita - x, baixo - y)


def calcular_iou(a: Retangulo, b: Retangulo) -> float:
    """IoU = área da interseção / área da união. 1.0 = retângulos idênticos.

    É a métrica padrão para comparar caixas em visão computacional. Na Etapa 6
    ela mede o quanto o corte do robô se aproxima do corte que você fez à mão.
    """
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.direita, b.direita)
    y2 = min(a.baixo, b.baixo)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersecao = (x2 - x1) * (y2 - y1)
    uniao = a.area + b.area - intersecao
    return intersecao / uniao if uniao > 0 else 0.0


@dataclass(frozen=True)
class ParametrosCorte:
    """Os parâmetros ajustáveis do corte, todos vindos do config.yaml.

    Ficam juntos num objeto para a Etapa 6 poder testar muitas combinações
    sem tocar no código.
    """

    # "bordas" procura os riscos que atravessam a imagem (foto de foto, álbum).
    # "fundo" supõe um fundo uniforme atrás da foto (scanner de mesa).
    metodo: str = "bordas"

    # --- comuns aos dois métodos ---
    margem_seguranca: int = 3
    area_minima_removida: float = 0.03
    area_minima_conteudo: float = 0.10
    largura_maxima_analise: int = 1200
    faixa_fundo: float = 0.03

    # --- só do método "fundo" ---
    limiar: int = 30
    ruido_kernel: int = 5

    # --- só do método "bordas" ---
    limiar_gradiente: int = 25   # quanto o brilho precisa variar para contar como borda
    forca_borda: float = 0.55    # fração da altura/largura que a borda precisa atravessar
    zona_busca: float = 0.40     # procura cada borda só nos 40% externos daquele lado
    suavizacao_perfil: int = 9   # junta a energia de uma borda levemente torta

    @classmethod
    def do_config(cls, cfg: Config) -> ParametrosCorte:
        bruto = cfg.corte
        padrao = cls()
        return cls(
            metodo=str(bruto.get("metodo", padrao.metodo)),
            limiar_gradiente=int(
                bruto.get("limiar_gradiente", padrao.limiar_gradiente)
            ),
            forca_borda=float(bruto.get("forca_borda", padrao.forca_borda)),
            zona_busca=float(bruto.get("zona_busca", padrao.zona_busca)),
            suavizacao_perfil=int(
                bruto.get("suavizacao_perfil", padrao.suavizacao_perfil)
            ),
            limiar=int(bruto.get("limiar", padrao.limiar)),
            ruido_kernel=int(bruto.get("ruido_kernel", padrao.ruido_kernel)),
            margem_seguranca=int(
                bruto.get("margem_seguranca", padrao.margem_seguranca)
            ),
            faixa_fundo=float(bruto.get("faixa_fundo", padrao.faixa_fundo)),
            area_minima_removida=float(
                bruto.get("area_minima_removida", padrao.area_minima_removida)
            ),
            area_minima_conteudo=float(
                bruto.get("area_minima_conteudo", padrao.area_minima_conteudo)
            ),
            largura_maxima_analise=int(
                bruto.get("largura_maxima_analise", padrao.largura_maxima_analise)
            ),
        )


@dataclass(frozen=True)
class ResultadoCorte:
    """O que o robô achou. `cortar=False` significa "sem corte"."""

    retangulo: Retangulo
    cortar: bool
    fracao_removida: float
    cor_fundo: tuple[int, int, int]  # em BGR, como o OpenCV usa
    motivo: str  # texto curto para você entender a decisão


def cor_do_fundo(imagem: np.ndarray, faixa: float) -> np.ndarray:
    """Estima a cor do fundo pela mediana de quatro faixas nas margens.

    Por que a mediana e não a média? Porque a mediana ignora valores extremos.
    Se um canto da faixa pegar um pedaço da foto, a média seria contaminada;
    a mediana continua representando o fundo, que é a maioria dos pixels ali.
    """
    altura, largura = imagem.shape[:2]
    espessura_v = max(1, int(round(altura * faixa)))
    espessura_h = max(1, int(round(largura * faixa)))

    faixas = [
        imagem[:espessura_v, :],   # topo
        imagem[-espessura_v:, :],  # base
        imagem[:, :espessura_h],   # esquerda
        imagem[:, -espessura_h:],  # direita
    ]
    # Empilha todos os pixels das margens numa lista só e tira a mediana por canal.
    pixels = np.concatenate([f.reshape(-1, imagem.shape[2]) for f in faixas], axis=0)
    return np.median(pixels, axis=0)


def mascara_do_conteudo(
    imagem: np.ndarray, cor_fundo: np.ndarray, limiar: int, ruido_kernel: int
) -> np.ndarray:
    """Marca em branco os pixels que são diferentes do fundo.

    A diferença é medida canal por canal, e ficamos com a maior das três. Isso
    pega mudanças de cor que quase não mudam o brilho — um cinza e um bege
    podem ter luminosidade parecida, mas um é claramente "não fundo".
    """
    diferenca = np.abs(imagem.astype(np.int16) - cor_fundo.astype(np.int16))
    distancia = diferenca.max(axis=2).astype(np.uint8)
    mascara = (distancia > limiar).astype(np.uint8) * 255

    if ruido_kernel > 1:
        kernel = np.ones((ruido_kernel, ruido_kernel), np.uint8)
        # OPEN apaga manchinhas isoladas: poeira do scanner, grão do papel.
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
        # CLOSE tampa buracos dentro da foto: um céu claro parecido com o fundo.
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    return mascara


def retangulo_do_conteudo(mascara: np.ndarray) -> Retangulo | None:
    """Pega a maior mancha contínua da máscara e o retângulo que a envolve.

    "A maior" porque a foto é o maior objeto na imagem; sobras menores são
    ruído, uma etiqueta ou a sombra da tampa do scanner.
    """
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    maior = max(contornos, key=cv2.contourArea)
    x, y, largura, altura = cv2.boundingRect(maior)
    if largura < 1 or altura < 1:
        return None
    return Retangulo(int(x), int(y), int(largura), int(altura))


def perfil_de_bordas(
    imagem: np.ndarray, limiar_gradiente: int, suavizacao: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mede, para cada coluna e cada linha, o quanto ali existe uma borda.

    A borda do papel é um risco que **atravessa a imagem inteira**. Um detalhe
    dentro da foto (o contorno de uma pessoa, a quina de um móvel) também gera
    borda, mas só num pedacinho da altura. Então, em vez de procurar o contorno
    do papel — que quase nunca fecha direito —, perguntamos coluna por coluna:
    "que fração da altura desta coluna tem borda vertical?". A resposta fica
    perto de 1.0 na borda do papel e baixa em qualquer outro lugar.

    Devolve dois vetores com valores de 0 a 1: um por coluna, um por linha.
    """
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (5, 5), 0)

    # Sobel mede a variação de brilho: em x, acha bordas verticais; em y,
    # horizontais. O valor absoluto porque não interessa claro→escuro ou o contrário.
    gx = np.abs(cv2.Sobel(cinza, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(cinza, cv2.CV_32F, 0, 1, ksize=3))

    colunas = (gx > limiar_gradiente).mean(axis=0)
    linhas = (gy > limiar_gradiente).mean(axis=1)

    if suavizacao > 1:
        # Foto levemente torta espalha a borda por várias colunas vizinhas;
        # suavizar junta essa energia de volta num pico só.
        nucleo = np.ones(suavizacao) / suavizacao
        colunas = np.convolve(colunas, nucleo, "same")
        linhas = np.convolve(linhas, nucleo, "same")

    return colunas, linhas


def _borda_mais_interna(
    perfil: np.ndarray, do_inicio: bool, zona: float, forca: float
) -> int | None:
    """Acha a borda forte mais próxima do centro, dentro da zona de busca.

    Por que a mais interna e não a mais forte? Porque a borda mais forte pode
    ser outra coisa — numa foto de álbum, a lombada metálica marca mais que o
    papel. Pegar a mais interna erra incluindo menos, nunca decepando a foto.
    """
    n = len(perfil)
    limite = max(1, int(n * zona))
    faixa = perfil[:limite] if do_inicio else perfil[n - limite :]
    acima = np.flatnonzero(faixa >= forca)
    if acima.size == 0:
        return None
    return int(acima.max()) if do_inicio else int(n - limite + acima.min())


def retangulo_por_bordas(
    imagem: np.ndarray, parametros: ParametrosCorte
) -> tuple[Retangulo, int]:
    """Procura as quatro bordas do papel e monta o retângulo.

    Cada lado é independente: se só os lados esquerdo e direito forem
    encontrados, o de cima e o de baixo ficam no limite da imagem. Devolve
    também quantos lados foram realmente detectados — zero significa que a foto
    provavelmente preenche o quadro inteiro e não há nada para cortar.
    """
    p = parametros
    altura, largura = imagem.shape[:2]
    colunas, linhas = perfil_de_bordas(imagem, p.limiar_gradiente, p.suavizacao_perfil)

    esquerda = _borda_mais_interna(colunas, True, p.zona_busca, p.forca_borda)
    direita = _borda_mais_interna(colunas, False, p.zona_busca, p.forca_borda)
    topo = _borda_mais_interna(linhas, True, p.zona_busca, p.forca_borda)
    base = _borda_mais_interna(linhas, False, p.zona_busca, p.forca_borda)

    lados = sum(lado is not None for lado in (esquerda, direita, topo, base))

    x = esquerda if esquerda is not None else 0
    y = topo if topo is not None else 0
    x2 = direita if direita is not None else largura
    y2 = base if base is not None else altura

    if x2 - x < 1 or y2 - y < 1:
        return Retangulo(0, 0, largura, altura), 0
    return Retangulo(x, y, x2 - x, y2 - y), lados


def sugerir_corte(
    imagem: np.ndarray, parametros: ParametrosCorte | None = None
) -> ResultadoCorte:
    """Calcula o corte de uma imagem já girada.

    Sempre devolve um retângulo válido. Quando não confia no que achou, devolve
    a imagem inteira com `cortar=False` — errar para o lado de não cortar é
    muito melhor que decepar a foto.
    """
    p = parametros or ParametrosCorte()
    inteira = Retangulo.imagem_inteira(imagem)

    def sem_corte(motivo: str) -> ResultadoCorte:
        return ResultadoCorte(inteira, False, 0.0, (0, 0, 0), motivo)

    if inteira.area == 0:
        return sem_corte("imagem vazia")

    # A análise roda numa versão reduzida: fica mais rápido e, principalmente,
    # faz os limiares significarem a mesma coisa em fotos de tamanhos diferentes.
    escala = 1.0
    analise = imagem
    if p.largura_maxima_analise > 0 and inteira.largura > p.largura_maxima_analise:
        escala = p.largura_maxima_analise / inteira.largura
        analise = cv2.resize(
            imagem,
            (p.largura_maxima_analise, max(1, int(round(inteira.altura * escala)))),
            interpolation=cv2.INTER_AREA,
        )

    # A cor do fundo é sempre calculada: o método "fundo" depende dela, e no
    # método "bordas" ela ainda é uma informação útil no debug.
    fundo = cor_do_fundo(analise, p.faixa_fundo)
    cor = tuple(int(c) for c in fundo[:3])

    if p.metodo == "fundo":
        mascara = mascara_do_conteudo(analise, fundo, p.limiar, p.ruido_kernel)
        achado = retangulo_do_conteudo(mascara)
        falha = "nada diferente do fundo"
    else:
        achado, lados = retangulo_por_bordas(analise, p)
        falha = "nenhuma borda forte encontrada"
        if lados == 0:
            achado = None

    if achado is None:
        return ResultadoCorte(inteira, False, 0.0, cor, falha)

    if escala != 1.0:
        achado = Retangulo(
            int(round(achado.x / escala)),
            int(round(achado.y / escala)),
            int(round(achado.largura / escala)),
            int(round(achado.altura / escala)),
        )
        # O arredondamento pode estourar a borda por um pixel.
        achado = Retangulo(
            max(0, achado.x),
            max(0, achado.y),
            min(inteira.largura - max(0, achado.x), achado.largura),
            min(inteira.altura - max(0, achado.y), achado.altura),
        )

    # Retângulo muito pequeno quase sempre é erro: a foto é o assunto principal
    # da imagem, não um detalhe no canto.
    if achado.area < inteira.area * p.area_minima_conteudo:
        return ResultadoCorte(
            inteira,
            False,
            0.0,
            cor,
            f"conteúdo suspeito: só {achado.area / inteira.area:.1%} da imagem",
        )

    final = achado.encolher(p.margem_seguranca, inteira)
    fracao = 1.0 - (final.area / inteira.area)

    if fracao <= p.area_minima_removida:
        return ResultadoCorte(
            inteira, False, fracao, cor, f"borda pequena ({fracao:.1%})"
        )

    return ResultadoCorte(final, True, fracao, cor, f"corta {fracao:.1%} da área")


# --- entrada e saída de imagens ------------------------------------------------


def abrir_como_bgr(caminho: Path) -> np.ndarray:
    """Abre uma imagem com o Pillow e converte para o BGR do OpenCV.

    Por que não usar `cv2.imread`? Porque ele falha em caminhos com acentos ou
    caracteres fora do ASCII no Windows, e não abre HEIC. O Pillow resolve os
    dois problemas.
    """
    from PIL import Image

    from .lotes import _registrar_heic

    _registrar_heic()
    with Image.open(caminho) as img:
        rgb = img.convert("RGB")
        array = np.array(rgb)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def salvar_bgr(imagem: np.ndarray, caminho: Path, qualidade: int = 90) -> None:
    """Grava um array BGR em disco, também via Pillow (mesmo motivo)."""
    from PIL import Image

    caminho.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(caminho, quality=qualidade)


def desenhar_debug(imagem: np.ndarray, resultado: ResultadoCorte) -> np.ndarray:
    """Devolve uma cópia da imagem com o retângulo e a decisão desenhados.

    Verde = vai cortar aqui. Vermelho = decidi não cortar.
    """
    saida = imagem.copy()
    altura, largura = saida.shape[:2]
    cor = (0, 200, 0) if resultado.cortar else (0, 0, 220)
    espessura = max(2, round(largura / 400))

    r = resultado.retangulo
    cv2.rectangle(saida, (r.x, r.y), (r.direita - 1, r.baixo - 1), cor, espessura)

    texto = ("CORTAR " if resultado.cortar else "SEM CORTE ") + resultado.motivo
    escala = max(0.5, largura / 1400)
    linha = max(1, round(escala * 2))
    altura_faixa = int(38 * escala)
    # Faixa escura atrás do texto, para ele ser legível em foto clara.
    cv2.rectangle(saida, (0, 0), (largura, altura_faixa), (0, 0, 0), -1)
    cv2.putText(
        saida,
        texto,
        (int(8 * escala), int(27 * escala)),
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        (255, 255, 255),
        linha,
        cv2.LINE_AA,
    )
    return saida
