"""Encontra o retângulo da foto dentro do fundo do scanner.

Visão computacional clássica, sem rede neural. A ideia em quatro passos:

1. Olhar as margens da imagem e concluir "o fundo tem esta cor".
2. Marcar todo pixel que é diferente dessa cor — isso é a foto.
3. Limpar a sujeira dessa marcação (poeira do scanner, grão do papel).
4. Pegar a maior mancha contínua e o retângulo que a envolve.

Por que sem rede neural? Porque o problema tem uma regra clara e visível ("o
fundo é uniforme, a foto não é"). Rede neural serve quando você não sabe
escrever a regra. Usar uma aqui seria mais lento, mais difícil de depurar e
precisaria de exemplos rotulados que você ainda não tem.

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

    limiar: int = 30
    ruido_kernel: int = 5
    margem_seguranca: int = 3
    faixa_fundo: float = 0.03
    area_minima_removida: float = 0.03
    area_minima_conteudo: float = 0.10
    largura_maxima_analise: int = 1200

    @classmethod
    def do_config(cls, cfg: Config) -> ParametrosCorte:
        bruto = cfg.corte
        padrao = cls()
        return cls(
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
    # faz o mesmo `limiar` e o mesmo `ruido_kernel` significarem a mesma coisa
    # em fotos de tamanhos diferentes.
    escala = 1.0
    analise = imagem
    if p.largura_maxima_analise > 0 and inteira.largura > p.largura_maxima_analise:
        escala = p.largura_maxima_analise / inteira.largura
        analise = cv2.resize(
            imagem,
            (p.largura_maxima_analise, max(1, int(round(inteira.altura * escala)))),
            interpolation=cv2.INTER_AREA,
        )

    fundo = cor_do_fundo(analise, p.faixa_fundo)
    mascara = mascara_do_conteudo(analise, fundo, p.limiar, p.ruido_kernel)
    achado = retangulo_do_conteudo(mascara)
    cor = tuple(int(c) for c in fundo[:3])

    if achado is None:
        return ResultadoCorte(inteira, False, 0.0, cor, "nada diferente do fundo")

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
