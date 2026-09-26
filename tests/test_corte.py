"""Testes da Etapa 2: corte de bordas.

A estratégia é montar imagens sintéticas onde nós sabemos a resposta exata:
colamos um retângulo de "foto" numa posição conhecida sobre um fundo uniforme e
conferimos se o robô encontra aquele mesmo retângulo.
"""

from __future__ import annotations

import numpy as np
import pytest

from robo_fotos.corte import (
    ParametrosCorte,
    Retangulo,
    calcular_iou,
    cor_do_fundo,
    desenhar_debug,
    mascara_do_conteudo,
    retangulo_do_conteudo,
    sugerir_corte,
)

# Nos testes desligamos a redução de escala e a margem de segurança, para medir
# a precisão do algoritmo em si, sem o arredondamento de um nem o recuo do outro.
EXATO = ParametrosCorte(margem_seguranca=0, largura_maxima_analise=0)


def montar_scan(
    tamanho: tuple[int, int],
    retangulo: Retangulo,
    cor_fundo: tuple[int, int, int],
    semente: int = 0,
) -> np.ndarray:
    """Simula um scan: fundo uniforme com uma "foto" colada em cima.

    A foto é ruído colorido, não uma cor sólida, porque uma foto de verdade tem
    textura — e é justamente a textura que a diferencia do fundo.
    """
    altura, largura = tamanho
    imagem = np.full((altura, largura, 3), cor_fundo, dtype=np.uint8)

    rng = np.random.default_rng(semente)
    conteudo = rng.integers(60, 200, size=(retangulo.altura, retangulo.largura, 3))
    imagem[
        retangulo.y : retangulo.baixo, retangulo.x : retangulo.direita
    ] = conteudo.astype(np.uint8)
    return imagem


# --- critério de aceite da etapa ------------------------------------------------


@pytest.mark.parametrize(
    "nome_do_fundo, cor_fundo",
    [("branco", (255, 255, 255)), ("preto", (0, 0, 0))],
)
def test_aceite_acha_o_retangulo_com_iou_acima_de_095(nome_do_fundo, cor_fundo):
    """Aceite da Etapa 2: IoU > 0.95 sobre fundo branco e sobre fundo preto."""
    esperado = Retangulo(120, 80, 600, 420)
    imagem = montar_scan((600, 800), esperado, cor_fundo)

    resultado = sugerir_corte(imagem, EXATO)

    assert resultado.cortar, f"deveria cortar no fundo {nome_do_fundo}"
    iou = calcular_iou(resultado.retangulo, esperado)
    assert iou > 0.95, f"fundo {nome_do_fundo}: IoU {iou:.4f}"


@pytest.mark.parametrize("cor_fundo", [(255, 255, 255), (0, 0, 0), (128, 128, 128)])
def test_acha_o_retangulo_mesmo_descentralizado(cor_fundo):
    """A foto raramente está no meio do scanner."""
    esperado = Retangulo(30, 220, 500, 330)
    imagem = montar_scan((600, 800), esperado, cor_fundo)

    resultado = sugerir_corte(imagem, EXATO)

    assert calcular_iou(resultado.retangulo, esperado) > 0.95


def test_funciona_com_a_reducao_de_escala_ligada():
    """Com a análise reduzida, a precisão cai um pouco — mas não abaixo do aceite."""
    esperado = Retangulo(300, 200, 1800, 1300)
    imagem = montar_scan((1700, 2400), esperado, (250, 250, 250))

    resultado = sugerir_corte(
        imagem, ParametrosCorte(margem_seguranca=0, largura_maxima_analise=1200)
    )

    assert resultado.cortar
    assert calcular_iou(resultado.retangulo, esperado) > 0.95


def test_aguenta_poeira_do_scanner():
    """Pontinhos isolados de sujeira não devem virar o "conteúdo" da imagem."""
    esperado = Retangulo(150, 120, 500, 350)
    imagem = montar_scan((600, 800), esperado, (255, 255, 255))
    # Cinco grãos de poeira escura espalhados pelo fundo.
    for x, y in [(10, 10), (780, 20), (20, 580), (770, 570), (400, 590)]:
        imagem[y : y + 3, x : x + 3] = (30, 30, 30)

    resultado = sugerir_corte(imagem, EXATO)

    assert calcular_iou(resultado.retangulo, esperado) > 0.95


# --- a decisão de cortar ou não -------------------------------------------------


def test_borda_pequena_nao_gera_corte():
    """Regra do projeto: só cortar se a borda for grande (mais de 3% da área)."""
    # Moldura de 2 pixels em 800x600: remove menos de 1% da área.
    esperado = Retangulo(2, 2, 796, 596)
    imagem = montar_scan((600, 800), esperado, (255, 255, 255))

    resultado = sugerir_corte(imagem, EXATO)

    assert not resultado.cortar
    assert resultado.retangulo == Retangulo(0, 0, 800, 600)
    assert "borda pequena" in resultado.motivo


def test_imagem_toda_uniforme_nao_gera_corte():
    """Sem nada diferente do fundo, não há o que cortar."""
    imagem = np.full((300, 400, 3), 200, dtype=np.uint8)

    resultado = sugerir_corte(imagem, EXATO)

    assert not resultado.cortar
    assert resultado.retangulo == Retangulo(0, 0, 400, 300)


def test_conteudo_minusculo_e_tratado_como_erro():
    """Uma manchinha no canto não é a foto: melhor não cortar nada."""
    imagem = np.full((600, 800, 3), 255, dtype=np.uint8)
    imagem[500:560, 700:760] = 40  # 0.75% da área

    resultado = sugerir_corte(imagem, EXATO)

    assert not resultado.cortar
    assert "suspeito" in resultado.motivo


def test_sem_corte_devolve_sempre_a_imagem_inteira():
    """Assim quem consome o resultado não precisa tratar caso especial."""
    imagem = np.full((120, 250, 3), 90, dtype=np.uint8)

    resultado = sugerir_corte(imagem, EXATO)

    assert resultado.retangulo == Retangulo(0, 0, 250, 120)
    assert resultado.fracao_removida == 0.0


# --- a margem de segurança ------------------------------------------------------


def test_margem_de_seguranca_encolhe_o_retangulo():
    esperado = Retangulo(100, 100, 400, 300)
    imagem = montar_scan((600, 800), esperado, (255, 255, 255))

    com_margem = sugerir_corte(
        imagem, ParametrosCorte(margem_seguranca=5, largura_maxima_analise=0)
    )

    r = com_margem.retangulo
    assert r.x >= esperado.x + 4
    assert r.direita <= esperado.direita - 4
    assert r.altura < esperado.altura


def test_margem_nao_pode_virar_o_retangulo_do_avesso():
    """Margem maior que o próprio retângulo deve ser ignorada, não inverter nada."""
    pequeno = Retangulo(10, 10, 6, 6)
    limite = Retangulo(0, 0, 100, 100)

    resultado = pequeno.encolher(50, limite)

    assert resultado == pequeno


def test_margem_nao_sai_dos_limites_da_imagem():
    r = Retangulo(0, 0, 100, 100)
    encolhido = r.encolher(3, Retangulo(0, 0, 100, 100))
    assert encolhido.x == 3 and encolhido.y == 3
    assert encolhido.direita == 97 and encolhido.baixo == 97


# --- as peças do algoritmo ------------------------------------------------------


def test_cor_do_fundo_usa_a_mediana_e_ignora_a_foto():
    """A mediana precisa resistir à foto invadindo a faixa da margem."""
    esperado = Retangulo(100, 50, 600, 500)  # encosta perto das bordas
    imagem = montar_scan((600, 800), esperado, (240, 245, 250))

    fundo = cor_do_fundo(imagem, 0.03)

    assert np.allclose(fundo, (240, 245, 250), atol=2)


def test_mascara_marca_o_conteudo_e_nao_o_fundo():
    retangulo = Retangulo(20, 20, 60, 40)
    imagem = montar_scan((100, 120), retangulo, (255, 255, 255))

    mascara = mascara_do_conteudo(imagem, np.array([255, 255, 255]), 30, 3)

    assert mascara[40, 50] == 255  # dentro da foto
    assert mascara[5, 5] == 0  # fundo


def test_retangulo_do_conteudo_pega_a_maior_mancha():
    mascara = np.zeros((200, 200), dtype=np.uint8)
    mascara[10:30, 10:30] = 255  # mancha pequena
    mascara[60:180, 50:190] = 255  # mancha grande

    achado = retangulo_do_conteudo(mascara)

    assert achado == Retangulo(50, 60, 140, 120)


def test_mascara_vazia_devolve_nada():
    assert retangulo_do_conteudo(np.zeros((50, 50), dtype=np.uint8)) is None


# --- o IoU ----------------------------------------------------------------------


def test_iou_identico_e_1():
    r = Retangulo(10, 20, 100, 50)
    assert calcular_iou(r, r) == pytest.approx(1.0)


def test_iou_sem_sobreposicao_e_0():
    a = Retangulo(0, 0, 10, 10)
    b = Retangulo(50, 50, 10, 10)
    assert calcular_iou(a, b) == 0.0


def test_iou_meia_sobreposicao():
    # Dois quadrados 10x10 sobrepostos pela metade: interseção 50, união 150.
    a = Retangulo(0, 0, 10, 10)
    b = Retangulo(5, 0, 10, 10)
    assert calcular_iou(a, b) == pytest.approx(50 / 150)


def test_iou_de_retangulo_contido():
    fora = Retangulo(0, 0, 10, 10)
    dentro = Retangulo(2, 2, 5, 5)
    assert calcular_iou(fora, dentro) == pytest.approx(25 / 100)


# --- utilidades -----------------------------------------------------------------


def test_retangulo_vai_e_volta_da_lista():
    r = Retangulo(40, 32, 1800, 1200)
    assert r.como_lista() == [40, 32, 1800, 1200]
    assert Retangulo.da_lista([40, 32, 1800, 1200]) == r


def test_debug_nao_altera_a_imagem_original():
    esperado = Retangulo(50, 50, 200, 150)
    imagem = montar_scan((300, 400), esperado, (255, 255, 255))
    copia = imagem.copy()

    saida = desenhar_debug(imagem, sugerir_corte(imagem, EXATO))

    assert np.array_equal(imagem, copia), "a original não pode ser modificada"
    assert saida.shape == imagem.shape
    assert not np.array_equal(saida, imagem), "o debug precisa desenhar algo"


def test_parametros_saem_do_config():
    from robo_fotos.config import carregar_config

    p = ParametrosCorte.do_config(carregar_config())

    assert p.limiar == 30
    assert p.margem_seguranca == 3
    assert p.area_minima_removida == pytest.approx(0.03)
