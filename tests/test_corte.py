"""Testes da Etapa 2: corte de bordas.

A estratégia é montar imagens sintéticas onde nós sabemos a resposta exata:
colamos um retângulo de "foto" numa posição conhecida e conferimos se o robô
encontra aquele mesmo retângulo.

Há dois métodos e dois cenários, e cada método é testado no cenário dele:

- "fundo"  -> scan: foto sobre fundo liso branco ou preto. Fixture `montar_scan`.
- "bordas" -> foto de foto: foto impressa apoiada num álbum, capturada pelo
              celular. Fixture `montar_foto_impressa`, que tem conteúdo liso e
              desfoque óptico, como qualquer foto de celular.

Testar o método de bordas com o fixture de ruído puro não faria sentido: ruído
pixel a pixel gera borda em todo lugar, e nenhuma foto de verdade é assim.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from robo_fotos.corte import (
    ParametrosCorte,
    Retangulo,
    calcular_iou,
    cor_do_fundo,
    desenhar_debug,
    mascara_do_conteudo,
    perfil_de_bordas,
    retangulo_do_conteudo,
    retangulo_por_bordas,
    sugerir_corte,
)

# Nos testes desligamos a redução de escala e a margem de segurança, para medir
# a precisão do algoritmo em si, sem o arredondamento de um nem o recuo do outro.
EXATO_FUNDO = ParametrosCorte(
    metodo="fundo", margem_seguranca=0, largura_maxima_analise=0
)
EXATO_BORDAS = ParametrosCorte(
    metodo="bordas", margem_seguranca=0, largura_maxima_analise=0
)


def montar_scan(
    tamanho: tuple[int, int],
    retangulo: Retangulo,
    cor_fundo: tuple[int, int, int],
    semente: int = 0,
) -> np.ndarray:
    """Simula um scan: fundo uniforme com uma "foto" colada em cima.

    A foto é ruído colorido porque, para o método "fundo", o que importa é só
    "isto é diferente da cor do fundo" — e ruído é o caso mais exigente.
    """
    altura, largura = tamanho
    imagem = np.full((altura, largura, 3), cor_fundo, dtype=np.uint8)

    rng = np.random.default_rng(semente)
    conteudo = rng.integers(60, 200, size=(retangulo.altura, retangulo.largura, 3))
    imagem[
        retangulo.y : retangulo.baixo, retangulo.x : retangulo.direita
    ] = conteudo.astype(np.uint8)
    return imagem


def montar_foto_impressa(
    tamanho: tuple[int, int],
    retangulo: Retangulo,
    cor_entorno: tuple[int, int, int],
    semente: int = 0,
    desfoque: int = 3,
) -> np.ndarray:
    """Simula uma foto de foto: papel impresso apoiado num álbum ou numa mesa.

    Três coisas importam para ser fiel ao real:
    - o conteúdo da foto é **liso** (degradê e formas grandes), não ruído;
    - o entorno tem textura leve, como tecido ou papel;
    - a imagem toda leva um desfoque, porque lente de celular sempre tem um.
    """
    altura, largura = tamanho
    rng = np.random.default_rng(semente)

    imagem = np.full((altura, largura, 3), cor_entorno, dtype=np.uint8)
    imagem = np.clip(
        imagem.astype(np.int16) + rng.integers(-6, 7, (altura, largura, 3)), 0, 255
    ).astype(np.uint8)

    ch, cw = retangulo.altura, retangulo.largura
    degrade = np.clip(
        np.linspace(40, 190, ch)[:, None] + np.linspace(-20, 20, cw)[None, :], 0, 255
    )
    foto = (
        np.dstack([degrade * 0.9, degrade, degrade * 1.1]).clip(0, 255).astype(np.uint8)
    )
    cv2.ellipse(foto, (cw // 3, int(ch * 0.6)), (cw // 6, ch // 4), 0, 0, 360, (60, 50, 45), -1)
    cv2.rectangle(foto, (int(cw * 0.6), int(ch * 0.5)), (int(cw * 0.85), ch - 5), (200, 180, 150), -1)

    imagem[retangulo.y : retangulo.baixo, retangulo.x : retangulo.direita] = foto
    return cv2.GaussianBlur(imagem, (desfoque * 2 + 1,) * 2, 0)


def corta_dentro_de(sugerido: Retangulo, foto: Retangulo) -> int:
    """Quantos pixels o corte sugerido comeu de dentro da foto (0 = nenhum)."""
    return max(
        0,
        sugerido.x - foto.x,
        sugerido.y - foto.y,
        foto.direita - sugerido.direita,
        foto.baixo - sugerido.baixo,
    )


# --- critério de aceite: método "fundo" (o cenário de scanner do CLAUDE.md) ----


@pytest.mark.parametrize(
    "nome_do_fundo, cor_fundo",
    [("branco", (255, 255, 255)), ("preto", (0, 0, 0))],
)
def test_aceite_fundo_acha_o_retangulo_com_iou_acima_de_095(nome_do_fundo, cor_fundo):
    """Aceite da Etapa 2: IoU > 0.95 sobre fundo branco e sobre fundo preto."""
    esperado = Retangulo(120, 80, 600, 420)
    imagem = montar_scan((600, 800), esperado, cor_fundo)

    resultado = sugerir_corte(imagem, EXATO_FUNDO)

    assert resultado.cortar, f"deveria cortar no fundo {nome_do_fundo}"
    iou = calcular_iou(resultado.retangulo, esperado)
    assert iou > 0.95, f"fundo {nome_do_fundo}: IoU {iou:.4f}"


@pytest.mark.parametrize("cor_fundo", [(255, 255, 255), (0, 0, 0), (128, 128, 128)])
def test_fundo_acha_o_retangulo_mesmo_descentralizado(cor_fundo):
    """A foto raramente está no meio do scanner."""
    esperado = Retangulo(30, 220, 500, 330)
    imagem = montar_scan((600, 800), esperado, cor_fundo)

    resultado = sugerir_corte(imagem, EXATO_FUNDO)

    assert calcular_iou(resultado.retangulo, esperado) > 0.95


def test_fundo_funciona_com_a_reducao_de_escala_ligada():
    """Com a análise reduzida, a precisão cai um pouco — mas não abaixo do aceite."""
    esperado = Retangulo(300, 200, 1800, 1300)
    imagem = montar_scan((1700, 2400), esperado, (250, 250, 250))

    resultado = sugerir_corte(
        imagem,
        ParametrosCorte(metodo="fundo", margem_seguranca=0, largura_maxima_analise=1200),
    )

    assert resultado.cortar
    assert calcular_iou(resultado.retangulo, esperado) > 0.95


def test_fundo_aguenta_poeira_do_scanner():
    """Pontinhos isolados de sujeira não devem virar o "conteúdo" da imagem."""
    esperado = Retangulo(150, 120, 500, 350)
    imagem = montar_scan((600, 800), esperado, (255, 255, 255))
    for x, y in [(10, 10), (780, 20), (20, 580), (770, 570), (400, 590)]:
        imagem[y : y + 3, x : x + 3] = (30, 30, 30)

    resultado = sugerir_corte(imagem, EXATO_FUNDO)

    assert calcular_iou(resultado.retangulo, esperado) > 0.95


def test_fundo_conteudo_minusculo_e_tratado_como_erro():
    """Uma manchinha no canto não é a foto: melhor não cortar nada."""
    imagem = np.full((600, 800, 3), 255, dtype=np.uint8)
    imagem[500:560, 700:760] = 40  # 0.75% da área

    resultado = sugerir_corte(imagem, EXATO_FUNDO)

    assert not resultado.cortar
    assert "suspeito" in resultado.motivo


# --- critério de aceite: método "bordas" (o cenário real deste acervo) ---------


def test_aceite_bordas_foto_ocupando_a_altura_toda():
    """O caso mais comum: você enquadra a foto impressa, sobrando álbum nas laterais."""
    esperado = Retangulo(150, 0, 500, 600)
    imagem = montar_foto_impressa((600, 800), esperado, (225, 228, 232))

    resultado = sugerir_corte(imagem, EXATO_BORDAS)

    assert resultado.cortar
    iou = calcular_iou(resultado.retangulo, esperado)
    assert iou > 0.95, f"IoU {iou:.4f}"


def test_aceite_bordas_foto_ocupando_a_largura_toda():
    """O mesmo, com a foto deitada: sobra álbum em cima e embaixo."""
    esperado = Retangulo(0, 100, 800, 400)
    imagem = montar_foto_impressa((600, 800), esperado, (225, 228, 232))

    resultado = sugerir_corte(imagem, EXATO_BORDAS)

    assert resultado.cortar
    assert calcular_iou(resultado.retangulo, esperado) > 0.95


def test_bordas_nao_corta_quando_a_foto_preenche_o_quadro():
    """Regressão do bug que motivou este método.

    Numa foto de celular em que o papel impresso preenche a imagem inteira, não
    há borda nenhuma para achar. O método antigo confundia a parede da própria
    foto com o fundo e propunha remover 75% da área, decepando o assunto.
    """
    inteira = Retangulo(0, 0, 800, 600)
    imagem = montar_foto_impressa((600, 800), inteira, (225, 228, 232))

    resultado = sugerir_corte(imagem, EXATO_BORDAS)

    assert not resultado.cortar
    assert resultado.retangulo == inteira


@pytest.mark.parametrize(
    "cor_entorno",
    [(225, 228, 232), (45, 48, 52), (90, 120, 160)],
    ids=["album claro", "album escuro", "mesa de madeira"],
)
def test_bordas_nunca_corta_dentro_da_foto(cor_entorno):
    """A garantia que mais importa: sobrar fundo é chato, decepar a foto é grave.

    Mesmo nos casos em que o robô não acha todas as quatro bordas, o retângulo
    sugerido não pode avançar para dentro da foto impressa.
    """
    esperado = Retangulo(120, 80, 600, 420)
    imagem = montar_foto_impressa((600, 800), esperado, cor_entorno)

    resultado = sugerir_corte(imagem, EXATO_BORDAS)

    invasao = corta_dentro_de(resultado.retangulo, esperado)
    assert invasao <= 6, f"o corte entrou {invasao}px na foto"


def test_bordas_encontra_cada_lado_de_forma_independente():
    """Achar dois lados e não os outros dois é um resultado válido, não um erro."""
    esperado = Retangulo(150, 0, 500, 600)
    imagem = montar_foto_impressa((600, 800), esperado, (225, 228, 232))

    achado, lados = retangulo_por_bordas(imagem, EXATO_BORDAS)

    assert lados == 2, "só existem borda esquerda e direita nesta imagem"
    assert achado.y == 0 and achado.altura == 600, "topo e base ficam no limite"


def test_bordas_prefere_a_borda_mais_interna():
    """Numa foto de álbum, a lombada metálica marca mais que o papel.

    Pegar a borda mais forte pegaria a lombada e deixaria página no corte.
    Pegar a mais interna erra sobrando menos.
    """
    esperado = Retangulo(150, 0, 500, 600)
    imagem = montar_foto_impressa((600, 800), esperado, (225, 228, 232))
    # Uma listra escura bem marcada ANTES da borda do papel: é a lombada.
    imagem[:, 40:52] = 10

    resultado = sugerir_corte(imagem, EXATO_BORDAS)

    assert resultado.retangulo.x > 100, "não podia parar na lombada, em x=46"
    assert calcular_iou(resultado.retangulo, esperado) > 0.95


# --- a decisão de cortar ou não (vale para os dois métodos) ---------------------


def test_borda_pequena_nao_gera_corte():
    """Regra do projeto: só cortar se a borda for grande (mais de 3% da área)."""
    # Moldura de 2 pixels em 800x600: remove menos de 1% da área.
    esperado = Retangulo(2, 2, 796, 596)
    imagem = montar_scan((600, 800), esperado, (255, 255, 255))

    resultado = sugerir_corte(imagem, EXATO_FUNDO)

    assert not resultado.cortar
    assert resultado.retangulo == Retangulo(0, 0, 800, 600)
    assert "borda pequena" in resultado.motivo


def test_imagem_toda_uniforme_nao_gera_corte():
    """Sem nada diferente do fundo, não há o que cortar."""
    imagem = np.full((300, 400, 3), 200, dtype=np.uint8)

    for parametros in (EXATO_FUNDO, EXATO_BORDAS):
        resultado = sugerir_corte(imagem, parametros)
        assert not resultado.cortar
        assert resultado.retangulo == Retangulo(0, 0, 400, 300)


def test_sem_corte_devolve_sempre_a_imagem_inteira():
    """Assim quem consome o resultado não precisa tratar caso especial."""
    imagem = np.full((120, 250, 3), 90, dtype=np.uint8)

    for parametros in (EXATO_FUNDO, EXATO_BORDAS):
        resultado = sugerir_corte(imagem, parametros)
        assert resultado.retangulo == Retangulo(0, 0, 250, 120)
        assert resultado.fracao_removida == 0.0


# --- a margem de segurança ------------------------------------------------------


def test_margem_de_seguranca_encolhe_o_retangulo():
    esperado = Retangulo(100, 100, 400, 300)
    imagem = montar_scan((600, 800), esperado, (255, 255, 255))

    com_margem = sugerir_corte(
        imagem,
        ParametrosCorte(metodo="fundo", margem_seguranca=5, largura_maxima_analise=0),
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


# --- as peças do método "fundo" -------------------------------------------------


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


# --- as peças do método "bordas" ------------------------------------------------


def test_perfil_sobe_na_borda_e_fica_baixo_no_resto():
    """O coração do método: a borda do papel atravessa a imagem, o resto não."""
    imagem = np.full((400, 400, 3), 230, dtype=np.uint8)
    imagem[:, 100:300] = 40  # faixa escura de cima a baixo: duas bordas verticais
    imagem[190:210, 150:170] = 230  # detalhe pequeno dentro, não é borda do papel

    colunas, linhas = perfil_de_bordas(imagem, 25, 9)

    # Uma borda perfeitamente nítida não chega a 1.0: ela ocupa umas 6 das 9
    # colunas da janela de suavização, então satura por volta de 0.67. O que
    # importa é passar de `forca_borda` (0.55), e passa com folga.
    assert colunas[100] > 0.6, "a borda vertical atravessa a altura toda"
    assert colunas[300] > 0.6
    assert colunas[200] < 0.2, "no meio da faixa não há borda vertical"
    assert linhas.max() < 0.5, "não há nenhuma borda horizontal atravessando"


def test_perfil_ignora_detalhe_que_nao_atravessa():
    """Um objeto no meio da foto gera borda, mas só num pedaço da altura."""
    imagem = np.full((400, 400, 3), 230, dtype=np.uint8)
    imagem[150:250, 150:250] = 30  # quadrado central: borda em 25% da altura

    colunas, _ = perfil_de_bordas(imagem, 25, 9)

    assert colunas.max() < 0.4, "25% da altura não deve passar de forca_borda"


def test_bordas_sem_nenhuma_borda_forte_devolve_zero_lados():
    imagem = np.full((300, 400, 3), 180, dtype=np.uint8)

    achado, lados = retangulo_por_bordas(imagem, EXATO_BORDAS)

    assert lados == 0
    assert achado == Retangulo(0, 0, 400, 300)


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

    saida = desenhar_debug(imagem, sugerir_corte(imagem, EXATO_FUNDO))

    assert np.array_equal(imagem, copia), "a original não pode ser modificada"
    assert saida.shape == imagem.shape
    assert not np.array_equal(saida, imagem), "o debug precisa desenhar algo"


def test_parametros_saem_do_config():
    from robo_fotos.config import carregar_config

    p = ParametrosCorte.do_config(carregar_config())

    assert p.metodo == "bordas"
    assert p.forca_borda == pytest.approx(0.55)
    assert p.margem_seguranca == 3
    assert p.area_minima_removida == pytest.approx(0.03)
