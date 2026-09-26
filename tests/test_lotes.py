"""Testes da Etapa 1: divisão em lotes.

Usamos fotos sintéticas minúsculas (40x30 pixels) com data EXIF escrita por nós.
Assim o teste é rápido, não depende das fotos de verdade e controla exatamente
qual data cada foto tem.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml
from PIL import Image

from robo_fotos.config import carregar_config
from robo_fotos.lotes import (
    NOME_MANIFESTO,
    carregar_manifesto,
    criar_lotes,
    data_da_foto,
    dividir_em_lotes,
    listar_fotos,
)

# Tags EXIF (mesmos números do módulo lotes).
TAG_DATETIME = 0x0132
TAG_EXIF_IFD = 0x8769
TAG_DATETIME_ORIGINAL = 0x9003


def escrever_foto(
    caminho: Path, data: datetime | None = None, tag_ifd0: bool = False
) -> Path:
    """Cria um JPEG minúsculo, opcionalmente com data EXIF."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (40, 30), (100, 120, 140))
    if data is None:
        img.save(caminho)
        return caminho

    texto = data.strftime("%Y:%m:%d %H:%M:%S")
    exif = Image.Exif()
    if tag_ifd0:
        exif[TAG_DATETIME] = texto
    else:
        exif.get_ifd(TAG_EXIF_IFD)[TAG_DATETIME_ORIGINAL] = texto
    img.save(caminho, exif=exif)
    return caminho


@pytest.fixture
def projeto(tmp_path: Path):
    """Um projeto de teste completo, isolado, com o seu próprio config.yaml."""
    config = {
        "pastas": {
            "entrada": "entrada",
            "lotes": "lotes",
            "saida": "saida",
            "treino_base": "treino_base",
            "dados": "dados",
            "modelos": "modelos",
        },
        "lotes": {
            "tamanho": 50,
            "extensoes_foto": [".jpg", ".jpeg", ".png", ".heic"],
            "copiar_arquivos": True,
        },
    }
    arquivo = tmp_path / "config.yaml"
    with arquivo.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    (tmp_path / "entrada").mkdir()
    return carregar_config(arquivo)


def test_aceite_120_fotos_geram_3_lotes(projeto):
    """Critério de aceite da Etapa 1: 120 fotos geram 3 lotes (50, 50, 20)."""
    base = datetime(2020, 1, 1, 8, 0, 0)
    for i in range(120):
        escrever_foto(
            projeto.entrada / f"foto_{i:03d}.jpg", base + timedelta(minutes=i)
        )

    resumo = criar_lotes(projeto)

    assert [r["total"] for r in resumo] == [50, 50, 20]
    assert [r["lote"] for r in resumo] == ["lote_001", "lote_002", "lote_003"]
    # As fotos foram copiadas de verdade (o manifesto não conta como foto).
    for r in resumo:
        copiadas = [p for p in r["pasta"].iterdir() if p.name != NOME_MANIFESTO]
        assert len(copiadas) == r["total"]


def test_ignora_videos_e_json_do_takeout(projeto):
    escrever_foto(projeto.entrada / "foto.jpg", datetime(2020, 1, 1))
    escrever_foto(projeto.entrada / "outra.png", datetime(2020, 1, 2))
    (projeto.entrada / "filme.mp4").write_bytes(b"nao e foto")
    (projeto.entrada / "clipe.MOV").write_bytes(b"nao e foto")
    (projeto.entrada / "foto.jpg.json").write_text("{}", encoding="utf-8")
    (projeto.entrada / "metadata.json").write_text("{}", encoding="utf-8")

    nomes = [f.caminho.name for f in listar_fotos(projeto)]

    assert nomes == ["foto.jpg", "outra.png"]


def test_ordena_por_data_e_nao_por_nome(projeto):
    """O nome do arquivo não deve influenciar: quem manda é a data."""
    escrever_foto(projeto.entrada / "aaa.jpg", datetime(2021, 5, 5, 10, 0))
    escrever_foto(projeto.entrada / "bbb.jpg", datetime(2019, 1, 1, 10, 0))
    escrever_foto(projeto.entrada / "ccc.jpg", datetime(2020, 3, 3, 10, 0))

    nomes = [f.caminho.name for f in listar_fotos(projeto)]

    assert nomes == ["bbb.jpg", "ccc.jpg", "aaa.jpg"]


def test_le_data_da_tag_datetime_quando_nao_ha_datetimeoriginal(projeto):
    foto = escrever_foto(
        projeto.entrada / "so_ifd0.jpg", datetime(2018, 7, 9, 11, 12, 13), tag_ifd0=True
    )

    data, origem = data_da_foto(foto)

    assert origem == "exif"
    assert data == datetime(2018, 7, 9, 11, 12, 13)


def test_usa_photoTakenTime_do_json_quando_nao_ha_exif(projeto):
    """Foto sem data no EXIF cai no JSON do Takeout, como manda o CLAUDE.md."""
    foto = escrever_foto(projeto.entrada / "sem_exif.jpg", data=None)
    esperado = datetime(2015, 6, 18, 14, 30, 0)
    sidecar = projeto.entrada / "sem_exif.jpg.supplemental-metadata.json"
    sidecar.write_text(
        json.dumps({"photoTakenTime": {"timestamp": str(int(esperado.timestamp()))}}),
        encoding="utf-8",
    )

    data, origem = data_da_foto(foto)

    assert origem == "json"
    assert data == esperado


def test_sem_exif_e_sem_json_cai_na_data_do_arquivo(projeto):
    foto = escrever_foto(projeto.entrada / "nada.jpg", data=None)

    _, origem = data_da_foto(foto)

    assert origem == "arquivo"


def test_entrada_nao_e_alterada(projeto):
    """A regra mais importante do projeto: entrada/ é somente leitura."""
    base = datetime(2020, 1, 1)
    for i in range(5):
        escrever_foto(projeto.entrada / f"f{i}.jpg", base + timedelta(days=i))

    def impressao_digital() -> dict[str, str]:
        return {
            p.relative_to(projeto.entrada).as_posix(): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted(projeto.entrada.rglob("*"))
            if p.is_file()
        }

    antes = impressao_digital()
    criar_lotes(projeto, tamanho=2)
    assert impressao_digital() == antes


def test_encontra_fotos_em_subpastas_do_takeout(projeto):
    """O Takeout vem em subpastas, uma por álbum."""
    escrever_foto(
        projeto.entrada / "Takeout" / "Album A" / "a.jpg", datetime(2020, 1, 1)
    )
    escrever_foto(
        projeto.entrada / "Takeout" / "Album B" / "b.jpg", datetime(2020, 1, 2)
    )

    fotos = listar_fotos(projeto)

    assert [f.relativo for f in fotos] == [
        "Takeout/Album A/a.jpg",
        "Takeout/Album B/b.jpg",
    ]


def test_nomes_repetidos_em_subpastas_nao_se_sobrescrevem(projeto):
    """Duas fotos podem ter o mesmo nome em álbuns diferentes."""
    escrever_foto(projeto.entrada / "A" / "IMG_1.jpg", datetime(2020, 1, 1))
    escrever_foto(projeto.entrada / "B" / "IMG_1.jpg", datetime(2020, 1, 2))

    resumo = criar_lotes(projeto)

    assert resumo[0]["total"] == 2
    manifesto = carregar_manifesto(projeto, "lote_001")
    nomes = [item["arquivo"] for item in manifesto["fotos"]]
    assert nomes == ["IMG_1.jpg", "IMG_1_2.jpg"]
    # As duas cópias existem no lote.
    for nome in nomes:
        assert (projeto.lotes / "lote_001" / nome).is_file()


def test_manifesto_guarda_a_origem_de_cada_foto(projeto):
    """O manifesto é o que permite, na Etapa 5, achar o original e ler o EXIF."""
    escrever_foto(projeto.entrada / "sub" / "x.jpg", datetime(2020, 8, 9, 10, 11, 12))

    criar_lotes(projeto)
    manifesto = carregar_manifesto(projeto, "lote_001")

    assert manifesto["lote"] == "lote_001"
    assert manifesto["total"] == 1
    item = manifesto["fotos"][0]
    assert item["arquivo"] == "x.jpg"
    assert item["origem"] == "sub/x.jpg"
    assert item["data"] == "2020-08-09T10:11:12"
    assert item["origem_data"] == "exif"


def test_nao_sobrescreve_lotes_sem_refazer(projeto):
    escrever_foto(projeto.entrada / "a.jpg", datetime(2020, 1, 1))
    criar_lotes(projeto)

    with pytest.raises(FileExistsError, match="--refazer"):
        criar_lotes(projeto)


def test_refazer_apaga_os_lotes_antigos(projeto):
    for i in range(4):
        escrever_foto(projeto.entrada / f"f{i}.jpg", datetime(2020, 1, 1 + i))
    criar_lotes(projeto, tamanho=1)
    assert len(list(projeto.lotes.glob("lote_*"))) == 4

    resumo = criar_lotes(projeto, tamanho=4, refazer=True)

    assert [r["total"] for r in resumo] == [4]
    assert sorted(p.name for p in projeto.lotes.glob("lote_*")) == ["lote_001"]


def test_entrada_vazia_nao_quebra(projeto):
    assert criar_lotes(projeto) == []


def test_dividir_em_lotes_respeita_o_tamanho():
    assert [len(g) for g in dividir_em_lotes(list(range(10)), 3)] == [3, 3, 3, 1]
    assert dividir_em_lotes([], 50) == []
    with pytest.raises(ValueError):
        dividir_em_lotes([1, 2], 0)


def test_divisao_e_repetivel(projeto):
    """Rodar duas vezes deve dar exatamente o mesmo resultado."""
    base = datetime(2020, 1, 1)
    # Duas fotos com a MESMA data, para o desempate pelo caminho entrar em ação.
    escrever_foto(projeto.entrada / "z.jpg", base)
    escrever_foto(projeto.entrada / "a.jpg", base)

    primeira = [f.relativo for f in listar_fotos(projeto)]
    segunda = [f.relativo for f in listar_fotos(projeto)]

    assert primeira == segunda == ["a.jpg", "z.jpg"]
