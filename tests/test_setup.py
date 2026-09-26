"""Testes da Etapa 0: o setup está de pé e a configuração é lida corretamente."""

from __future__ import annotations

import pytest

from robo_fotos.config import carregar_config
from robo_fotos.__main__ import construir_parser, main


def test_help_funciona(capsys):
    """Critério de aceite da Etapa 0: `python -m robo_fotos --help` funciona."""
    with pytest.raises(SystemExit) as saida:
        main(["--help"])
    assert saida.value.code == 0
    texto = capsys.readouterr().out
    assert "dividir" in texto
    assert "revisar" in texto


def test_sem_comando_mostra_ajuda(capsys):
    assert main([]) == 0
    assert "usage:" in capsys.readouterr().out


def test_todos_os_comandos_estao_registrados():
    parser = construir_parser()
    # Pega a lista de subcomandos declarados no parser.
    acoes = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    registrados = set(acoes[0].choices)
    esperados = {
        "info",
        "dividir",
        "treinar-base",
        "sugerir-corte",
        "sugerir",
        "revisar",
        "exportar",
        "treinar",
    }
    assert esperados <= registrados


def test_comandos_futuros_avisam_e_falham(capsys):
    """Comando ainda não implementado deve avisar em qual etapa ele chega."""
    assert main(["dividir"]) == 1
    assert "Etapa 1" in capsys.readouterr().out


def test_config_resolve_caminhos_absolutos():
    cfg = carregar_config()
    assert cfg.entrada.is_absolute()
    assert cfg.entrada.name == "entrada"
    assert cfg.correcoes.name == "correcoes.jsonl"
    # sugestoes fica dentro de dados
    assert cfg.sugestoes.parent == cfg.dados


def test_config_le_os_parametros_do_yaml():
    cfg = carregar_config()
    assert cfg.lotes_cfg["tamanho"] == 50
    assert cfg.exportar["jpeg_qualidade"] == 95
    assert ".jpg" in cfg.extensoes_foto
    assert ".heic" in cfg.extensoes_foto
    # vídeo não é foto
    assert ".mp4" not in cfg.extensoes_foto


def test_config_inexistente_nao_quebra(tmp_path):
    cfg = carregar_config(tmp_path / "nao_existe.yaml")
    assert cfg.bruto == {}
    assert cfg.extensoes_foto  # cai no padrão
