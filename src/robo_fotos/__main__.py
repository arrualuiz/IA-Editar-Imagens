"""CLI do robô de fotos: `python -m robo_fotos <comando>`.

Cada comando corresponde a uma etapa do projeto. Os imports dos módulos pesados
(PyTorch, OpenCV) ficam dentro das funções, e não no topo do arquivo, para que
`--help` responda instantaneamente sem carregar a rede neural.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import carregar_config

# Comandos ainda não implementados apontam para a etapa que os traz.
ETAPA_DE = {
    "dividir": "Etapa 1 — Lotes",
    "sugerir-corte": "Etapa 2 — Corte",
    "sugerir": "Etapa 3 — Orientação v1",
    "treinar-base": "Etapa 3 — Orientação v1",
    "revisar": "Etapa 4 — Página de revisão",
    "exportar": "Etapa 5 — Exportar",
    "treinar": "Etapa 6 — Treino com correções",
}


def _ainda_nao(comando: str) -> int:
    etapa = ETAPA_DE.get(comando, "uma etapa futura")
    print(f"O comando '{comando}' ainda não foi implementado. Ele chega na {etapa}.")
    return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Mostra onde o app está olhando e o que já existe. Útil para conferir o setup."""
    cfg = carregar_config()
    print(f"robo_fotos {__version__}")
    print(f"Python      {sys.version.split()[0]}")
    print(f"Raiz        {cfg.raiz}")
    print()

    def conta_fotos(pasta) -> str:
        if not pasta.exists():
            return "(pasta não existe)"
        n = sum(
            1
            for p in pasta.rglob("*")
            if p.is_file() and p.suffix.lower() in cfg.extensoes_foto
        )
        return f"{n} foto(s)"

    print("Pastas:")
    for nome, pasta in [
        ("entrada", cfg.entrada),
        ("treino_base", cfg.treino_base),
        ("lotes", cfg.lotes),
        ("saida", cfg.saida),
        ("dados", cfg.dados),
        ("modelos", cfg.modelos),
    ]:
        marca = "ok " if pasta.exists() else "-- "
        extra = ""
        if nome in ("entrada", "treino_base"):
            extra = f"  [{conta_fotos(pasta)}]"
        print(f"  {marca}{nome:<12} {pasta}{extra}")

    lotes_existentes = (
        sorted(p.name for p in cfg.lotes.glob("lote_*") if p.is_dir())
        if cfg.lotes.exists()
        else []
    )
    print()
    print(f"Lotes criados: {', '.join(lotes_existentes) if lotes_existentes else 'nenhum'}")

    n_correcoes = 0
    if cfg.correcoes.exists():
        with cfg.correcoes.open("r", encoding="utf-8") as f:
            n_correcoes = sum(1 for linha in f if linha.strip())
    print(f"Correções registradas: {n_correcoes}")

    modelos = (
        sorted(p.name for p in cfg.modelos.glob("*.pt")) if cfg.modelos.exists() else []
    )
    print(f"Modelos treinados: {', '.join(modelos) if modelos else 'nenhum'}")
    return 0


def cmd_dividir(args: argparse.Namespace) -> int:
    return _ainda_nao("dividir")


def cmd_sugerir_corte(args: argparse.Namespace) -> int:
    return _ainda_nao("sugerir-corte")


def cmd_sugerir(args: argparse.Namespace) -> int:
    return _ainda_nao("sugerir")


def cmd_treinar_base(args: argparse.Namespace) -> int:
    return _ainda_nao("treinar-base")


def cmd_revisar(args: argparse.Namespace) -> int:
    return _ainda_nao("revisar")


def cmd_exportar(args: argparse.Namespace) -> int:
    return _ainda_nao("exportar")


def cmd_treinar(args: argparse.Namespace) -> int:
    return _ainda_nao("treinar")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m robo_fotos",
        description=(
            "Robô que corrige a orientação e corta as bordas das suas fotos, "
            "aprendendo com as suas correções."
        ),
        epilog=(
            "Fluxo normal de trabalho:\n"
            "  1) python -m robo_fotos dividir\n"
            "  2) python -m robo_fotos sugerir lote_001\n"
            "  3) python -m robo_fotos revisar lote_001\n"
            "  4) python -m robo_fotos exportar lote_001\n"
            "  5) python -m robo_fotos treinar\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"robo_fotos {__version__}")

    subs = parser.add_subparsers(dest="comando", metavar="<comando>")

    p = subs.add_parser("info", help="mostra o estado do projeto (pastas, lotes, modelos)")
    p.set_defaults(func=cmd_info)

    p = subs.add_parser(
        "dividir",
        help="lê entrada/ e cria lotes/lote_001, lote_002... com 50 fotos cada",
    )
    p.add_argument(
        "--tamanho", type=int, default=None, help="fotos por lote (padrão: o do config.yaml)"
    )
    p.add_argument(
        "--refazer", action="store_true", help="apaga os lotes existentes e divide de novo"
    )
    p.set_defaults(func=cmd_dividir)

    p = subs.add_parser(
        "treinar-base",
        help="treino inicial da orientação usando as fotos de treino_base/",
    )
    p.set_defaults(func=cmd_treinar_base)

    p = subs.add_parser(
        "sugerir-corte", help="calcula e desenha o corte sugerido de um lote (debug)"
    )
    p.add_argument("lote", help="nome do lote, ex.: lote_001")
    p.set_defaults(func=cmd_sugerir_corte)

    p = subs.add_parser(
        "sugerir",
        help="gera dados/sugestoes/<lote>.json com rotação, corte e confiança",
    )
    p.add_argument("lote", help="nome do lote, ex.: lote_001")
    p.set_defaults(func=cmd_sugerir)

    p = subs.add_parser(
        "revisar", help="abre a página de revisão no navegador para aprovar/corrigir"
    )
    p.add_argument("lote", help="nome do lote, ex.: lote_001")
    p.add_argument("--porta", type=int, default=None, help="porta do servidor local")
    p.set_defaults(func=cmd_revisar)

    p = subs.add_parser(
        "exportar",
        help="aplica rotação e corte, preserva o EXIF e grava em saida/<lote>/",
    )
    p.add_argument("lote", help="nome do lote, ex.: lote_001")
    p.set_defaults(func=cmd_exportar)

    p = subs.add_parser(
        "treinar",
        help="treina uma nova versão do modelo com as suas correções e compara com a anterior",
    )
    p.set_defaults(func=cmd_treinar)

    return parser


def _forcar_utf8() -> None:
    """Garante acentos corretos mesmo quando a saída é redirecionada para um arquivo.

    No Windows, quando a saída não é o console, o Python usa a codificação
    antiga do sistema (cp1252) e acentos viram erro. Aqui pedimos UTF-8.
    """
    for fluxo in (sys.stdout, sys.stderr):
        recodificar = getattr(fluxo, "reconfigure", None)
        if recodificar is not None:
            try:
                recodificar(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass  # se não der, seguimos com a codificação padrão


def main(argv: list[str] | None = None) -> int:
    _forcar_utf8()
    parser = construir_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    # Num clone novo do repositório as pastas de trabalho não existem (o Git não
    # guarda pasta vazia). Criamos aqui. `entrada/` nunca é tocada.
    carregar_config().criar_pastas_de_trabalho()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
