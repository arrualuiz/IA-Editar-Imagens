"""CLI do robô de fotos: `python -m robo_fotos <comando>`.

Cada comando corresponde a uma etapa do projeto. Os imports dos módulos pesados
(PyTorch, OpenCV) ficam dentro das funções, e não no topo do arquivo, para que
`--help` responda instantaneamente sem carregar a rede neural.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    """Divide as fotos de entrada/ em lotes, ordenadas por data."""
    from .lotes import criar_lotes

    cfg = carregar_config()
    if not cfg.entrada.exists():
        print(f"A pasta de entrada não existe: {cfg.entrada}")
        return 1

    print(f"Lendo {cfg.entrada} ...")
    try:
        resumo = criar_lotes(cfg, tamanho=args.tamanho, refazer=args.refazer)
    except FileExistsError as erro:
        print(erro)
        return 1

    if not resumo:
        print("Nenhuma foto encontrada. Coloque as fotos em entrada/ e rode de novo.")
        print(f"Formatos aceitos: {', '.join(sorted(cfg.extensoes_foto))}")
        return 1

    total = sum(r["total"] for r in resumo)
    print(f"\n{total} foto(s) em {len(resumo)} lote(s):\n")
    print(f"  {'lote':<12} {'fotos':>5}  período")
    for r in resumo:
        de = r["primeira_data"].strftime("%d/%m/%Y")
        ate = r["ultima_data"].strftime("%d/%m/%Y")
        periodo = de if de == ate else f"{de} a {ate}"
        print(f"  {r['lote']:<12} {r['total']:>5}  {periodo}")

    print(f"\nAs cópias estão em {cfg.lotes}")
    print(f"Próximo passo: python -m robo_fotos sugerir {resumo[0]['lote']}")
    return 0


def cmd_sugerir_corte(args: argparse.Namespace) -> int:
    """Calcula o corte de um lote e grava imagens de debug para você conferir."""
    from .corte import (
        ParametrosCorte,
        abrir_como_bgr,
        desenhar_debug,
        salvar_bgr,
        sugerir_corte,
    )
    from .lotes import carregar_manifesto

    cfg = carregar_config()
    try:
        manifesto = carregar_manifesto(cfg, args.lote)
    except FileNotFoundError as erro:
        print(erro)
        return 1

    parametros = ParametrosCorte.do_config(cfg)
    pasta_lote = cfg.lotes / args.lote
    pasta_debug = cfg.dados / "debug_corte" / args.lote
    limite = args.limite or len(manifesto["fotos"])

    print(f"Lote {args.lote}: {len(manifesto['fotos'])} foto(s)")
    print(
        f"Parâmetros: limiar={parametros.limiar} "
        f"margem={parametros.margem_seguranca} "
        f"área mínima={parametros.area_minima_removida:.0%}"
    )
    print()

    cortadas = 0
    falhas = 0
    resultados = []
    for item in manifesto["fotos"][:limite]:
        caminho = pasta_lote / item["arquivo"]
        try:
            imagem = abrir_como_bgr(caminho)
        except Exception as erro:
            print(f"  !! {item['arquivo']}: não consegui abrir ({erro})")
            falhas += 1
            continue

        resultado = sugerir_corte(imagem, parametros)
        resultados.append((item["arquivo"], resultado))
        if resultado.cortar:
            cortadas += 1

        if not args.sem_debug:
            salvar_bgr(
                desenhar_debug(imagem, resultado),
                pasta_debug / f"{Path(item['arquivo']).stem}.jpg",
            )

    analisadas = len(resultados)
    if not analisadas:
        print("Nenhuma foto pôde ser analisada.")
        return 1

    print(f"  com corte sugerido : {cortadas}")
    print(f"  sem corte          : {analisadas - cortadas}")
    if falhas:
        print(f"  falhas ao abrir    : {falhas}")

    # As maiores bordas primeiro: são as que valem conferir no debug.
    com_corte = sorted(
        (r for r in resultados if r[1].cortar),
        key=lambda r: r[1].fracao_removida,
        reverse=True,
    )
    if com_corte:
        print("\nMaiores bordas encontradas:")
        for nome, r in com_corte[:5]:
            print(f"  {r.fracao_removida:>6.1%}  {nome}  -> {r.retangulo.como_lista()}")

    if not args.sem_debug:
        print(f"\nImagens de debug em {pasta_debug}")
        print("Verde = vai cortar · Vermelho = sem corte")
    return 0


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
    p.add_argument(
        "--limite", type=int, default=None, help="analisa só as N primeiras fotos"
    )
    p.add_argument(
        "--sem-debug",
        action="store_true",
        help="não gera as imagens de debug (só o resumo no terminal)",
    )
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
