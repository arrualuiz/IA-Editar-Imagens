"""Divide as fotos de `entrada/` em lotes de 50, ordenadas por data.

Por que ordenar por data? Fotos tiradas ou escaneadas na mesma sessão tendem a
ter o mesmo problema (mesmo scanner, mesma posição do papel). Mantê-las juntas
no mesmo lote faz você revisar coisas parecidas em sequência, o que é mais
rápido, e faz o robô aprender um padrão por vez.

Esta etapa NUNCA escreve em `entrada/`: só lê e copia para `lotes/`.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import Config

# Nome do arquivo que guarda, dentro de cada lote, de onde veio cada foto.
# As etapas seguintes usam isso para achar o original e ler o EXIF dele.
NOME_MANIFESTO = "_lote.json"

_RE_LOTE = re.compile(r"^lote_(\d+)$")

# Tags EXIF que interessam aqui (os números são o padrão EXIF).
_TAG_DATETIME = 0x0132               # DateTime, na IFD0
_TAG_EXIF_IFD = 0x8769               # ponteiro para o bloco ExifIFD
_TAG_DATETIME_ORIGINAL = 0x9003      # DateTimeOriginal, dentro do ExifIFD
_TAG_DATETIME_DIGITALIZADA = 0x9004  # DateTimeDigitized, idem

_heic_registrado = False


def _registrar_heic() -> None:
    """Ensina o Pillow a abrir HEIC (fotos de iPhone). Só precisa uma vez."""
    global _heic_registrado
    if _heic_registrado:
        return
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass  # sem pillow-heif o app segue funcionando com JPG e PNG
    _heic_registrado = True


def _texto_para_data(texto: object) -> datetime | None:
    """Converte a data do EXIF (formato "2020:02:04 18:52:56") em datetime."""
    if not isinstance(texto, str):
        return None
    limpo = texto.strip().rstrip("\x00")
    if not limpo or limpo.startswith("0000"):
        return None
    for formato in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d"):
        try:
            return datetime.strptime(limpo[: len(formato) + 4], formato)
        except ValueError:
            continue
    return None


def data_do_exif(caminho: Path) -> datetime | None:
    """Lê a data de captura no EXIF, na ordem de confiabilidade."""
    _registrar_heic()
    try:
        from PIL import Image

        with Image.open(caminho) as img:
            exif = img.getexif()
            if not exif:
                return None
            bloco = exif.get_ifd(_TAG_EXIF_IFD)
            for tag in (_TAG_DATETIME_ORIGINAL, _TAG_DATETIME_DIGITALIZADA):
                data = _texto_para_data(bloco.get(tag))
                if data:
                    return data
            return _texto_para_data(exif.get(_TAG_DATETIME))
    except Exception:
        # Foto corrompida ou formato que o Pillow não abre: seguimos sem data.
        return None


def json_do_takeout(caminho: Path) -> Path | None:
    """Acha o .json que o Google Takeout deixa ao lado da foto.

    O Takeout não é consistente nos nomes: pode ser IMG_0123.jpg.json,
    IMG_0123.jpg.supplemental-metadata.json ou versões com o nome cortado,
    porque o Google trunca nomes longos. Tentamos os casos conhecidos e, se
    nada casar, procuramos qualquer .json que comece com o nome da foto.
    """
    pasta = caminho.parent
    candidatos = [
        pasta / f"{caminho.name}.json",
        pasta / f"{caminho.name}.supplemental-metadata.json",
        pasta / f"{caminho.stem}.json",
    ]
    for candidato in candidatos:
        if candidato.is_file():
            return candidato
    # Último recurso: nome truncado pelo Google (ex.: ".supplemental-m.json").
    for achado in sorted(pasta.glob(f"{caminho.name}*.json")):
        return achado
    return None


def data_do_json(caminho_json: Path) -> datetime | None:
    """Lê photoTakenTime do JSON do Takeout (um timestamp Unix em UTC)."""
    try:
        with caminho_json.open("r", encoding="utf-8") as f:
            dados = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(dados, dict):
        return None
    for chave in ("photoTakenTime", "creationTime"):
        bloco = dados.get(chave)
        if isinstance(bloco, dict) and bloco.get("timestamp"):
            try:
                # fromtimestamp converte para a hora local, igual ao EXIF.
                return datetime.fromtimestamp(int(bloco["timestamp"]))
            except (ValueError, OSError, OverflowError):
                continue
    return None


def data_da_foto(caminho: Path) -> tuple[datetime, str]:
    """Descobre a data da foto e de onde ela veio.

    Ordem: EXIF -> JSON do Takeout -> data de modificação do arquivo. A última
    é um chute, porque copiar ou baixar o arquivo costuma mudar essa data.
    """
    data = data_do_exif(caminho)
    if data:
        return data, "exif"

    sidecar = json_do_takeout(caminho)
    if sidecar:
        data = data_do_json(sidecar)
        if data:
            return data, "json"

    return datetime.fromtimestamp(caminho.stat().st_mtime), "arquivo"


@dataclass(frozen=True)
class Foto:
    """Uma foto encontrada em `entrada/`, já com a data resolvida."""

    caminho: Path      # onde ela está de verdade (dentro de entrada/)
    relativo: str      # caminho relativo a entrada/, para o manifesto
    data: datetime
    origem_data: str   # "exif", "json" ou "arquivo"


def listar_fotos(cfg: Config) -> list[Foto]:
    """Varre `entrada/` (inclusive subpastas) e devolve as fotos em ordem de data.

    Vídeos, .json do Takeout e qualquer outra extensão ficam de fora.
    """
    if not cfg.entrada.exists():
        return []

    extensoes = cfg.extensoes_foto
    fotos: list[Foto] = []
    for caminho in sorted(cfg.entrada.rglob("*")):
        if not caminho.is_file():
            continue
        if caminho.suffix.lower() not in extensoes:
            continue
        if caminho.name.startswith("."):
            continue  # arquivos ocultos / lixo do sistema
        data, origem = data_da_foto(caminho)
        fotos.append(
            Foto(
                caminho=caminho,
                relativo=caminho.relative_to(cfg.entrada).as_posix(),
                data=data,
                origem_data=origem,
            )
        )

    # Desempate pelo caminho, para a divisão ser sempre igual se você rodar de novo.
    fotos.sort(key=lambda f: (f.data, f.relativo))
    return fotos


def dividir_em_lotes(fotos: list[Foto], tamanho: int) -> list[list[Foto]]:
    """Corta a lista em pedaços de `tamanho`. O último lote pode vir menor."""
    if tamanho < 1:
        raise ValueError("o tamanho do lote precisa ser pelo menos 1")
    return [fotos[i : i + tamanho] for i in range(0, len(fotos), tamanho)]


def lotes_existentes(cfg: Config) -> list[Path]:
    """Pastas lote_XXX que já existem, em ordem."""
    if not cfg.lotes.exists():
        return []
    achados = [p for p in cfg.lotes.iterdir() if p.is_dir() and _RE_LOTE.match(p.name)]
    return sorted(achados, key=lambda p: int(_RE_LOTE.match(p.name).group(1)))


def _nome_livre(nome: str, usados: set[str]) -> str:
    """Evita duas fotos com o mesmo nome no lote (comum em subpastas do Takeout)."""
    if nome not in usados:
        usados.add(nome)
        return nome
    base = Path(nome)
    n = 2
    while f"{base.stem}_{n}{base.suffix}" in usados:
        n += 1
    novo = f"{base.stem}_{n}{base.suffix}"
    usados.add(novo)
    return novo


def criar_lotes(
    cfg: Config, tamanho: int | None = None, refazer: bool = False
) -> list[dict]:
    """Divide `entrada/` em lotes e devolve um resumo de cada um.

    Levanta FileExistsError se já houver lotes e `refazer` for False, para não
    bagunçar uma revisão em andamento.
    """
    tamanho = tamanho or int(cfg.lotes_cfg.get("tamanho", 50))
    copiar = bool(cfg.lotes_cfg.get("copiar_arquivos", True))

    antigos = lotes_existentes(cfg)
    if antigos and not refazer:
        raise FileExistsError(
            f"já existem {len(antigos)} lote(s) em {cfg.lotes}. "
            "Use --refazer para apagá-los e dividir de novo."
        )

    fotos = listar_fotos(cfg)
    if not fotos:
        return []

    if antigos:
        for pasta in antigos:
            shutil.rmtree(pasta)

    cfg.lotes.mkdir(parents=True, exist_ok=True)

    resumo: list[dict] = []
    for indice, grupo in enumerate(dividir_em_lotes(fotos, tamanho), start=1):
        nome_lote = f"lote_{indice:03d}"
        pasta = cfg.lotes / nome_lote
        pasta.mkdir(parents=True, exist_ok=True)

        usados: set[str] = set()
        itens = []
        for foto in grupo:
            nome = _nome_livre(foto.caminho.name, usados)
            if copiar:
                # copy2 preserva a data de modificação do arquivo original.
                shutil.copy2(foto.caminho, pasta / nome)
            itens.append(
                {
                    "arquivo": nome,
                    "origem": foto.relativo,
                    "data": foto.data.isoformat(timespec="seconds"),
                    "origem_data": foto.origem_data,
                }
            )

        manifesto = {
            "lote": nome_lote,
            "criado_em": datetime.now().isoformat(timespec="seconds"),
            "arquivos_copiados": copiar,
            "total": len(itens),
            "fotos": itens,
        }
        with (pasta / NOME_MANIFESTO).open("w", encoding="utf-8") as f:
            json.dump(manifesto, f, ensure_ascii=False, indent=2)

        resumo.append(
            {
                "lote": nome_lote,
                "total": len(itens),
                "primeira_data": grupo[0].data,
                "ultima_data": grupo[-1].data,
                "pasta": pasta,
            }
        )

    return resumo


def carregar_manifesto(cfg: Config, nome_lote: str) -> dict:
    """Lê o _lote.json de um lote. Usado pelas etapas seguintes."""
    caminho = cfg.lotes / nome_lote / NOME_MANIFESTO
    if not caminho.is_file():
        raise FileNotFoundError(
            f"lote '{nome_lote}' não encontrado (esperado: {caminho}). "
            "Rode `python -m robo_fotos dividir` primeiro."
        )
    with caminho.open("r", encoding="utf-8") as f:
        return json.load(f)
