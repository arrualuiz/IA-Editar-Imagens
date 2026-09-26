"""Carrega o config.yaml e resolve os caminhos do projeto.

A ideia é que nenhum outro módulo precise saber onde as pastas ficam: todos
perguntam aqui. Assim o app funciona igual no Windows e no Mac, porque usamos
`pathlib` e nunca caminhos fixos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Este arquivo está em <raiz>/src/robo_fotos/config.py, então a raiz do projeto
# fica três níveis acima.
RAIZ_PROJETO = Path(__file__).resolve().parents[2]

ARQUIVO_CONFIG = RAIZ_PROJETO / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Configuração do projeto, já com os caminhos resolvidos."""

    bruto: dict[str, Any]
    raiz: Path

    # --- atalhos para as pastas -------------------------------------------------
    @property
    def entrada(self) -> Path:
        return self._pasta("entrada")

    @property
    def lotes(self) -> Path:
        return self._pasta("lotes")

    @property
    def saida(self) -> Path:
        return self._pasta("saida")

    @property
    def treino_base(self) -> Path:
        return self._pasta("treino_base")

    @property
    def dados(self) -> Path:
        return self._pasta("dados")

    @property
    def modelos(self) -> Path:
        return self._pasta("modelos")

    @property
    def sugestoes(self) -> Path:
        return self.dados / "sugestoes"

    @property
    def correcoes(self) -> Path:
        return self.dados / "correcoes.jsonl"

    # --- atalhos para as seções de parâmetros -----------------------------------
    @property
    def lotes_cfg(self) -> dict[str, Any]:
        return self.bruto.get("lotes", {})

    @property
    def corte(self) -> dict[str, Any]:
        return self.bruto.get("corte", {})

    @property
    def orientacao(self) -> dict[str, Any]:
        return self.bruto.get("orientacao", {})

    @property
    def exportar(self) -> dict[str, Any]:
        return self.bruto.get("exportar", {})

    @property
    def servidor(self) -> dict[str, Any]:
        return self.bruto.get("servidor", {})

    @property
    def extensoes_foto(self) -> set[str]:
        """Extensões aceitas, em minúsculas e com o ponto (ex.: {'.jpg', '.png'})."""
        brutas = self.lotes_cfg.get("extensoes_foto", [".jpg", ".jpeg", ".png"])
        return {str(e).lower() for e in brutas}

    def _pasta(self, nome: str) -> Path:
        pastas = self.bruto.get("pastas", {})
        relativo = pastas.get(nome, nome)
        caminho = Path(relativo)
        # Um caminho absoluto no config.yaml é respeitado; um relativo parte da raiz.
        return caminho if caminho.is_absolute() else self.raiz / caminho

    def criar_pastas_de_trabalho(self) -> None:
        """Cria as pastas que o app escreve. Nunca toca em `entrada/`."""
        for pasta in (self.lotes, self.saida, self.dados, self.sugestoes, self.modelos):
            pasta.mkdir(parents=True, exist_ok=True)


def carregar_config(caminho: Path | None = None) -> Config:
    """Lê o config.yaml. Se o arquivo não existir, usa os valores padrão vazios."""
    arquivo = caminho or ARQUIVO_CONFIG
    if arquivo.exists():
        with arquivo.open("r", encoding="utf-8") as f:
            bruto = yaml.safe_load(f) or {}
    else:
        bruto = {}
    raiz = arquivo.resolve().parent if caminho else RAIZ_PROJETO
    return Config(bruto=bruto, raiz=raiz)
