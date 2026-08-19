"""Deterministic character-level tokenizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class CharacterTokenizer:
    """Map Unicode characters to stable integer ids.

    The four special ids are intentionally fixed so checkpoints are easy to
    inspect: pad=0, bos=1, eos=2, unk=3. Remaining characters are sorted by
    Unicode code point for deterministic vocabulary construction.
    """

    SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __init__(self, characters: Iterable[str]) -> None:
        unique = {character for character in characters}
        if any(len(character) != 1 for character in unique):
            raise ValueError("CharacterTokenizer vocabulary entries must be one character")
        self.id_to_token = list(self.SPECIAL_TOKENS) + sorted(
            unique.difference(self.SPECIAL_TOKENS)
        )
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        return cls(text)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self.token_to_id.get(character, self.unk_id) for character in text]
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        decoded: list[str] = []
        for index in ids:
            if not 0 <= int(index) < self.vocab_size:
                raise ValueError(f"Token id out of range: {index}")
            token = self.id_to_token[int(index)]
            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue
            decoded.append(token)
        return "".join(decoded)

    def state_dict(self) -> dict[str, object]:
        return {"id_to_token": self.id_to_token}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "CharacterTokenizer":
        tokens = state.get("id_to_token")
        if not isinstance(tokens, list) or tokens[:4] != list(cls.SPECIAL_TOKENS):
            raise ValueError("Invalid tokenizer state")
        tokenizer = cls(tokens[4:])
        if tokenizer.id_to_token != tokens:
            raise ValueError("Tokenizer state contains duplicate or invalid tokens")
        return tokenizer

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.state_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))
