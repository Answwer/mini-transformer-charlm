"""A small, deterministic word-boundary BPE tokenizer with no extra dependency."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Iterable


class BPETokenizer:
    """Train and apply a compact BPE vocabulary while preserving text exactly.

    BPE merges are learned inside words, whitespace runs, and punctuation runs;
    merges never cross a lexical boundary. This keeps the implementation easy
    to inspect while reducing long character sequences into word/subword ids.
    """

    SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
    _piece_pattern = re.compile(r"\s+|[A-Za-z]+|[^A-Za-z\s]")

    def __init__(self, vocabulary: Iterable[str], merges: Iterable[tuple[str, str]]) -> None:
        base = list(vocabulary)
        merge_list = [(str(left), str(right)) for left, right in merges]
        tokens = list(self.SPECIAL_TOKENS) + base
        for left, right in merge_list:
            merged = left + right
            if merged not in tokens:
                tokens.append(merged)
        if len(set(tokens)) != len(tokens):
            raise ValueError("BPE vocabulary contains duplicate tokens")
        self.id_to_token = tokens
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.merges = merge_list

    @classmethod
    def from_text(
        cls,
        text: str,
        vocab_size: int = 512,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        if vocab_size <= len(cls.SPECIAL_TOKENS):
            raise ValueError("vocab_size must leave room for ordinary tokens")
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        pieces = cls._piece_pattern.findall(text)
        if not pieces:
            raise ValueError("cannot train BPE on empty text")
        piece_counts = Counter(pieces)
        sequences = {piece: tuple(piece) for piece in piece_counts}
        base_tokens = sorted({character for piece in pieces for character in piece})
        token_set = set(base_tokens)
        merges: list[tuple[str, str]] = []
        target_merges = max(0, vocab_size - len(cls.SPECIAL_TOKENS) - len(base_tokens))

        for _ in range(target_merges):
            pair_counts: Counter[tuple[str, str]] = Counter()
            for piece, frequency in piece_counts.items():
                sequence = sequences[piece]
                for index in range(len(sequence) - 1):
                    pair_counts[(sequence[index], sequence[index + 1])] += frequency
            if not pair_counts:
                break
            best_pair, best_count = min(
                pair_counts.items(), key=lambda item: (-item[1], item[0])
            )
            if best_count < min_frequency:
                break
            merged = best_pair[0] + best_pair[1]
            if merged in token_set:
                break
            merges.append(best_pair)
            token_set.add(merged)
            for piece, sequence in sequences.items():
                sequences[piece] = cls._merge_sequence(sequence, best_pair)

        return cls(base_tokens, merges)

    @staticmethod
    def _merge_sequence(
        sequence: tuple[str, ...], pair: tuple[str, str]
    ) -> tuple[str, ...]:
        merged: list[str] = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                merged.append(pair[0] + pair[1])
                index += 2
            else:
                merged.append(sequence[index])
                index += 1
        return tuple(merged)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    def _encode_piece(self, piece: str) -> list[str]:
        sequence = tuple(piece)
        for pair in self.merges:
            sequence = self._merge_sequence(sequence, pair)
        return list(sequence)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens: list[str] = []
        for piece in self._piece_pattern.findall(text):
            tokens.extend(self._encode_piece(piece))
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        decoded: list[str] = []
        for index in ids:
            index = int(index)
            if not 0 <= index < self.vocab_size:
                raise ValueError(f"Token id out of range: {index}")
            token = self.id_to_token[index]
            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue
            decoded.append(token)
        return "".join(decoded)

    def state_dict(self) -> dict[str, object]:
        return {
            "type": "bpe",
            "id_to_token": self.id_to_token,
            "merges": [list(pair) for pair in self.merges],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "BPETokenizer":
        tokens = state.get("id_to_token")
        merges = state.get("merges")
        if not isinstance(tokens, list) or tokens[:4] != list(cls.SPECIAL_TOKENS):
            raise ValueError("Invalid BPE tokenizer state")
        if not isinstance(merges, list):
            raise ValueError("BPE tokenizer state is missing merges")
        merge_pairs: list[tuple[str, str]] = []
        for pair in merges:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Invalid BPE merge entry")
            merge_pairs.append((str(pair[0]), str(pair[1])))
        tokenizer = cls(tokens[4:], merge_pairs)
        if tokenizer.id_to_token != tokens:
            raise ValueError("BPE tokenizer state contains inconsistent tokens")
        return tokenizer

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.state_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))
