import re
from collections import Counter
from typing import Iterable

PRETOKENIZE_PATTERN = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|\s?[A-Za-z]+|\s?\d+|\s?[^A-Za-z\d\s]+|\s+")


class BPE:
    def __init__(self):
        self.token_to_id: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        self.id_to_token: dict[int, bytes] = {token_id: token for token, token_id in self.token_to_id.items()}
        self.merges: list[tuple[int, int]] = []

    def train(self, corpus: Iterable[str], vocab_size: int, min_frequency: int = 2):
        word_counts = Counter()
        for line in corpus:
            for token in self.pretokenize(line):
                b = token.encode("utf-8")
                if b:
                    word_counts[b] += 1

        word_symbols: dict[bytes, list[int]] = {word: list(word) for word in word_counts}

        while len(self.token_to_id) < vocab_size:
            pair_frequencies = Counter()

            for word, symbols in word_symbols.items():
                word_count = word_counts[word]
                for i in range(len(symbols) - 1):
                    pair_frequencies[symbols[i], symbols[i + 1]] += word_count

            if not pair_frequencies:
                break

            (symbol_left, symbol_right), pair_frequency = pair_frequencies.most_common(1)[0]
            if pair_frequency < min_frequency:
                break

            new_id = len(self.token_to_id)
            new_token = self.id_to_token[symbol_left] + self.id_to_token[symbol_right]

            self.token_to_id[new_token] = new_id
            self.id_to_token[new_id] = new_token
            self.merges.append((symbol_left, symbol_right))

            self._merge(word_symbols, symbol_left, symbol_right, new_id)

    @classmethod
    def pretokenize(cls, text: str) -> list[str]:
        return [m.group(0) for m in PRETOKENIZE_PATTERN.finditer(text)]

    @classmethod
    def _merge(cls, word_symbols: dict[bytes, list[int]], left: bytes, right: bytes, new_id: int) -> None:
        for word, symbols in word_symbols.items():
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == left and symbols[i + 1] == right:
                    new_symbols.append(new_id)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            word_symbols[word] = new_symbols

    def encode(self, text: str) -> list[int]:
        ids = []
        for token in self.pretokenize(text):
            symbols = list(token.encode("utf-8"))
            symbols = self._apply_bpe(symbols)
            ids.extend(symbols)
        return ids

    def _apply_bpe(self, symbols: list[int]) -> list[int]:
        for left, right in self.merges:
            new_token = self.id_to_token[left] + self.id_to_token[right]
            new_id = self.token_to_id[new_token]
            i = 0
            output = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == left and symbols[i + 1] == right:
                    output.append(new_id)
                    i += 2
                else:
                    output.append(symbols[i])
                    i += 1
            symbols = output
        return symbols
