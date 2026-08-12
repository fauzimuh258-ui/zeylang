"""
ZeyLang — Programming Language for AI, Robot, Space
Part 2a: Lexer (pure Python, no ANTLR runtime)

Turns ZeyLang source text into a flat list of Token objects for
the recursive-descent Parser (parser.py). Mirrors the tokens
defined in ZeyLang.g4 (Part 1) exactly.

Namespace method calls use `@`, not `.` (ai@chat(), robot@arm(), ...) —
AT is the only symbol for this now; `.` is no longer a valid standalone
token (it still works fine inside a number literal, e.g. 3.14 — that's
handled inside _read_number, independent of the AT/removed-DOT symbol).
"""

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    # Keywords
    DEF = auto()
    IF = auto()
    ELIF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()
    PASS = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    BOOL = auto()
    NONE = auto()

    # Literals / names
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()

    # Operators
    ASSIGN = auto()
    EQ = auto()
    NEQ = auto()
    LE = auto()
    GE = auto()
    LT = auto()
    GT = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()

    # Symbols
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACK = auto()
    RBRACK = auto()
    COMMA = auto()
    AT = auto()

    EOF = auto()


KEYWORDS = {
    "def": TokenType.DEF,
    "if": TokenType.IF,
    "elif": TokenType.ELIF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "for": TokenType.FOR,
    "in": TokenType.IN,
    "return": TokenType.RETURN,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "pass": TokenType.PASS,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
    "true": TokenType.BOOL,
    "false": TokenType.BOOL,
    "none": TokenType.NONE,
}

# Order matters: longer symbols must be checked before their single-char prefix
# ("==" before "=", "<=" before "<", etc.) since this lexer has no automatic
# longest-match like ANTLR4 — we do it manually here.
SYMBOLS = (
    ("==", TokenType.EQ),
    ("!=", TokenType.NEQ),
    ("<=", TokenType.LE),
    (">=", TokenType.GE),
    ("<", TokenType.LT),
    (">", TokenType.GT),
    ("=", TokenType.ASSIGN),
    ("+", TokenType.PLUS),
    ("-", TokenType.MINUS),
    ("*", TokenType.STAR),
    ("/", TokenType.SLASH),
    ("%", TokenType.PERCENT),
    ("(", TokenType.LPAREN),
    (")", TokenType.RPAREN),
    ("{", TokenType.LBRACE),
    ("}", TokenType.RBRACE),
    ("[", TokenType.LBRACK),
    ("]", TokenType.RBRACK),
    (",", TokenType.COMMA),
    ("@", TokenType.AT),
)

ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", '"': '"', "'": "'", "\\": "\\"}


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"


class LexerError(Exception):
    def __init__(self, message, line, column):
        super().__init__(f"[Lexer] Line {line}:{column} — {message}")
        self.line = line
        self.column = column


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.length = len(source)

    def _peek(self, offset=0):
        i = self.pos + offset
        return self.source[i] if i < self.length else ""

    def _advance(self):
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def _skip_ignored(self):
        while self.pos < self.length:
            ch = self._peek()
            if ch in " \t\r\n":
                self._advance()
            elif ch == "#":
                while self.pos < self.length and self._peek() != "\n":
                    self._advance()
            else:
                break

    def _read_string(self):
        quote = self._advance()  # consume opening quote
        start_line, start_col = self.line, self.column
        chars = []
        while True:
            if self.pos >= self.length:
                raise LexerError("unterminated string", start_line, start_col)
            ch = self._advance()
            if ch == quote:
                break
            if ch == "\n":
                raise LexerError("unterminated string", start_line, start_col)
            if ch == "\\":
                esc = self._advance()
                chars.append(ESCAPES.get(esc, esc))
            else:
                chars.append(ch)
        return "".join(chars)

    def _read_number(self):
        start = self.pos
        while self._peek().isdigit():
            self._advance()
        if self._peek() == "." and self._peek(1).isdigit():
            self._advance()
            while self._peek().isdigit():
                self._advance()
        return self.source[start:self.pos]

    def _read_identifier(self):
        start = self.pos
        while self._peek().isalnum() or self._peek() == "_":
            self._advance()
        return self.source[start:self.pos]

    def tokenize(self):
        tokens = []
        while True:
            self._skip_ignored()
            if self.pos >= self.length:
                tokens.append(Token(TokenType.EOF, "", self.line, self.column))
                break

            line, col = self.line, self.column
            ch = self._peek()

            if ch in "\"'":
                value = self._read_string()
                tokens.append(Token(TokenType.STRING, value, line, col))
                continue

            if ch.isdigit():
                value = self._read_number()
                tokens.append(Token(TokenType.NUMBER, value, line, col))
                continue

            if ch.isalpha() or ch == "_":
                value = self._read_identifier()
                ttype = KEYWORDS.get(value, TokenType.IDENTIFIER)
                tokens.append(Token(ttype, value, line, col))
                continue

            matched = False
            for symbol, ttype in SYMBOLS:
                if self.source.startswith(symbol, self.pos):
                    for _ in symbol:
                        self._advance()
                    tokens.append(Token(ttype, symbol, line, col))
                    matched = True
                    break
            if matched:
                continue

            raise LexerError(f"unexpected character {ch!r}", line, col)

        return tokens
