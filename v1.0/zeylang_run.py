"""
ZeyLang — Programming Language for AI, Robot, Space
Small CLI entry point to run a .zey file directly with the interpreter
(Lexer -> Parser -> Interpreter) — no REPL, no compiling to C. Mirrors
zeylang_compiler.py's CLI error-handling shape.

Usage: python3 zeylang_run.py program.zey
"""

import sys

from lexer import Lexer, LexerError
from parser import Parser, ParserError
from visitor import Interpreter, ZeyRuntimeError


def run_file(path: str) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
    except OSError as e:
        print(f"[Error] couldn't read '{path}': {e.strerror}", file=sys.stderr)
        sys.exit(1)
    try:
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
        Interpreter().run(program)
    except (LexerError, ParserError, ZeyRuntimeError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python3 zeylang_run.py <file.zey>", file=sys.stderr)
        sys.exit(1)
    run_file(sys.argv[1])


if __name__ == "__main__":
    main()
