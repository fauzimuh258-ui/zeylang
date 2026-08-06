"""
ZeyLang — Programming Language for AI, Robot, Space
Part 3a: Interactive REPL (pure Python, built on lexer.py / parser.py / visitor.py)
Part 4: adds multi-line prompt input for long ai@prompt() text, using a
Python-style triple-double-quote block — see _read_statement.
Part 7: namespace calls use `@` now (ai@chat(), not ai.chat()) — this
file only needed its example text updated (help text, docstrings); the
REPL itself doesn't special-case `.` or `@` anywhere, it just tokenizes
and parses whatever the user typed via lexer.py/parser.py.

Reads a statement (possibly spanning several lines if braces are still
open, or if a triple-quote raw-text block is open), executes it against
a persistent Interpreter, and auto-displays the value of a trailing bare
expression — e.g. typing `2 + 2` shows `=> 4`, matching how most
language REPLs behave.
"""

from lexer import Lexer, LexerError, TokenType
from parser import Parser, ParserError, ExprStmt
from visitor import Interpreter, ZeyRuntimeError

PROMPT = "ZeyLang> "
CONTINUE_PROMPT = "...      "
RAW_PROMPT = '"""...   '

TRIPLE = '"""'

HELP_TEXT = """\
Perintah:
  help   - tampilkan bantuan ini
  exit   - keluar dari REPL (atau quit, atau Ctrl-D)

Ketik kode ZeyLang lalu Enter untuk langsung dieksekusi.
Blok dengan '{' yang belum ditutup akan lanjut ke baris berikutnya
sampai '}' cocok. Tekan Ctrl-C untuk membatalkan input yang sedang
diketik dan kembali ke prompt.

Untuk prompt AI yang panjang/multi-baris, bungkus dengan \"\"\" — REPL
akan menerima teks apa adanya baris demi baris sampai \"\"\" penutup,
lalu otomatis menjadikannya satu string literal ZeyLang:
  ZeyLang> ai@prompt(\"\"\"
  \"\"\"...   Buat kode Python untuk kalkulator
  \"\"\"...   dengan tambah, kurang, kali, bagi.
  \"\"\"...   \"\"\")

Contoh lain:
  ZeyLang> ai@chat("Halo dunia!")
  ZeyLang> x = 5
  ZeyLang> if x > 3 {
  ...          robot@walk(x)
  ...      }
"""


def _escape_for_zeylang_string(text: str) -> str:
    """Turns raw, possibly multi-line text into the body of a ZeyLang
    double-quoted string literal (real newlines -> \\n, etc.), so the
    result is ordinary single-physical-line ZeyLang the existing
    Lexer/Parser already handle — no grammar changes needed."""
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "")
    )
    return f'"{escaped}"'


def _is_balanced(tokens) -> bool:
    depth = 0
    for tok in tokens:
        if tok.type == TokenType.LBRACE:
            depth += 1
        elif tok.type == TokenType.RBRACE:
            depth -= 1
    return depth <= 0


def _read_statement() -> str | None:
    """Reads lines until braces balance. Returns the source text, "" for
    a blank/aborted entry, or None on EOF (Ctrl-D).

    A \"\"\" starts a raw multi-line block (handy for long ai@prompt()
    text): lines are captured verbatim, with no lexing or brace-balance
    checks, until a closing \"\"\" — then the captured text is escaped
    into a normal ZeyLang string literal and spliced back into the
    buffer, and ordinary brace-balance continuation resumes."""
    buffer = ""
    prompt = PROMPT
    raw_mode = False
    raw_lines: list[str] = []

    while True:
        try:
            line = input(prompt)
        except EOFError:
            return None

        if raw_mode:
            if TRIPLE in line:
                before, after = line.split(TRIPLE, 1)
                raw_lines.append(before)
                buffer += _escape_for_zeylang_string("\n".join(raw_lines)) + after + "\n"
                raw_mode = False
                raw_lines = []
            else:
                raw_lines.append(line)
                prompt = RAW_PROMPT
                continue
        elif TRIPLE in line:
            before, rest = line.split(TRIPLE, 1)
            if TRIPLE in rest:
                inner, after = rest.split(TRIPLE, 1)
                buffer += before + _escape_for_zeylang_string(inner) + after + "\n"
            else:
                buffer += before
                raw_mode = True
                raw_lines = [rest]
                prompt = RAW_PROMPT
                continue
        else:
            buffer += line + "\n"

        if not buffer.strip():
            return ""
        try:
            tokens = Lexer(buffer).tokenize()
        except LexerError as e:
            print(e)
            return ""
        if _is_balanced(tokens):
            return buffer
        prompt = CONTINUE_PROMPT


def _run_and_display(interpreter: Interpreter, program) -> None:
    """Executes every statement normally, except a trailing bare
    expression, whose value is auto-displayed if it isn't None —
    mirrors Python's own REPL. Reuses Interpreter's internals directly
    since this file lives alongside visitor.py in the same project."""
    statements = program.statements
    for i, stmt in enumerate(statements):
        if i == len(statements) - 1 and isinstance(stmt, ExprStmt):
            value = interpreter._eval(stmt.expr, interpreter.globals)
            if value is not None:
                print(f"=> {value!r}")
        else:
            interpreter._exec(stmt, interpreter.globals)


def main() -> None:
    print("ZeyLang REPL — ketik 'help' untuk bantuan, 'exit' untuk keluar.")
    interpreter = Interpreter()

    while True:
        try:
            source = _read_statement()
        except KeyboardInterrupt:
            print()  # move past the ^C
            continue

        if source is None:  # Ctrl-D
            print("exit")
            break

        stripped = source.strip()
        if not stripped:
            continue
        if stripped.lower() in ("exit", "quit"):
            break
        if stripped.lower() == "help":
            print(HELP_TEXT)
            continue

        try:
            tokens = Lexer(source).tokenize()
            program = Parser(tokens).parse()
            _run_and_display(interpreter, program)
        except LexerError as e:
            print(e)
        except ParserError as e:
            print(e)
        except ZeyRuntimeError as e:
            print(e)


if __name__ == "__main__":
    main()
