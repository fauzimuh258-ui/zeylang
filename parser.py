"""
ZeyLang — Programming Language for AI, Robot, Space
Part 2b: AST nodes + recursive-descent Parser (pure Python, no ANTLR runtime)
Part 6: adds read-only list indexing (data[0], data[-1]) — see Index
below and the LBRACK branch in _call_expr. Added because Part 5/6
introduced several functions that return lists (robot@sensor("gyro"),
space@navigate(), space@telemetry()); without this, `[` after a value
wasn't recognized at all, so `x[0]` silently mis-parsed as two
unrelated statements (`x` alone, then a separate stray `[0]` list
literal) instead of failing loudly — worth fixing outright rather
than leaving as a silent trap. Write-indexing (data[0] = 5) is not
supported, since nothing so far needs it.
Part 7: namespace calls use `@` instead of `.` (ai@chat(), robot@arm(),
...) — _call_expr now matches TokenType.AT where it used to match DOT.
The AST node is still named Attribute (an internal detail — nothing
about it is specific to which symbol triggered it).

Structure mirrors ZeyLang.g4 (Part 1) rule-for-rule: program -> statement
-> simpleStmt/ifStmt/whileStmt/forStmt/funcDef, and the expr precedence
chain or -> and -> not -> comparison -> add -> mul -> unary -> call -> atom.
"""

from dataclasses import dataclass, field
from lexer import Token, TokenType


# ===================== AST NODES =====================

@dataclass
class Program:
    statements: list


# -- statements --

@dataclass
class Assign:
    name: str
    value: object


@dataclass
class Return:
    value: object = None


@dataclass
class Break:
    pass


@dataclass
class Continue:
    pass


@dataclass
class Pass:
    pass


@dataclass
class ExprStmt:
    expr: object


@dataclass
class If:
    branches: list          # [(condition, body), ...] — first is `if`, rest are `elif`
    else_body: list = None  # None if there's no `else`


@dataclass
class While:
    condition: object
    body: list


@dataclass
class For:
    var_name: str
    iterable: object
    body: list


@dataclass
class FuncDef:
    name: str
    params: list
    body: list


# -- expressions --

@dataclass
class Literal:
    value: object   # Python str / int / float / bool / None


@dataclass
class Identifier:
    name: str


@dataclass
class ListLiteral:
    elements: list


@dataclass
class BinOp:
    op: str
    left: object
    right: object


@dataclass
class UnaryOp:
    op: str
    operand: object


@dataclass
class Attribute:
    obj: object
    name: str


@dataclass
class Call:
    callee: object
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)   # name -> expr, e.g. ai@config(temp=0.7)


@dataclass
class Index:
    obj: object
    index: object   # e.g. data[0]


# Tokens that can legally start an expression — used to tell
# `return` (no value) apart from `return <expr>` without a
# statement terminator to lean on.
_EXPR_START = frozenset({
    TokenType.NUMBER, TokenType.STRING, TokenType.BOOL, TokenType.NONE,
    TokenType.IDENTIFIER, TokenType.LPAREN, TokenType.LBRACK,
    TokenType.MINUS, TokenType.NOT,
})


class ParserError(Exception):
    def __init__(self, message, token: Token):
        super().__init__(
            f"[Parser] Line {token.line}:{token.column} — {message} "
            f"(got {token.type.name} {token.value!r})"
        )


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    # ---- token helpers ----
    def _peek(self):
        return self.tokens[self.pos]

    def _advance(self):
        tok = self.tokens[self.pos]
        if tok.type != TokenType.EOF:
            self.pos += 1
        return tok

    def _check(self, *types):
        return self._peek().type in types

    def _match(self, *types):
        if self._check(*types):
            return self._advance()
        return None

    def _expect(self, ttype, message):
        if self._check(ttype):
            return self._advance()
        raise ParserError(message, self._peek())

    # ---- entry point ----
    def parse(self):
        statements = []
        while not self._check(TokenType.EOF):
            statements.append(self._statement())
        return Program(statements)

    # ---- statements ----
    def _statement(self):
        if self._check(TokenType.IF):
            return self._if_stmt()
        if self._check(TokenType.WHILE):
            return self._while_stmt()
        if self._check(TokenType.FOR):
            return self._for_stmt()
        if self._check(TokenType.DEF):
            return self._funcdef()
        return self._simple_stmt()

    def _simple_stmt(self):
        if self._check(TokenType.RETURN):
            self._advance()
            value = self._expr() if self._peek().type in _EXPR_START else None
            return Return(value)
        if self._match(TokenType.BREAK):
            return Break()
        if self._match(TokenType.CONTINUE):
            return Continue()
        if self._match(TokenType.PASS):
            return Pass()
        if self._check(TokenType.IDENTIFIER) and self.tokens[self.pos + 1].type == TokenType.ASSIGN:
            name = self._advance().value
            self._advance()  # consume '='
            return Assign(name, self._expr())
        return ExprStmt(self._expr())

    def _block(self):
        self._expect(TokenType.LBRACE, "expected '{'")
        statements = []
        while not self._check(TokenType.RBRACE):
            statements.append(self._statement())
        self._expect(TokenType.RBRACE, "expected '}'")
        return statements

    def _if_stmt(self):
        self._advance()  # 'if'
        branches = [(self._expr(), self._block())]
        while self._check(TokenType.ELIF):
            self._advance()
            branches.append((self._expr(), self._block()))
        else_body = None
        if self._match(TokenType.ELSE):
            else_body = self._block()
        return If(branches, else_body)

    def _while_stmt(self):
        self._advance()  # 'while'
        condition = self._expr()
        return While(condition, self._block())

    def _for_stmt(self):
        self._advance()  # 'for'
        name = self._expect(TokenType.IDENTIFIER, "expected loop variable").value
        self._expect(TokenType.IN, "expected 'in'")
        iterable = self._expr()
        return For(name, iterable, self._block())

    def _funcdef(self):
        self._advance()  # 'def'
        name = self._expect(TokenType.IDENTIFIER, "expected function name").value
        self._expect(TokenType.LPAREN, "expected '('")
        params = []
        if not self._check(TokenType.RPAREN):
            params.append(self._expect(TokenType.IDENTIFIER, "expected parameter name").value)
            while self._match(TokenType.COMMA):
                params.append(self._expect(TokenType.IDENTIFIER, "expected parameter name").value)
        self._expect(TokenType.RPAREN, "expected ')'")
        return FuncDef(name, params, self._block())

    # ---- expressions (precedence climbing, mirrors ZeyLang.g4) ----
    def _expr(self):
        return self._or_expr()

    def _or_expr(self):
        left = self._and_expr()
        while self._match(TokenType.OR):
            left = BinOp("or", left, self._and_expr())
        return left

    def _and_expr(self):
        left = self._not_expr()
        while self._match(TokenType.AND):
            left = BinOp("and", left, self._not_expr())
        return left

    def _not_expr(self):
        if self._match(TokenType.NOT):
            return UnaryOp("not", self._not_expr())
        return self._comp_expr()

    def _comp_expr(self):
        left = self._add_expr()
        while self._check(TokenType.EQ, TokenType.NEQ, TokenType.LE, TokenType.GE, TokenType.LT, TokenType.GT):
            op = self._advance().value
            left = BinOp(op, left, self._add_expr())
        return left

    def _add_expr(self):
        left = self._mul_expr()
        while self._check(TokenType.PLUS, TokenType.MINUS):
            op = self._advance().value
            left = BinOp(op, left, self._mul_expr())
        return left

    def _mul_expr(self):
        left = self._unary_expr()
        while self._check(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self._advance().value
            left = BinOp(op, left, self._unary_expr())
        return left

    def _unary_expr(self):
        if self._check(TokenType.MINUS, TokenType.NOT):
            op = self._advance().value
            return UnaryOp(op, self._unary_expr())
        return self._call_expr()

    def _call_expr(self):
        expr = self._atom()
        while True:
            if self._match(TokenType.AT):
                name = self._expect(TokenType.IDENTIFIER, "expected attribute name after '@'").value
                expr = Attribute(expr, name)
                if self._check(TokenType.LPAREN):
                    args, kwargs = self._call_args()
                    expr = Call(expr, args, kwargs)
            elif self._check(TokenType.LPAREN):
                args, kwargs = self._call_args()
                expr = Call(expr, args, kwargs)
            elif self._match(TokenType.LBRACK):
                index = self._expr()
                self._expect(TokenType.RBRACK, "expected ']'")
                expr = Index(expr, index)
            else:
                break
        return expr

    def _call_args(self):
        """Parses a parenthesized argument list where each argument is
        either a plain `expr` (positional) or `IDENTIFIER = expr`
        (keyword, e.g. ai@config(temp=0.7)). Returns (args, kwargs)."""
        self._advance()  # '('
        args, kwargs = [], {}
        if not self._check(TokenType.RPAREN):
            self._one_arg(args, kwargs)
            while self._match(TokenType.COMMA):
                self._one_arg(args, kwargs)
        self._expect(TokenType.RPAREN, "expected ')'")
        return args, kwargs

    def _one_arg(self, args, kwargs):
        if self._check(TokenType.IDENTIFIER) and self.tokens[self.pos + 1].type == TokenType.ASSIGN:
            name = self._advance().value
            self._advance()  # '='
            kwargs[name] = self._expr()
        else:
            args.append(self._expr())

    def _atom(self):
        tok = self._peek()

        if tok.type == TokenType.NUMBER:
            self._advance()
            return Literal(float(tok.value) if "." in tok.value else int(tok.value))

        if tok.type == TokenType.STRING:
            self._advance()
            return Literal(tok.value)

        if tok.type == TokenType.BOOL:
            self._advance()
            return Literal(tok.value == "true")

        if tok.type == TokenType.NONE:
            self._advance()
            return Literal(None)

        if tok.type == TokenType.IDENTIFIER:
            self._advance()
            return Identifier(tok.value)

        if tok.type == TokenType.LPAREN:
            self._advance()
            expr = self._expr()
            self._expect(TokenType.RPAREN, "expected ')'")
            return expr

        if tok.type == TokenType.LBRACK:
            self._advance()
            elements = []
            if not self._check(TokenType.RBRACK):
                elements.append(self._expr())
                while self._match(TokenType.COMMA):
                    elements.append(self._expr())
            self._expect(TokenType.RBRACK, "expected ']'")
            return ListLiteral(elements)

        raise ParserError("unexpected token in expression", tok)
