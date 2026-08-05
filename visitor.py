"""
ZeyLang — Programming Language for AI, Robot, Space
Part 2c: Visitor / tree-walking interpreter (pure Python, no ANTLR runtime)
Part 4: adds real AI integration (ai.prompt/model/system/config) via the
Zey AI gateway, using only the stdlib (urllib) — no extra pip install.
Part 5: adds the robot library (arm/grip/sensor/speak) to _Robot. All
simulated (no hardware) — see class _Robot below for details.
Part 6: adds the space library (launch/thrust/navigate/telemetry/land)
to _Space (orbit already existed). Same simulated approach as Part 5.
Also adds _eval_Index (data[0], data[-1]) — see parser.py's Part 6 note
for why: without it, list-returning functions like robot.sensor("gyro")
and space.navigate()/telemetry() had no way to read a single element out.

Walks the AST produced by parser.py and executes it directly.

Scoping matches Python: only function calls open a new scope.
if/while/for run their body in the *same* environment as their
caller, so a variable assigned inside an if-block is still visible
after it, exactly like real Python.

ai / robot / space are plain identifiers pre-bound in the global
environment (see BUILTINS below) — the grammar and parser don't
know about them at all, matching the Part 1 design note.

SECURITY NOTE on the AI API key: it is intentionally NOT hardcoded here.
The default is read from the ZEY_AI_API_KEY environment variable, and a
ZeyLang script can also set it at runtime with ai.config(key="..."). A
literal key baked into source (and, once Part 3's compiler is involved,
into every compiled binary) is easy to leak — env var / config() keeps
the actual secret out of the interpreter/compiler source itself.
"""

import json
import os
import random
import urllib.error
import urllib.request

from parser import (
    Assign, Return, Break, Continue, Pass, ExprStmt, If, While, For, FuncDef,
    Literal, Identifier, ListLiteral, BinOp, UnaryOp, Attribute, Call, Index,
)


class ZeyRuntimeError(Exception):
    def __init__(self, message):
        super().__init__(f"[Runtime] {message}")


class BreakSignal(Exception):
    pass


class ContinueSignal(Exception):
    pass


class ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value


class Environment:
    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent

    def get(self, name):
        if name in self.vars:
            return self.vars[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise ZeyRuntimeError(f"undefined variable '{name}'")

    def set(self, name, value):
        """Define/overwrite `name` directly in THIS scope."""
        self.vars[name] = value

    def assign(self, name, value):
        """`x = value` — reuse an existing binding wherever it lives up
        the chain; if none exists yet, create it in this scope (matches
        Python's implicit-declaration-on-first-assignment behavior)."""
        env = self
        while env is not None:
            if name in env.vars:
                env.vars[name] = value
                return
            env = env.parent
        self.vars[name] = value


class ZeyFunction:
    def __init__(self, node: FuncDef, closure: Environment):
        self.node = node
        self.closure = closure

    def call(self, interpreter, args):
        if len(args) != len(self.node.params):
            raise ZeyRuntimeError(
                f"'{self.node.name}' expects {len(self.node.params)} arg(s), got {len(args)}"
            )
        local = Environment(self.closure)          # only functions open a new scope
        for name, value in zip(self.node.params, args):
            local.set(name, value)
        try:
            interpreter._exec_block(self.node.body, local)
        except ReturnSignal as signal:
            return signal.value
        return None

    def __repr__(self):
        return f"<function {self.node.name}>"


# ===================== Built-ins (ai / robot / space) =====================
# ai.chat / robot.walk / space.orbit are minimal stubs so the Part 1
# example runs end-to-end. ai.prompt/model/system/config (Part 4) call
# the real Zey AI gateway.

_AI_ENDPOINT = os.environ.get("ZEY_AI_ENDPOINT", "https://zey-ai.vercel.app/api/chat")
_AI_DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _extract_ai_text(raw_json: str) -> str:
    """The gateway is assumed to return a Groq/OpenAI-compatible chat
    completions body (the model name in the Part 4 example is a real
    Groq-hosted model, so the gateway most likely proxies Groq), i.e.
    data["choices"][0]["message"]["content"]. Falls back to a couple of
    simpler top-level shapes, then raises with the raw body so a schema
    mismatch is visible and easy to fix here if the real gateway differs."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json

    if isinstance(data, dict) and "error" in data:
        raise ZeyRuntimeError(f"AI gateway returned an error: {data['error']}")
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        pass
    for key in ("response", "message", "content", "text"):
        if isinstance(data.get(key), str):
            return data[key]
    raise ZeyRuntimeError(f"couldn't parse AI response (unrecognized shape): {raw_json[:300]}")


class _AI:
    def __init__(self):
        self._api_key = os.environ.get("ZEY_AI_API_KEY", "")
        self._model = _AI_DEFAULT_MODEL
        self._system_prompt = None
        self._temperature = 0.7
        self._max_tokens = 1024

    def chat(self, message):
        print(f"[ai.chat] {message}")
        return message

    def model(self, name):
        self._model = name
        return name

    def system(self, text):
        self._system_prompt = text
        return text

    def config(self, **kwargs):
        unknown = set(kwargs) - {"key", "temp", "max"}
        if unknown:
            raise ZeyRuntimeError(f"ai.config: unknown option(s) {sorted(unknown)}")
        if "key" in kwargs:
            self._api_key = kwargs["key"]
        if "temp" in kwargs:
            self._temperature = float(kwargs["temp"])
        if "max" in kwargs:
            self._max_tokens = int(kwargs["max"])
        return None

    def prompt(self, text):
        if not self._api_key:
            raise ZeyRuntimeError(
                "no AI API key set — call ai.config(key=\"...\") or set the "
                "ZEY_AI_API_KEY environment variable"
            )
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": text})

        payload = json.dumps({
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }).encode("utf-8")

        req = urllib.request.Request(
            _AI_ENDPOINT,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json", "x-api-key": self._api_key},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            raise ZeyRuntimeError(f"AI request failed: HTTP {e.code} — {body}")
        except urllib.error.URLError as e:
            raise ZeyRuntimeError(f"AI request failed: {e.reason}")

        return _extract_ai_text(raw)


class _Robot:
    """All simulated (no hardware). arm/grip/walk/speak print a
    `[robot.X]` line describing the action, matching ai.chat's style.
    sensor() is a query, not a command, so it stays quiet and just
    returns a value — matching how the Part 5 examples use it
    (`temp = robot.sensor("temp")`, no print shown)."""

    _SENSOR_RANGES = {
        "temp": (20.0, 30.0),   # deg C, plausible ambient reading
        "dist": (5.0, 200.0),   # cm, plausible ultrasonic reading
    }

    def walk(self, steps):
        print(f"[robot.walk] walking {steps} step(s)")
        return steps

    def arm(self, servo, angle):
        print(f"[robot.arm] servo {servo} -> {angle}\u00b0")
        return angle

    def grip(self, state):
        if state not in ("open", "close"):
            raise ZeyRuntimeError(f'robot.grip(...) expects "open" or "close", got {state!r}')
        print(f"[robot.grip] gripper -> {state}")
        return state

    def sensor(self, kind):
        if kind == "gyro":
            return [round(random.uniform(-5.0, 5.0), 2) for _ in range(3)]  # [x, y, z] deg/s
        if kind in self._SENSOR_RANGES:
            lo, hi = self._SENSOR_RANGES[kind]
            return round(random.uniform(lo, hi), 2)
        raise ZeyRuntimeError(
            f'robot.sensor(...): unknown sensor type {kind!r} (expected "temp", "dist", or "gyro")'
        )

    def speak(self, text):
        print(f"[robot.speak] {text}")
        return text


class _Space:
    """All simulated. orbit/launch/thrust/navigate/land print a
    `[space.X]` line; telemetry() is a query, so — like robot.sensor()
    in Part 5 — it stays quiet and just returns a value, matching how
    the Part 6 example uses it (`data = space.telemetry()`, no print
    shown). telemetry() returns a fixed-order 5-element list:
    [speed, altitude, fuel, temp, position], position itself a
    3-element [x, y, z] list — ZeyLang has no dict/map type yet, and
    the Part 6 spec explicitly allows list as an alternative."""

    def orbit(self, altitude, unit="km"):
        print(f"[space.orbit] orbiting at {altitude}{unit}")
        return altitude

    def launch(self, payload, target):
        print(f"[space.launch] {payload} -> {target}")
        return payload

    def thrust(self, percent):
        if not (0 <= percent <= 100):
            raise ZeyRuntimeError(f"space.thrust(...) expects 0-100, got {percent!r}")
        print(f"[space.thrust] engine -> {percent}%")
        return percent

    def navigate(self, x, y, z):
        print(f"[space.navigate] heading to ({x}, {y}, {z})")
        return [x, y, z]

    def telemetry(self):
        return [
            round(random.uniform(20000.0, 28000.0), 2),  # speed, km/h
            round(random.uniform(300.0, 450.0), 2),       # altitude, km
            round(random.uniform(10.0, 95.0), 2),         # fuel, %
            round(random.uniform(-20.0, 60.0), 2),        # temp, C
            [round(random.uniform(-1000.0, 1000.0), 2) for _ in range(3)],  # position [x, y, z]
        ]

    def land(self, coord):
        print(f"[space.land] landing at {coord}")
        return coord


def _make_globals():
    env = Environment()
    env.set("ai", _AI())
    env.set("robot", _Robot())
    env.set("space", _Space())
    env.set("range", range)   # needed for `for i in range(n) { ... }`
    env.set("print", print)   # basic debug output
    return env


_BIN_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "%": lambda a, b: a % b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}


class Interpreter:
    def __init__(self):
        self.globals = _make_globals()

    def run(self, program):
        self._exec_block(program.statements, self.globals)

    def _truthy(self, value):
        return bool(value)

    # ---- statement execution ----
    def _exec_block(self, statements, env):
        for stmt in statements:
            self._exec(stmt, env)

    def _exec(self, node, env):
        method = getattr(self, f"_exec_{type(node).__name__}", None)
        if method is None:
            raise ZeyRuntimeError(f"no executor for statement {type(node).__name__}")
        method(node, env)

    def _exec_Assign(self, node: Assign, env):
        env.assign(node.name, self._eval(node.value, env))

    def _exec_ExprStmt(self, node: ExprStmt, env):
        self._eval(node.expr, env)

    def _exec_Pass(self, node: Pass, env):
        pass

    def _exec_Break(self, node: Break, env):
        raise BreakSignal()

    def _exec_Continue(self, node: Continue, env):
        raise ContinueSignal()

    def _exec_Return(self, node: Return, env):
        raise ReturnSignal(self._eval(node.value, env) if node.value is not None else None)

    def _exec_If(self, node: If, env):
        for condition, body in node.branches:
            if self._truthy(self._eval(condition, env)):
                self._exec_block(body, env)          # same scope as caller
                return
        if node.else_body is not None:
            self._exec_block(node.else_body, env)

    def _exec_While(self, node: While, env):
        while self._truthy(self._eval(node.condition, env)):
            try:
                self._exec_block(node.body, env)      # same scope as caller
            except BreakSignal:
                break
            except ContinueSignal:
                continue

    def _exec_For(self, node: For, env):
        for item in self._eval(node.iterable, env):
            env.set(node.var_name, item)              # same scope as caller
            try:
                self._exec_block(node.body, env)
            except BreakSignal:
                break
            except ContinueSignal:
                continue

    def _exec_FuncDef(self, node: FuncDef, env):
        env.set(node.name, ZeyFunction(node, env))

    # ---- expression evaluation ----
    def _eval(self, node, env):
        method = getattr(self, f"_eval_{type(node).__name__}", None)
        if method is None:
            raise ZeyRuntimeError(f"no evaluator for expression {type(node).__name__}")
        return method(node, env)

    def _eval_Literal(self, node: Literal, env):
        return node.value

    def _eval_Identifier(self, node: Identifier, env):
        return env.get(node.name)

    def _eval_ListLiteral(self, node: ListLiteral, env):
        return [self._eval(el, env) for el in node.elements]

    def _eval_UnaryOp(self, node: UnaryOp, env):
        value = self._eval(node.operand, env)
        if node.op == "-":
            return -value
        if node.op == "not":
            return not self._truthy(value)
        raise ZeyRuntimeError(f"unknown unary operator '{node.op}'")

    def _eval_BinOp(self, node: BinOp, env):
        if node.op == "and":
            left = self._eval(node.left, env)
            return left if not self._truthy(left) else self._eval(node.right, env)
        if node.op == "or":
            left = self._eval(node.left, env)
            return left if self._truthy(left) else self._eval(node.right, env)

        left = self._eval(node.left, env)
        right = self._eval(node.right, env)
        try:
            return _BIN_OPS[node.op](left, right)
        except KeyError:
            raise ZeyRuntimeError(f"unknown operator '{node.op}'")
        except ZeroDivisionError:
            raise ZeyRuntimeError("division by zero")
        except TypeError as e:
            raise ZeyRuntimeError(str(e))

    def _eval_Attribute(self, node: Attribute, env):
        obj = self._eval(node.obj, env)
        if not hasattr(obj, node.name):
            raise ZeyRuntimeError(f"'{type(obj).__name__}' has no attribute '{node.name}'")
        return getattr(obj, node.name)

    def _eval_Index(self, node: Index, env):
        obj = self._eval(node.obj, env)
        index = self._eval(node.index, env)
        if not isinstance(obj, list):
            raise ZeyRuntimeError(f"'{type(obj).__name__}' is not subscriptable")
        if not isinstance(index, (int, float)) or isinstance(index, bool):
            raise ZeyRuntimeError(f"list index must be a number, got {type(index).__name__}")
        try:
            return obj[int(index)]
        except IndexError:
            raise ZeyRuntimeError(f"list index {int(index)} out of range (length {len(obj)})")

    def _eval_Call(self, node: Call, env):
        callee = self._eval(node.callee, env)
        args = [self._eval(arg, env) for arg in node.args]
        kwargs = {name: self._eval(expr, env) for name, expr in node.kwargs.items()}
        if isinstance(callee, ZeyFunction):
            if kwargs:
                raise ZeyRuntimeError("keyword arguments aren't supported when calling ZeyLang functions")
            return callee.call(self, args)
        if callable(callee):
            try:
                return callee(*args, **kwargs)
            except TypeError as e:
                raise ZeyRuntimeError(f"invalid arguments: {e}")
        raise ZeyRuntimeError(f"'{callee}' is not callable")


if __name__ == "__main__":
    from lexer import Lexer

    source = '''
ai.chat("Halo dunia!")
robot.walk(10)
space.orbit(400, "km")
'''
    from parser import Parser
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    Interpreter().run(program)
