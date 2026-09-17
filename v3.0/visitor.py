"""
ZeyLang — Programming Language for AI, Robot, Space
Interpreter (pure Python). @ namespace calls, robot/space libraries,
real AI via Zey AI gateway, string indexing/len/error, file I/O +
process exec + argv (v3.0) — see zeylang_compiler.py's C_IO_RUNTIME
note for why: a self-hosted compiler needs to read a source file,
write generated C, and shell out to gcc.

SECURITY NOTE: the AI API key is never hardcoded — default from
ZEY_AI_API_KEY env var, override via ai@config(key="...").
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
        self.vars[name] = value

    def assign(self, name, value):
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
            raise ZeyRuntimeError(f"'{self.node.name}' expects {len(self.node.params)} arg(s), got {len(args)}")
        local = Environment(self.closure)
        for name, value in zip(self.node.params, args):
            local.set(name, value)
        try:
            interpreter._exec_block(self.node.body, local)
        except ReturnSignal as signal:
            return signal.value
        return None

    def __repr__(self):
        return f"<function {self.node.name}>"


_AI_ENDPOINT = os.environ.get("ZEY_AI_ENDPOINT", "https://zey-ai.vercel.app/api/chat")
_AI_DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _extract_ai_text(raw_json: str) -> str:
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
        print(f"[ai@chat] {message}")
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
            raise ZeyRuntimeError(f"ai@config: unknown option(s) {sorted(unknown)}")
        if "key" in kwargs:
            self._api_key = kwargs["key"]
        if "temp" in kwargs:
            self._temperature = float(kwargs["temp"])
        if "max" in kwargs:
            self._max_tokens = int(kwargs["max"])
        return None

    def prompt(self, text):
        if not self._api_key:
            raise ZeyRuntimeError("no AI API key set — call ai@config(key=\"...\") or set ZEY_AI_API_KEY")
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": text})
        payload = json.dumps({
            "model": self._model, "messages": messages,
            "temperature": self._temperature, "max_tokens": self._max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            _AI_ENDPOINT, data=payload, method="POST",
            headers={"Content-Type": "application/json", "x-api-key": self._api_key},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise ZeyRuntimeError(f"AI request failed: HTTP {e.code} — {e.read().decode('utf-8', 'ignore')}")
        except urllib.error.URLError as e:
            raise ZeyRuntimeError(f"AI request failed: {e.reason}")
        return _extract_ai_text(raw)


class _Robot:
    _SENSOR_RANGES = {"temp": (20.0, 30.0), "dist": (5.0, 200.0)}

    def walk(self, steps):
        print(f"[robot@walk] walking {steps} step(s)")
        return steps

    def arm(self, servo, angle):
        print(f"[robot@arm] servo {servo} -> {angle}\u00b0")
        return angle

    def grip(self, state):
        if state not in ("open", "close"):
            raise ZeyRuntimeError(f'robot@grip(...) expects "open" or "close", got {state!r}')
        print(f"[robot@grip] gripper -> {state}")
        return state

    def sensor(self, kind):
        if kind == "gyro":
            return [round(random.uniform(-5.0, 5.0), 2) for _ in range(3)]
        if kind in self._SENSOR_RANGES:
            lo, hi = self._SENSOR_RANGES[kind]
            return round(random.uniform(lo, hi), 2)
        raise ZeyRuntimeError(f'robot@sensor(...): unknown sensor type {kind!r}')

    def speak(self, text):
        print(f"[robot@speak] {text}")
        return text


class _Space:
    def orbit(self, altitude, unit="km"):
        print(f"[space@orbit] orbiting at {altitude}{unit}")
        return altitude

    def launch(self, payload, target):
        print(f"[space@launch] {payload} -> {target}")
        return payload

    def thrust(self, percent):
        if not (0 <= percent <= 100):
            raise ZeyRuntimeError(f"space@thrust(...) expects 0-100, got {percent!r}")
        print(f"[space@thrust] engine -> {percent}%")
        return percent

    def navigate(self, x, y, z):
        print(f"[space@navigate] heading to ({x}, {y}, {z})")
        return [x, y, z]

    def telemetry(self):
        return [
            round(random.uniform(20000.0, 28000.0), 2), round(random.uniform(300.0, 450.0), 2),
            round(random.uniform(10.0, 95.0), 2), round(random.uniform(-20.0, 60.0), 2),
            [round(random.uniform(-1000.0, 1000.0), 2) for _ in range(3)],
        ]

    def touchdown(self, coord):
        print(f"[space@touchdown] touchdown at {coord}")
        return coord


def _zey_error_builtin(msg):
    raise ZeyRuntimeError(str(msg))


def _zey_read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise ZeyRuntimeError(f"read_file: {e.strerror or e}")


def _zey_write_file(path, content):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        raise ZeyRuntimeError(f"write_file: {e.strerror or e}")
    return None


def _zey_run_command(cmd):
    import subprocess as _sp
    return _sp.run(cmd, shell=True).returncode


def _zey_argv(i):
    import sys as _sys
    i = int(i)
    return _sys.argv[i] if 0 <= i < len(_sys.argv) else ""


def _make_globals():
    env = Environment()
    env.set("ai", _AI())
    env.set("robot", _Robot())
    env.set("space", _Space())
    env.set("range", range)
    env.set("print", print)
    env.set("len", len)
    env.set("error", _zey_error_builtin)
    env.set("read_file", _zey_read_file)
    env.set("write_file", _zey_write_file)
    env.set("run_command", _zey_run_command)
    env.set("argv", _zey_argv)
    return env


_BIN_OPS = {
    "+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
    "/": lambda a, b: a / b, "%": lambda a, b: a % b,
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b, ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
}


class Interpreter:
    def __init__(self):
        self.globals = _make_globals()

    def run(self, program):
        self._exec_block(program.statements, self.globals)

    def _truthy(self, value):
        return bool(value)

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
                self._exec_block(body, env)
                return
        if node.else_body is not None:
            self._exec_block(node.else_body, env)

    def _exec_While(self, node: While, env):
        while self._truthy(self._eval(node.condition, env)):
            try:
                self._exec_block(node.body, env)
            except BreakSignal:
                break
            except ContinueSignal:
                continue

    def _exec_For(self, node: For, env):
        for item in self._eval(node.iterable, env):
            env.set(node.var_name, item)
            try:
                self._exec_block(node.body, env)
            except BreakSignal:
                break
            except ContinueSignal:
                continue

    def _exec_FuncDef(self, node: FuncDef, env):
        env.set(node.name, ZeyFunction(node, env))

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
        if not isinstance(obj, (list, str)):
            raise ZeyRuntimeError(f"'{type(obj).__name__}' is not subscriptable")
        if not isinstance(index, (int, float)) or isinstance(index, bool):
            raise ZeyRuntimeError(f"index must be a number, got {type(index).__name__}")
        try:
            return obj[int(index)]
        except IndexError:
            raise ZeyRuntimeError(f"index {int(index)} out of range (length {len(obj)})")

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
