"""
ZeyLang — Programming Language for AI, Robot, Space
Part 3b: Compiler, ZeyLang -> C (pure Python, no ANTLR runtime)
Part 4: adds ai@prompt/model/system/config, compiled to C that calls
the Zey AI gateway.
Part 5: adds robot@arm/grip/sensor/speak (robot@walk already existed).
All simulated (no hardware) — unconditionally in C_RUNTIME alongside
walk/orbit, since none of them need anything beyond stdlib (sensor()
uses rand()/srand(), seeded once at the top of every generated main()).
Part 6: adds space@launch/thrust/navigate/telemetry/touchdown (space@orbit
already existed) the same way, plus zey_index() for data[i] (Part 6
also added Index to parser.py — see its module docstring for why:
list-returning functions like sensor("gyro")/navigate()/telemetry()
were otherwise unusable beyond printing or iterating the whole thing).
Part 7: namespace calls use `@` instead of `.` (ai@chat(), not ai.chat())
— the actual token change lives in lexer.py/parser.py; this file's
_compile_call dispatch already worked off ns/method strings pulled from
the parsed AST, so it didn't need structural changes, just every C
string/comment/docstring here that spelled out a call updated to match.
Also renamed space.land() -> space@touchdown() (an actual rename, not
just notation) — see zey_space_touchdown below.

Compiles a well-defined SUBSET of ZeyLang to standalone C, buildable
with GCC (uses GCC's statement-expression extension for `and`/`or`
short-circuit and inline list literals, so build with plain `gcc`,
not `-std=c11`).

Supported: variables, all literal types, arithmetic/comparison/
boolean ops, if/elif/else, while (+break/continue), for-in over
range()/lists (+break/continue), list literals, read-only list
indexing (data[0], data[-1] — Part 6), top-level functions (including
recursion), ai@chat / robot@walk / space@orbit / print, ai@prompt /
ai@model / ai@system / ai@config (Part 4), robot@arm / robot@grip /
robot@sensor / robot@speak (Part 5), and space@launch / space@thrust /
space@navigate / space@telemetry / space@touchdown (Part 6).

NOT supported (raises CompileError, doesn't silently miscompile):
nested function definitions (top-level only — no closures), keyword
arguments anywhere except ai@config, index-assignment (data[0] = x),
and attribute access other than ai/robot/space's methods above.

All ZeyLang values compile to a single tagged C struct, ZeyValue —
see C_RUNTIME below. This mirrors visitor.py's dynamic typing, just
enforced with a tag instead of Python's own type system.

ai@prompt() shells out to the `curl` command-line tool via fork/exec/
pipe rather than linking libcurl — deliberately, so `gcc -o program
program.c` still needs nothing extra: just `curl` present on PATH at
runtime, which is far more commonly available than libcurl's *dev*
headers (this was in fact verified against a local mock server using
exactly this fork/exec/pipe approach — libcurl-dev wasn't even
available to test against here). C_AI_RUNTIME below is only prepended
when a program actually calls one of the four ai@ Part 4 functions,
so programs that don't use them build exactly as they did in Part 3.

SECURITY NOTE on the AI API key: not hardcoded here, same reasoning as
visitor.py — a literal key baked into the compiler would end up
embedded in plaintext in every compiled binary. The default is read
from ZEY_AI_API_KEY at process start; a ZeyLang program can override
it at runtime with ai@config(key="...").
"""

import os
import subprocess
import sys

from lexer import Lexer, LexerError
from parser import (
    Parser, ParserError,
    Assign, Return, Break, Continue, Pass, ExprStmt, If, While, For, FuncDef,
    Literal, Identifier, ListLiteral, BinOp, UnaryOp, Attribute, Call, Index,
)


class CompileError(Exception):
    def __init__(self, message):
        super().__init__(f"[Compiler] {message}")


def _c_string_literal(s: str) -> str:
    out = ['"']
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 32:
            out.append(f"\\x{ord(ch):02x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _collect_locals(statements, exclude):
    """Every name ever assigned in `statements`, walking into if/while/for
    bodies (which share scope, like Python) but NOT into nested funcdefs
    (unsupported by this compiler — top-level functions only)."""
    names, seen = [], set(exclude)

    def visit(stmt):
        if isinstance(stmt, Assign):
            if stmt.name not in seen:
                seen.add(stmt.name)
                names.append(stmt.name)
        elif isinstance(stmt, For):
            if stmt.var_name not in seen:
                seen.add(stmt.var_name)
                names.append(stmt.var_name)
            for s in stmt.body:
                visit(s)
        elif isinstance(stmt, If):
            for _, body in stmt.branches:
                for s in body:
                    visit(s)
            if stmt.else_body:
                for s in stmt.else_body:
                    visit(s)
        elif isinstance(stmt, While):
            for s in stmt.body:
                visit(s)
        elif isinstance(stmt, FuncDef):
            raise CompileError(
                f"nested function '{stmt.name}' is not supported — "
                "the compiler only supports top-level functions"
            )
        # ExprStmt / Return / Break / Continue / Pass: nothing to collect

    for s in statements:
        visit(s)
    return names


# ===================== Fixed C runtime, prepended to every build =====================

C_RUNTIME = r"""// ==== ZeyLang runtime — generated by zeylang_compiler.py, do not edit ====
// Requires GCC (uses the statement-expression extension). Build with:
//   gcc -o program program.c
// (plain `gcc`, not `-std=c11`, which rejects GNU extensions)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef enum { ZEY_NONE, ZEY_BOOL, ZEY_NUM, ZEY_STR, ZEY_LIST } ZeyTag;

typedef struct ZeyValue ZeyValue;

typedef struct ZeyList {
    ZeyValue *items;
    int count;
    int capacity;
} ZeyList;

struct ZeyValue {
    ZeyTag tag;
    union {
        int b;
        double n;
        const char *s;
        ZeyList *list;
    } as;
};

static ZeyValue zey_none(void) { ZeyValue v; v.tag = ZEY_NONE; return v; }
static ZeyValue zey_bool(int b) { ZeyValue v; v.tag = ZEY_BOOL; v.as.b = b; return v; }
static ZeyValue zey_num(double n) { ZeyValue v; v.tag = ZEY_NUM; v.as.n = n; return v; }
static ZeyValue zey_str(const char *s) { ZeyValue v; v.tag = ZEY_STR; v.as.s = s; return v; }

static ZeyValue zey_list_new(void) {
    ZeyValue v; v.tag = ZEY_LIST;
    v.as.list = (ZeyList*)malloc(sizeof(ZeyList));
    v.as.list->items = NULL;
    v.as.list->count = 0;
    v.as.list->capacity = 0;
    return v;
}

static void zey_list_push(ZeyValue listval, ZeyValue item) {
    ZeyList *l = listval.as.list;
    if (l->count >= l->capacity) {
        l->capacity = l->capacity ? l->capacity * 2 : 4;
        l->items = (ZeyValue*)realloc(l->items, sizeof(ZeyValue) * l->capacity);
    }
    l->items[l->count++] = item;
}

static void zey_error(const char *msg) {
    fprintf(stderr, "[Runtime] %s\n", msg);
    exit(1);
}

/* Part 6: data[i] — negative indices count from the end, like Python. */
static ZeyValue zey_index(ZeyValue obj, ZeyValue index) {
    if (obj.tag != ZEY_LIST) zey_error("value is not subscriptable");
    if (index.tag != ZEY_NUM) zey_error("list index must be a number");
    int i = (int)index.as.n;
    if (i < 0) i += obj.as.list->count;
    if (i < 0 || i >= obj.as.list->count) zey_error("list index out of range");
    return obj.as.list->items[i];
}

static int zey_truthy(ZeyValue v) {
    switch (v.tag) {
        case ZEY_NONE: return 0;
        case ZEY_BOOL: return v.as.b;
        case ZEY_NUM:  return v.as.n != 0;
        case ZEY_STR:  return v.as.s[0] != '\0';
        case ZEY_LIST: return v.as.list->count > 0;
    }
    return 0;
}

static void zey_print_value(ZeyValue v) {
    switch (v.tag) {
        case ZEY_NONE: printf("None"); break;
        case ZEY_BOOL: printf(v.as.b ? "True" : "False"); break;
        case ZEY_NUM:
            if (v.as.n == (long long)v.as.n) printf("%lld", (long long)v.as.n);
            else printf("%g", v.as.n);
            break;
        case ZEY_STR: printf("%s", v.as.s); break;
        case ZEY_LIST:
            printf("[");
            for (int i = 0; i < v.as.list->count; i++) {
                if (i > 0) printf(", ");
                zey_print_value(v.as.list->items[i]);
            }
            printf("]");
            break;
    }
}

static ZeyValue zey_print(ZeyValue v) {
    zey_print_value(v);
    printf("\n");
    return zey_none();
}

static ZeyValue zey_ai_chat(ZeyValue message) {
    printf("[ai@chat] ");
    zey_print_value(message);
    printf("\n");
    return message;
}

static ZeyValue zey_robot_walk(ZeyValue steps) {
    printf("[robot@walk] walking ");
    zey_print_value(steps);
    printf(" step(s)\n");
    return steps;
}

/* ---- Part 5: robot library (all simulated, no hardware) ---- */

static ZeyValue zey_robot_arm(ZeyValue servo, ZeyValue angle) {
    printf("[robot@arm] servo ");
    zey_print_value(servo);
    printf(" -> ");
    zey_print_value(angle);
    printf("\xc2\xb0\n");   /* degree sign, as explicit UTF-8 bytes */
    return angle;
}

static ZeyValue zey_robot_grip(ZeyValue state) {
    if (state.tag != ZEY_STR || (strcmp(state.as.s, "open") != 0 && strcmp(state.as.s, "close") != 0)) {
        zey_error("robot@grip(...) expects \"open\" or \"close\"");
    }
    printf("[robot@grip] gripper -> %s\n", state.as.s);
    return state;
}

/* sensor() is a query, not a command — no printf, just returns a
   dummy value in a plausible range (or a 3-element [x,y,z] list for
   gyro), matching how the Part 5 examples use it. */
static ZeyValue zey_robot_sensor(ZeyValue kind) {
    if (kind.tag != ZEY_STR) zey_error("robot@sensor(...) expects a string");
    double r = (double)rand() / (double)RAND_MAX;
    if (strcmp(kind.as.s, "temp") == 0) return zey_num(20.0 + r * 10.0);   /* 20..30 C */
    if (strcmp(kind.as.s, "dist") == 0) return zey_num(5.0 + r * 195.0);   /* 5..200 cm */
    if (strcmp(kind.as.s, "gyro") == 0) {
        ZeyValue list = zey_list_new();
        for (int i = 0; i < 3; i++) {
            double axis = -5.0 + ((double)rand() / (double)RAND_MAX) * 10.0;  /* -5..5 deg/s */
            zey_list_push(list, zey_num(axis));
        }
        return list;
    }
    zey_error("robot@sensor(...): unknown sensor type (expected \"temp\", \"dist\", or \"gyro\")");
    return zey_none();
}

static ZeyValue zey_robot_speak(ZeyValue text) {
    printf("[robot@speak] ");
    zey_print_value(text);
    printf("\n");
    return text;
}

static ZeyValue zey_space_orbit(ZeyValue altitude, ZeyValue unit) {
    printf("[space@orbit] orbiting at ");
    zey_print_value(altitude);
    zey_print_value(unit);
    printf("\n");
    return altitude;
}

/* ---- Part 6: space library (all simulated) ---- */

static ZeyValue zey_space_launch(ZeyValue payload, ZeyValue target) {
    printf("[space@launch] ");
    zey_print_value(payload);
    printf(" -> ");
    zey_print_value(target);
    printf("\n");
    return payload;
}

static ZeyValue zey_space_thrust(ZeyValue percent) {
    if (percent.tag != ZEY_NUM || percent.as.n < 0 || percent.as.n > 100) {
        zey_error("space@thrust(...) expects 0-100");
    }
    printf("[space@thrust] engine -> ");
    zey_print_value(percent);
    printf("%%\n");
    return percent;
}

static ZeyValue zey_space_navigate(ZeyValue x, ZeyValue y, ZeyValue z) {
    printf("[space@navigate] heading to (");
    zey_print_value(x);
    printf(", ");
    zey_print_value(y);
    printf(", ");
    zey_print_value(z);
    printf(")\n");
    ZeyValue list = zey_list_new();
    zey_list_push(list, x);
    zey_list_push(list, y);
    zey_list_push(list, z);
    return list;
}

/* telemetry() is a query, not a command — stays quiet like
   robot@sensor(), matching how the Part 6 example uses it. Fixed-order
   5-element list: [speed, altitude, fuel, temp, position], position
   itself a 3-element [x, y, z] list (see the _Space docstring in
   visitor.py for why a list instead of a dict). */
static ZeyValue zey_space_telemetry(void) {
    ZeyValue result = zey_list_new();
    zey_list_push(result, zey_num(20000.0 + ((double)rand() / (double)RAND_MAX) * 8000.0));
    zey_list_push(result, zey_num(300.0 + ((double)rand() / (double)RAND_MAX) * 150.0));
    zey_list_push(result, zey_num(10.0 + ((double)rand() / (double)RAND_MAX) * 85.0));
    zey_list_push(result, zey_num(-20.0 + ((double)rand() / (double)RAND_MAX) * 80.0));
    ZeyValue pos = zey_list_new();
    for (int i = 0; i < 3; i++) {
        zey_list_push(pos, zey_num(-1000.0 + ((double)rand() / (double)RAND_MAX) * 2000.0));
    }
    zey_list_push(result, pos);
    return result;
}

static ZeyValue zey_space_touchdown(ZeyValue coord) {
    printf("[space@touchdown] touchdown at ");
    zey_print_value(coord);
    printf("\n");
    return coord;
}

static ZeyValue zey_add(ZeyValue a, ZeyValue b) {
    if (a.tag == ZEY_NUM && b.tag == ZEY_NUM) return zey_num(a.as.n + b.as.n);
    if (a.tag == ZEY_STR && b.tag == ZEY_STR) {
        size_t len = strlen(a.as.s) + strlen(b.as.s) + 1;
        char *buf = (char*)malloc(len);
        snprintf(buf, len, "%s%s", a.as.s, b.as.s);
        return zey_str(buf);
    }
    zey_error("unsupported operand types for +");
    return zey_none();
}
static ZeyValue zey_sub(ZeyValue a, ZeyValue b) {
    if (a.tag == ZEY_NUM && b.tag == ZEY_NUM) return zey_num(a.as.n - b.as.n);
    zey_error("unsupported operand types for -");
    return zey_none();
}
static ZeyValue zey_mul(ZeyValue a, ZeyValue b) {
    if (a.tag == ZEY_NUM && b.tag == ZEY_NUM) return zey_num(a.as.n * b.as.n);
    zey_error("unsupported operand types for *");
    return zey_none();
}
static ZeyValue zey_div(ZeyValue a, ZeyValue b) {
    if (a.tag == ZEY_NUM && b.tag == ZEY_NUM) {
        if (b.as.n == 0) zey_error("division by zero");
        return zey_num(a.as.n / b.as.n);
    }
    zey_error("unsupported operand types for /");
    return zey_none();
}
static ZeyValue zey_mod(ZeyValue a, ZeyValue b) {
    if (a.tag == ZEY_NUM && b.tag == ZEY_NUM) {
        long long bi = (long long)b.as.n;
        if (bi == 0) zey_error("modulo by zero");
        return zey_num((double)((long long)a.as.n % bi));
    }
    zey_error("unsupported operand types for %%");
    return zey_none();
}
static ZeyValue zey_lt(ZeyValue a, ZeyValue b) {
    if (a.tag == ZEY_NUM && b.tag == ZEY_NUM) return zey_bool(a.as.n < b.as.n);
    if (a.tag == ZEY_STR && b.tag == ZEY_STR) return zey_bool(strcmp(a.as.s, b.as.s) < 0);
    zey_error("unsupported operand types for <");
    return zey_none();
}
static ZeyValue zey_gt(ZeyValue a, ZeyValue b) {
    if (a.tag == ZEY_NUM && b.tag == ZEY_NUM) return zey_bool(a.as.n > b.as.n);
    if (a.tag == ZEY_STR && b.tag == ZEY_STR) return zey_bool(strcmp(a.as.s, b.as.s) > 0);
    zey_error("unsupported operand types for >");
    return zey_none();
}
static ZeyValue zey_le(ZeyValue a, ZeyValue b) {
    if (a.tag == ZEY_NUM && b.tag == ZEY_NUM) return zey_bool(a.as.n <= b.as.n);
    if (a.tag == ZEY_STR && b.tag == ZEY_STR) return zey_bool(strcmp(a.as.s, b.as.s) <= 0);
    zey_error("unsupported operand types for <=");
    return zey_none();
}
static ZeyValue zey_ge(ZeyValue a, ZeyValue b) {
    if (a.tag == ZEY_NUM && b.tag == ZEY_NUM) return zey_bool(a.as.n >= b.as.n);
    if (a.tag == ZEY_STR && b.tag == ZEY_STR) return zey_bool(strcmp(a.as.s, b.as.s) >= 0);
    zey_error("unsupported operand types for >=");
    return zey_none();
}

static int zey_values_equal(ZeyValue a, ZeyValue b) {
    if (a.tag != b.tag) return 0;
    switch (a.tag) {
        case ZEY_NONE: return 1;
        case ZEY_BOOL: return a.as.b == b.as.b;
        case ZEY_NUM:  return a.as.n == b.as.n;
        case ZEY_STR:  return strcmp(a.as.s, b.as.s) == 0;
        case ZEY_LIST: {
            if (a.as.list->count != b.as.list->count) return 0;
            for (int i = 0; i < a.as.list->count; i++)
                if (!zey_values_equal(a.as.list->items[i], b.as.list->items[i])) return 0;
            return 1;
        }
    }
    return 0;
}
static ZeyValue zey_eq(ZeyValue a, ZeyValue b) { return zey_bool(zey_values_equal(a, b)); }
static ZeyValue zey_neq(ZeyValue a, ZeyValue b) { return zey_bool(!zey_values_equal(a, b)); }

static ZeyValue zey_neg(ZeyValue a) {
    if (a.tag == ZEY_NUM) return zey_num(-a.as.n);
    zey_error("unsupported operand type for unary -");
    return zey_none();
}
static ZeyValue zey_not(ZeyValue a) { return zey_bool(!zey_truthy(a)); }

#define ZEY_AND(a_expr, b_expr) ({ ZeyValue _zt = (a_expr); zey_truthy(_zt) ? (b_expr) : _zt; })
#define ZEY_OR(a_expr, b_expr)  ({ ZeyValue _zt = (a_expr); zey_truthy(_zt) ? _zt : (b_expr); })
// ==== end runtime — compiled program follows ===="""


# Only prepended when the program actually calls ai@prompt/model/system/config
# (Compiler.uses_ai_runtime, set during _compile_call). Needs ZeyValue/zey_str/
# zey_none/zey_error from C_RUNTIME above, so it's always appended after it.
C_AI_RUNTIME = r"""// ==== ZeyLang AI runtime (Part 4) ====
// ai@prompt() shells out to the `curl` CLI via fork/exec/pipe (no libcurl
// linking) so building still needs nothing beyond `gcc -o program program.c`
// — just `curl` present on PATH at runtime.
#include <unistd.h>
#include <sys/wait.h>

static const char *zey_ai_endpoint = "https://zey-ai.vercel.app/api/chat";
static char   zey_ai_api_key[512]  = "";   // set via ai@config(key=...) or ZEY_AI_API_KEY
static char   zey_ai_model[256]    = "llama-3.3-70b-versatile";
static char  *zey_ai_system_prompt = NULL; // NULL = none set (via ai@system(...))
static double zey_ai_temperature   = 0.7;
static int    zey_ai_max_tokens    = 1024;

static void zey_ai_init(void) {
    const char *env_key = getenv("ZEY_AI_API_KEY");
    if (env_key) {
        strncpy(zey_ai_api_key, env_key, sizeof(zey_ai_api_key) - 1);
        zey_ai_api_key[sizeof(zey_ai_api_key) - 1] = '\0';
    }
}

static ZeyValue zey_ai_set_model(ZeyValue name) {
    if (name.tag != ZEY_STR) zey_error("ai@model(...) expects a string");
    strncpy(zey_ai_model, name.as.s, sizeof(zey_ai_model) - 1);
    zey_ai_model[sizeof(zey_ai_model) - 1] = '\0';
    return name;
}
static ZeyValue zey_ai_set_system(ZeyValue text) {
    if (text.tag != ZEY_STR) zey_error("ai@system(...) expects a string");
    zey_ai_system_prompt = (char*)text.as.s;
    return text;
}
static void zey_ai_set_key(ZeyValue v) {
    if (v.tag != ZEY_STR) zey_error("ai@config(key=...) expects a string");
    strncpy(zey_ai_api_key, v.as.s, sizeof(zey_ai_api_key) - 1);
    zey_ai_api_key[sizeof(zey_ai_api_key) - 1] = '\0';
}
static void zey_ai_set_temp(ZeyValue v) {
    if (v.tag != ZEY_NUM) zey_error("ai@config(temp=...) expects a number");
    zey_ai_temperature = v.as.n;
}
static void zey_ai_set_max(ZeyValue v) {
    if (v.tag != ZEY_NUM) zey_error("ai@config(max=...) expects a number");
    zey_ai_max_tokens = (int)v.as.n;
}

static char *zey_json_escape(const char *s) {
    size_t len = strlen(s);
    char *out = (char*)malloc(len * 6 + 1);
    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') { out[j++] = '\\'; out[j++] = (char)c; }
        else if (c == '\n') { out[j++] = '\\'; out[j++] = 'n'; }
        else if (c == '\t') { out[j++] = '\\'; out[j++] = 't'; }
        else if (c == '\r') { out[j++] = '\\'; out[j++] = 'r'; }
        else if (c < 0x20) { j += (size_t)snprintf(out + j, 7, "\\u%04x", c); }
        else { out[j++] = (char)c; }
    }
    out[j] = '\0';
    return out;
}

// Minimal, targeted extraction — NOT a general JSON parser. Looks for
// "content"/"response"/"message"/"text" (a Groq/OpenAI-compatible chat
// completions shape is assumed, since the model name in the Part 4
// example is a real Groq-hosted model). Falls back to the raw body so
// a schema mismatch is visible here instead of silently wrong.
static char *zey_extract_ai_text(const char *raw) {
    static const char *keys[] = { "\"content\"", "\"response\"", "\"message\"", "\"text\"" };
    for (int k = 0; k < 4; k++) {
        const char *p = strstr(raw, keys[k]);
        if (!p) continue;
        p = strchr(p + strlen(keys[k]), ':');
        if (!p) continue;
        p++;
        while (*p == ' ' || *p == '\t' || *p == '\n') p++;
        if (*p != '"') continue;
        p++;
        char *out = (char*)malloc(strlen(p) + 1);
        size_t j = 0;
        while (*p && *p != '"') {
            if (*p == '\\' && *(p + 1)) {
                p++;
                if (*p == 'n') out[j++] = '\n';
                else if (*p == 't') out[j++] = '\t';
                else if (*p == 'r') out[j++] = '\r';
                else out[j++] = *p;
            } else {
                out[j++] = *p;
            }
            p++;
        }
        out[j] = '\0';
        return out;
    }
    char *out = (char*)malloc(strlen(raw) + 1);
    strcpy(out, raw);
    return out;
}

/* Runs `curl <argv...>`, feeding `body` to its stdin and returning its
   stdout, via fork/exec/pipe — no libcurl linking, just the `curl`
   binary on PATH at runtime. */
static char *zey_run_curl(char *const argv[], const char *body) {
    int in_pipe[2], out_pipe[2];
    if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) zey_error("failed to create pipes for HTTP request");
    pid_t pid = fork();
    if (pid < 0) zey_error("fork() failed");
    if (pid == 0) {
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        close(in_pipe[0]); close(in_pipe[1]);
        close(out_pipe[0]); close(out_pipe[1]);
        execvp("curl", argv);
        fprintf(stderr, "[Runtime] failed to exec curl - is it installed and on PATH?\n");
        _exit(127);
    }
    close(in_pipe[0]);
    close(out_pipe[1]);

    size_t body_len = strlen(body), written = 0;
    while (written < body_len) {
        ssize_t n = write(in_pipe[1], body + written, body_len - written);
        if (n <= 0) break;
        written += (size_t)n;
    }
    close(in_pipe[1]);

    size_t cap = 4096, len = 0;
    char *out = (char*)malloc(cap);
    char chunk[4096];
    ssize_t n;
    while ((n = read(out_pipe[0], chunk, sizeof(chunk))) > 0) {
        if (len + (size_t)n + 1 > cap) {
            cap = (len + (size_t)n + 1) * 2;
            out = (char*)realloc(out, cap);
        }
        memcpy(out + len, chunk, (size_t)n);
        len += (size_t)n;
    }
    out[len] = '\0';
    close(out_pipe[0]);
    int status;
    waitpid(pid, &status, 0);
    return out;
}

static ZeyValue zey_ai_prompt(ZeyValue text) {
    if (text.tag != ZEY_STR) zey_error("ai@prompt(...) expects a string");
    if (zey_ai_api_key[0] == '\0') {
        zey_error("no AI API key set - call ai@config(key=\"...\") or set the ZEY_AI_API_KEY environment variable");
    }

    char *escaped_text = zey_json_escape(text.as.s);
    char *escaped_system = zey_ai_system_prompt ? zey_json_escape(zey_ai_system_prompt) : NULL;

    size_t body_cap = strlen(escaped_text) + (escaped_system ? strlen(escaped_system) : 0) + strlen(zey_ai_model) + 256;
    char *body = (char*)malloc(body_cap);
    if (escaped_system) {
        snprintf(body, body_cap,
            "{\"model\":\"%s\",\"messages\":[{\"role\":\"system\",\"content\":\"%s\"},"
            "{\"role\":\"user\",\"content\":\"%s\"}],\"temperature\":%g,\"max_tokens\":%d}",
            zey_ai_model, escaped_system, escaped_text, zey_ai_temperature, zey_ai_max_tokens);
    } else {
        snprintf(body, body_cap,
            "{\"model\":\"%s\",\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}],"
            "\"temperature\":%g,\"max_tokens\":%d}",
            zey_ai_model, escaped_text, zey_ai_temperature, zey_ai_max_tokens);
    }
    free(escaped_text);
    if (escaped_system) free(escaped_system);

    char key_header[600];
    snprintf(key_header, sizeof(key_header), "x-api-key: %s", zey_ai_api_key);

    char *argv[] = {
        (char*)"curl", (char*)"-s", (char*)"-X", (char*)"POST", (char*)zey_ai_endpoint,
        (char*)"-H", key_header,
        (char*)"-H", (char*)"Content-Type: application/json",
        (char*)"-d", (char*)"@-",
        (char*)"-w", (char*)"\nZEYSTATUS:%{http_code}",
        NULL
    };

    char *raw = zey_run_curl(argv, body);
    free(body);

    char *marker = strstr(raw, "\nZEYSTATUS:");
    long http_status = 0;
    if (marker) {
        http_status = strtol(marker + strlen("\nZEYSTATUS:"), NULL, 10);
        *marker = '\0';
    }

    if (http_status < 200 || http_status >= 300) {
        char msg[700];
        snprintf(msg, sizeof(msg), "AI request failed (HTTP %ld): %s", http_status, raw);
        free(raw);
        zey_error(msg);
    }

    char *text_out = zey_extract_ai_text(raw);
    free(raw);
    return zey_str(text_out);
}
// ==== end AI runtime ===="""


_BINOP_FUNCS = {
    "+": "zey_add", "-": "zey_sub", "*": "zey_mul", "/": "zey_div", "%": "zey_mod",
    "==": "zey_eq", "!=": "zey_neq", "<": "zey_lt", ">": "zey_gt", "<=": "zey_le", ">=": "zey_ge",
}


class Compiler:
    def __init__(self):
        self._temp_counter = 0
        self.function_arities = {}
        self.uses_ai_runtime = False   # set True on first ai@prompt/model/system/config call

    def _temp_name(self):
        self._temp_counter += 1
        return f"_t{self._temp_counter}"

    # ---- expressions ----
    def compile_expr(self, node) -> str:
        if isinstance(node, Literal):
            return self._compile_literal(node)
        if isinstance(node, Identifier):
            return f"zv_{node.name}"
        if isinstance(node, ListLiteral):
            return self._compile_list_literal(node)
        if isinstance(node, BinOp):
            return self._compile_binop(node)
        if isinstance(node, UnaryOp):
            return self._compile_unaryop(node)
        if isinstance(node, Call):
            return self._compile_call(node)
        if isinstance(node, Index):
            return self._compile_index(node)
        if isinstance(node, Attribute):
            raise CompileError("bare attribute access is not supported by the compiler")
        raise CompileError(f"unsupported expression: {type(node).__name__}")

    def _compile_literal(self, node: Literal):
        value = node.value
        if value is None:
            return "zey_none()"
        if isinstance(value, bool):
            return f"zey_bool({1 if value else 0})"
        if isinstance(value, (int, float)):
            return f"zey_num({value})"
        if isinstance(value, str):
            return f"zey_str({_c_string_literal(value)})"
        raise CompileError(f"unsupported literal type: {type(value).__name__}")

    def _compile_list_literal(self, node: ListLiteral):
        tmp = self._temp_name()
        parts = [f"ZeyValue {tmp} = zey_list_new()"]
        parts += [f"zey_list_push({tmp}, {self.compile_expr(el)})" for el in node.elements]
        parts.append(tmp)
        return "({ " + "; ".join(parts) + "; })"

    def _compile_index(self, node: Index):
        return f"zey_index({self.compile_expr(node.obj)}, {self.compile_expr(node.index)})"
func = _BINOP_FUNCS.get(node.op)
        if func is None:
            raise CompileError(f"unsupported operator '{node.op}'")
        return f"{func}({self.compile_expr(node.left)}, {self.compile_expr(node.right)})"

    def _compile_unaryop(self, node: UnaryOp):
        if node.op == "-":
            return f"zey_neg({self.compile_expr(node.operand)})"
        if node.op == "not":
            return f"zey_not({self.compile_expr(node.operand)})"
        raise CompileError(f"unsupported unary operator '{node.op}'")

    def _check_arity(self, name, args, expected):
        if len(args) != expected:
            raise CompileError(f"'{name}' expects {expected} arg(s), got {len(args)}")

    def _compile_call(self, node: Call):
        callee, args = node.callee, node.args

        if isinstance(callee, Attribute) and isinstance(callee.obj, Identifier):
            ns, method = callee.obj.name, callee.name
            if ns == "ai" and method == "chat":
                self._check_arity("ai@chat", args, 1)
                return f"zey_ai_chat({self.compile_expr(args[0])})"
            if ns == "robot" and method == "walk":
                self._check_arity("robot@walk", args, 1)
                return f"zey_robot_walk({self.compile_expr(args[0])})"
            if ns == "robot" and method == "arm":
                self._check_arity("robot@arm", args, 2)
                return f"zey_robot_arm({self.compile_expr(args[0])}, {self.compile_expr(args[1])})"
            if ns == "robot" and method == "grip":
                self._check_arity("robot@grip", args, 1)
                return f"zey_robot_grip({self.compile_expr(args[0])})"
            if ns == "robot" and method == "sensor":
                self._check_arity("robot@sensor", args, 1)
                return f"zey_robot_sensor({self.compile_expr(args[0])})"
            if ns == "robot" and method == "speak":
                self._check_arity("robot@speak", args, 1)
                return f"zey_robot_speak({self.compile_expr(args[0])})"
            if ns == "space" and method == "orbit":
                if len(args) == 1:
                    return f'zey_space_orbit({self.compile_expr(args[0])}, zey_str("km"))'
                self._check_arity("space@orbit", args, 2)
                return f"zey_space_orbit({self.compile_expr(args[0])}, {self.compile_expr(args[1])})"
            if ns == "space" and method == "launch":
                self._check_arity("space@launch", args, 2)
                return f"zey_space_launch({self.compile_expr(args[0])}, {self.compile_expr(args[1])})"
            if ns == "space" and method == "thrust":
                self._check_arity("space@thrust", args, 1)
                return f"zey_space_thrust({self.compile_expr(args[0])})"
            if ns == "space" and method == "navigate":
                self._check_arity("space@navigate", args, 3)
                a0, a1, a2 = (self.compile_expr(a) for a in args)
                return f"zey_space_navigate({a0}, {a1}, {a2})"
            if ns == "space" and method == "telemetry":
                self._check_arity("space@telemetry", args, 0)
                return "zey_space_telemetry()"
            if ns == "space" and method == "touchdown":
                self._check_arity("space@touchdown", args, 1)
                return f"zey_space_touchdown({self.compile_expr(args[0])})"
            if ns == "ai" and method == "prompt":
                self.uses_ai_runtime = True
                self._check_arity("ai@prompt", args, 1)
                return f"zey_ai_prompt({self.compile_expr(args[0])})"
            if ns == "ai" and method == "model":
                self.uses_ai_runtime = True
                self._check_arity("ai@model", args, 1)
                return f"zey_ai_set_model({self.compile_expr(args[0])})"
            if ns == "ai" and method == "system":
                self.uses_ai_runtime = True
                self._check_arity("ai@system", args, 1)
                return f"zey_ai_set_system({self.compile_expr(args[0])})"
            if ns == "ai" and method == "config":
                self.uses_ai_runtime = True
                if args:
                    raise CompileError("ai@config only accepts keyword arguments: key=, temp=, max=")
                unknown = set(node.kwargs) - {"key", "temp", "max"}
                if unknown:
                    raise CompileError(f"ai@config: unknown option(s) {sorted(unknown)}")
                setters = []
                if "key" in node.kwargs:
                    setters.append(f"zey_ai_set_key({self.compile_expr(node.kwargs['key'])})")
                if "temp" in node.kwargs:
                    setters.append(f"zey_ai_set_temp({self.compile_expr(node.kwargs['temp'])})")
                if "max" in node.kwargs:
                    setters.append(f"zey_ai_set_max({self.compile_expr(node.kwargs['max'])})")
                if not setters:
                    return "zey_none()"
                return "(" + ", ".join(setters) + ", zey_none())"
            raise CompileError(
                f"'{ns}@{method}' is not supported by the compiler (only ai@chat, ai@prompt, "
                "ai@model, ai@system, ai@config, robot@walk, robot@arm, robot@grip, "
                "robot@sensor, robot@speak, space@orbit, space@launch, space@thrust, "
                "space@navigate, space@telemetry, space@touchdown)"
            )

        if isinstance(callee, Identifier):
            if callee.name == "print":
                self._check_arity("print", args, 1)
                return f"zey_print({self.compile_expr(args[0])})"
            if callee.name == "range":
                raise CompileError("range() is only supported directly inside 'for ... in range(...)'")
            if callee.name in self.function_arities:
                expected = self.function_arities[callee.name]
                self._check_arity(callee.name, args, expected)
                arg_list = ", ".join(self.compile_expr(a) for a in args)
                return f"zeyfn_{callee.name}({arg_list})"
            raise CompileError(f"call to unknown function '{callee.name}'")

        raise CompileError("unsupported call target")

    # ---- statements ----
    def compile_stmt(self, node, indent: str):
        method = getattr(self, f"_compile_stmt_{type(node).__name__}", None)
        if method is None:
            raise CompileError(f"unsupported statement: {type(node).__name__}")
        return method(node, indent)

    def _compile_stmt_Assign(self, node: Assign, indent):
        return [f"{indent}zv_{node.name} = {self.compile_expr(node.value)};"]

    def _compile_stmt_Return(self, node: Return, indent):
        value = self.compile_expr(node.value) if node.value is not None else "zey_none()"
        return [f"{indent}return {value};"]

    def _compile_stmt_Break(self, node: Break, indent):
        return [f"{indent}break;"]

    def _compile_stmt_Continue(self, node: Continue, indent):
        return [f"{indent}continue;"]

    def _compile_stmt_Pass(self, node: Pass, indent):
        return []

    def _compile_stmt_ExprStmt(self, node: ExprStmt, indent):
        return [f"{indent}{self.compile_expr(node.expr)};"]

    def _compile_stmt_If(self, node: If, indent):
        lines = []
        for idx, (cond, body) in enumerate(node.branches):
            prefix = "if" if idx == 0 else "} else if"
            lines.append(f"{indent}{prefix} (zey_truthy({self.compile_expr(cond)})) {{")
            for stmt in body:
                lines.extend(self.compile_stmt(stmt, indent + "    "))
        if node.else_body is not None:
            lines.append(f"{indent}}} else {{")
            for stmt in node.else_body:
                lines.extend(self.compile_stmt(stmt, indent + "    "))
        lines.append(f"{indent}}}")
        return lines

    def _compile_stmt_While(self, node: While, indent):
        lines = [f"{indent}while (zey_truthy({self.compile_expr(node.condition)})) {{"]
        for stmt in node.body:
            lines.extend(self.compile_stmt(stmt, indent + "    "))
        lines.append(f"{indent}}}")
        return lines

    def _compile_stmt_For(self, node: For, indent):
        iterable = node.iterable
        is_range = (
            isinstance(iterable, Call)
            and isinstance(iterable.callee, Identifier)
            and iterable.callee.name == "range"
        )
        if is_range:
            args = iterable.args
            if len(args) == 1:
                start_c, stop_c = "0.0", f"({self.compile_expr(args[0])}).as.n"
            elif len(args) == 2:
                start_c = f"({self.compile_expr(args[0])}).as.n"
                stop_c = f"({self.compile_expr(args[1])}).as.n"
            else:
                raise CompileError("range() takes 1 or 2 arguments in compiled code")
            i = self._temp_name()
            lines = [f"{indent}for (double {i} = {start_c}; {i} < {stop_c}; {i} += 1.0) {{"]
            lines.append(f"{indent}    zv_{node.var_name} = zey_num({i});")
            for stmt in node.body:
                lines.extend(self.compile_stmt(stmt, indent + "    "))
            lines.append(f"{indent}}}")
            return lines

        it, idx = self._temp_name(), self._temp_name()
        lines = [
            f"{indent}{{",
            f"{indent}    ZeyValue {it} = {self.compile_expr(iterable)};",
            f'{indent}    if ({it}.tag != ZEY_LIST) zey_error("\'for\' target is not iterable");',
            f"{indent}    for (int {idx} = 0; {idx} < {it}.as.list->count; {idx}++) {{",
            f"{indent}        zv_{node.var_name} = {it}.as.list->items[{idx}];",
        ]
        for stmt in node.body:
            lines.extend(self.compile_stmt(stmt, indent + "        "))
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        return lines

    # ---- top level ----
    def _compile_prototype(self, node: FuncDef):
        params = ", ".join(f"ZeyValue zv_{p}" for p in node.params) if node.params else "void"
        return f"static ZeyValue zeyfn_{node.name}({params});"

    def _compile_funcdef(self, node: FuncDef):
        params = ", ".join(f"ZeyValue zv_{p}" for p in node.params) if node.params else "void"
        locals_ = _collect_locals(node.body, exclude=set(node.params))
        lines = [f"static ZeyValue zeyfn_{node.name}({params}) {{"]
        for name in locals_:
            lines.append(f"    ZeyValue zv_{name} = zey_none();")
        for stmt in node.body:
            lines.extend(self.compile_stmt(stmt, "    "))
        lines.append("    return zey_none();")
        lines.append("}")
        return lines

    def _compile_main(self, main_stmts):
        locals_ = _collect_locals(main_stmts, exclude=set())
        lines = ["int main(void) {", "    srand((unsigned int)time(NULL) ^ (unsigned int)getpid());"]
        for name in locals_:
            lines.append(f"    ZeyValue zv_{name} = zey_none();")
        for stmt in main_stmts:
            lines.extend(self.compile_stmt(stmt, "    "))
        lines.append("    return 0;")
        lines.append("}")
        return lines


def compile_program(program) -> str:
    compiler = Compiler()
    funcdefs = [s for s in program.statements if isinstance(s, FuncDef)]
    main_stmts = [s for s in program.statements if not isinstance(s, FuncDef)]
    compiler.function_arities = {f.name: len(f.params) for f in funcdefs}

    proto_lines = [compiler._compile_prototype(f) for f in funcdefs]
    funcdef_lines = []
    for f in funcdefs:
        funcdef_lines.extend(compiler._compile_funcdef(f))
        funcdef_lines.append("")
    main_lines = compiler._compile_main(main_stmts)

    # compiler.uses_ai_runtime is only known for certain once every call in
    # the program has been compiled (funcdefs above, main just now), so the
    # AI runtime + its init call are decided here, after the fact.
    if compiler.uses_ai_runtime:
        main_lines.insert(1, "    zey_ai_init();")

    runtime = C_RUNTIME + ("\n\n" + C_AI_RUNTIME if compiler.uses_ai_runtime else "")

    lines = [runtime, ""]
    lines.extend(proto_lines)
    if funcdefs:
        lines.append("")
    lines.extend(funcdef_lines)
    lines.extend(main_lines)
    return "\n".join(lines) + "\n"


def compile_source(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    return compile_program(program)


def main():
    import argparse

    argp = argparse.ArgumentParser(description="Compile ZeyLang source to C.")
    argp.add_argument("source", help="path to a .zey source file")
    argp.add_argument("-o", "--output", help="output .c path (default: <source>.c)")
    argp.add_argument("--build", action="store_true", help="also invoke gcc to build a binary")
    argp.add_argument("--run", action="store_true", help="build and run the binary (implies --build)")
    args = argp.parse_args()

    with open(args.source, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        c_code = compile_source(source)
    except (LexerError, ParserError, CompileError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    out_path = args.output or os.path.splitext(args.source)[0] + ".c"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(c_code)
    print(f"Wrote {out_path}")

    if args.build or args.run:
        binary_path = os.path.splitext(out_path)[0]
        result = subprocess.run(["gcc", "-o", binary_path, out_path])
        if result.returncode != 0:
            print("gcc build failed", file=sys.stderr)
            sys.exit(1)
        print(f"Built {binary_path}")
        if args.run:
            subprocess.run([os.path.abspath(binary_path)])


if __name__ == "__main__":
    main()
