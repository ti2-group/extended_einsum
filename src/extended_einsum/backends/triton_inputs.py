"""Triton input evaluation and input-gradient kernels.

Cirkit adapters and Torch references live in extended_einsum.inputs.
"""

from __future__ import annotations

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib


@tr.jit
def _rna(x, PRECISION: tl.constexpr):
    if PRECISION == "tf32":
        return tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", constraints="=f,f", args=[x], dtype=tl.float32, is_pure=True, pack=1)
    return x


@tr.jit
def _leaves(Pixels, Variables, LogCat, Nodes, B: tl.constexpr, V: tl.constexpr, F: tl.constexpr, C: tl.constexpr, U: tl.constexpr, M: tl.constexpr, P: tl.constexpr):
    f = tl.program_id(1)
    b = tl.program_id(0) * M + tl.arange(0, M)
    u = tl.arange(0, P)
    variable = tl.load(Variables + f).to(tl.int64)
    valid_variable = (variable >= 0) & (variable < V)
    c = tl.load(Pixels + b * V + variable.to(tl.int32), (b < B) & valid_variable, other=0).to(tl.int64)
    valid = valid_variable & (c >= 0) & (c < C)
    value = tl.load(LogCat + (f * C + c.to(tl.int32)[:, None]) * U + u[None, :], (b < B)[:, None] & valid[:, None] & (u < U)[None, :], other=0.0)
    value = tl.where(valid[:, None], value, float("nan"))
    tl.store(Nodes + (f * B + b[:, None]) * U + u[None, :], value, (b < B)[:, None] & (u < U)[None, :])


@tr.jit
def _categorical_backward(
    Pixels,
    Variables,
    Flow,
    LogCat,
    GradCat,
    Model,
    Momentum,
    Bias,
    B: tl.constexpr,
    V: tl.constexpr,
    F: tl.constexpr,
    C: tl.constexpr,
    U: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    STAGES: tl.constexpr,
    PRECISION: tl.constexpr,
    UPDATE: tl.constexpr,
    ALPHA: tl.constexpr,
    MOM: tl.constexpr,
):
    f = tl.program_id(1)
    u = tl.program_id(0) * N + tl.arange(0, N)
    c = tl.arange(0, P)
    kk = tl.arange(0, K)
    acc = tl.full((P, N), 0.0, tl.float32)
    for panel in tl.range(tr.cdiv(B, K), num_stages=STAGES):
        b = panel * K + kk
        variable = tl.load(Variables + f).to(tl.int64)
        valid = (variable >= 0) & (variable < V)
        cat = tl.load(Pixels + b * V + variable.to(tl.int32), (b < B) & valid, other=-1).to(tl.int64)
        indicator = ((c[:, None] == cat[None, :]) & (b < B)[None, :] & (c < C)[:, None]).to(tl.float32)
        flow = tl.load(Flow + (f * B + b[:, None]) * U + u[None, :], (b < B)[:, None] & (u < U)[None, :], other=0.0)
        acc = tl.dot(indicator, _rna(flow, PRECISION), acc, input_precision=PRECISION)
    pos = (f * C + c[:, None]) * U + u[None, :]
    mask = (c < C)[:, None] & (u < U)[None, :]
    if UPDATE:
        data = -acc
        if MOM > 0.0:
            running = MOM * tl.load(Momentum + pos, mask, other=0.0) + (1.0 - MOM) * data
            tl.store(Momentum + pos, running, mask)
            data = running / tl.load(Bias)
        mf = tl.load(Model + f * U + u, u < U, other=0.0)
        w = tl.exp(tl.load(LogCat + pos, mask, other=float("-inf")))
        numerator = tl.maximum((1.0 - ALPHA) * w * mf[None, :] + ALPHA * data, 1.1754943508222875e-38)
        numerator = tl.where((c < C)[:, None], numerator, 0.0)
        logp = tl.log(numerator) - tl.log(tl.sum(numerator, 0))[None, :]
        tl.store(LogCat + pos, logp, mask)
    else:
        tl.store(GradCat + pos, acc, mask)


@tr.jit
def _categorical_values(Pixels, Tables, Variables, Banks, root, rows, cols, B: tl.constexpr, V: tl.constexpr, C: tl.constexpr, BANKS: tl.constexpr, U: tl.constexpr, LOOKUP: tl.constexpr):
    variable = tl.load(Variables + root)
    bank = tl.load(Banks + root)
    valid_id = (variable >= 0) & (variable < V) & (bank >= 0) & (bank < BANKS)
    if LOOKUP == 0:
        category = tl.load(Pixels + rows * V + variable.to(tl.int32), (rows < B) & valid_id, other=0).to(tl.int64)
        valid = valid_id & (category >= 0) & (category < C)
        x = tl.load(Tables + (bank.to(tl.int32) * C + category.to(tl.int32)[:, None]) * U + cols[None, :], (rows < B)[:, None] & valid[:, None] & (cols < U)[None, :], other=0.0)
        x = tl.where(valid[:, None], x, float("nan"))
    else:
        category = tl.load(Pixels + rows[:, None] * V + variable.to(tl.int32), (rows < B)[:, None] & valid_id, other=0).to(tl.int64)
        valid = valid_id & (category >= 0) & (category < C)
        x = tl.load(Tables + (bank.to(tl.int32) * C + category.to(tl.int32)) * U + cols[None, :], (rows < B)[:, None] & valid & (cols < U)[None, :], other=0.0)
        x = tl.where(valid, x, float("nan"))
    return x


@tr.jit
def _raw(X, Routes, f, b, B: tl.constexpr, V: tl.constexpr):
    variable = tl.load(Routes + f).to(tl.int64)
    valid = (b < B) & (variable >= 0) & (variable < V)
    x = tl.load(X + b * V + variable.to(tl.int32), valid, other=0)
    return x, valid


@tr.jit
def _evaluate(
    x,
    f,
    u,
    P0,
    P1,
    P2,
    U: tl.constexpr,
    C: tl.constexpr,
    KIND: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    COUNT: tl.constexpr,
    DEGREE: tl.constexpr,
    PART: tl.constexpr,
):
    if KIND == "categorical_logits" or KIND == "categorical_probs" or KIND == "embedding":
        c = x.to(tl.int64)
        c = tl.where(c < 0, c + C, c)
        valid = (c >= 0) & (c < C) & (u < U)
        p = tl.load(P0 + f * S0 + u * S1 + c.to(tl.int32) * S2, valid, other=0.0)
        y = tl.log(p) if KIND == "categorical_probs" else p
        return tl.where(valid, y, float("nan"))
    elif KIND == "gaussian":
        mean = tl.load(P0 + f * S0 + u * S1, u < U, other=0.0)
        sd = tl.load(P1 + f * T0 + u * T1, u < U, other=1.0)
        z = (x.to(tl.float32) - mean) / sd
        y = -0.5 * z * z - tl.log(sd) - 0.9189385332046727
        if PART:
            y += tl.load(P2 + f * R0 + u * R1, u < U, other=0.0)
        return y
    elif KIND == "binomial_logits" or KIND == "binomial_probs":
        p = tl.load(P0 + f * S0 + u * S1, u < U, other=0.0)
        if KIND == "binomial_probs":
            p = tl.minimum(tl.maximum(p, 1.1920928955078125e-7), 1.0 - 1.1920928955078125e-7)
            p = tl.log(p) - lib.log1p(-p)
        k = x.to(tl.int64).to(tl.float32)
        y = k * p - COUNT * tl.maximum(p, 0.0) - COUNT * lib.log1p(tl.exp(-tl.abs(p)))
        y += lib.lgamma(COUNT + 1.0) - lib.lgamma(k + 1.0) - lib.lgamma(COUNT - k + 1.0)
        return tl.where((k >= 0) & (k <= COUNT), y, float("nan"))
    else:
        y = tl.full(x.shape, 0.0, tl.float32) + tl.full(u.shape, 0.0, tl.float32)
        for d in tl.static_range(DEGREE, -1, -1):
            a = tl.load(P0 + f * S0 + u * S1 + d * S2, u < U, other=0.0)
            y = a + x.to(tl.float32) * y
        return y


@tr.jit
def _input_forward(
    X,
    Routes,
    P0,
    P1,
    P2,
    Out,
    B: tl.constexpr,
    V: tl.constexpr,
    U: tl.constexpr,
    C: tl.constexpr,
    KIND: tl.constexpr,
    LOG: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    COUNT: tl.constexpr,
    DEGREE: tl.constexpr,
    PART: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH_FAST: tl.constexpr,
):
    f = tl.program_id(2)
    b = tl.program_id(1) * M + tl.arange(0, M)
    u = tl.program_id(0) * N + tl.arange(0, N)
    x, valid = _raw(X, Routes, f, b, B, V)
    if BATCH_FAST:
        xx, uu = x[None, :], u[:, None]
    else:
        xx, uu = x[:, None], u[None, :]
    y = _evaluate(xx, f, uu, P0, P1, P2, U, C, KIND, S0, S1, S2, T0, T1, R0, R1, COUNT, DEGREE, PART)
    if KIND == "embedding" or KIND == "polynomial":
        if LOG:
            y = tl.log(y)
    elif not LOG:
        y = tl.exp(y)
    if BATCH_FAST:
        y = tl.where(valid[None, :], y, float("nan"))
        tl.store(Out + (f * U + u[:, None]) * B + b[None, :], y, (u < U)[:, None] & (b < B)[None, :])
    else:
        y = tl.where(valid[:, None], y, float("nan"))
        tl.store(Out + (f * B + b[:, None]) * U + u[None, :], y, (b < B)[:, None] & (u < U)[None, :])


@tr.jit
def _input_backward(
    X,
    Routes,
    P0,
    P1,
    P2,
    Out,
    Grad,
    G0,
    G1,
    G2,
    GX,
    B: tl.constexpr,
    V: tl.constexpr,
    U: tl.constexpr,
    C: tl.constexpr,
    KIND: tl.constexpr,
    LOG: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    COUNT: tl.constexpr,
    DEGREE: tl.constexpr,
    PART: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    WANT0: tl.constexpr,
    WANT1: tl.constexpr,
    WANT2: tl.constexpr,
    WANTX: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    f = tl.program_id(1)
    u = tl.program_id(0) * N + tl.arange(0, N)
    b = tl.arange(0, M)
    x, valid = _raw(X, Routes, f, b, B, V)
    mask = (b < B)[:, None] & (u < U)[None, :]
    g = tl.load(Grad + f * GS0 + b[:, None] * GS1 + u[None, :] * GS2, mask, other=0.0)
    out = tl.load(Out + f * OS0 + b[:, None] * OS1 + u[None, :] * OS2, mask, other=0.0)
    if KIND == "polynomial":
        if LOG:
            g *= tl.exp(-out)
        g = tl.where(mask, g, 0.0)
        if WANT0:
            power = tl.full((M,), 1.0, tl.float32)
            for d in tl.static_range(DEGREE + 1):
                value = tl.sum(g * power[:, None], 0)
                tl.store(G0 + (f * U + u) * C + d, value, u < U)
                power *= x.to(tl.float32)
        if WANTX:
            y = tl.full((M, N), 0.0, tl.float32)
            dy = tl.full((M, N), 0.0, tl.float32)
            for d in tl.static_range(DEGREE, -1, -1):
                a = tl.load(P0 + f * S0 + u[None, :] * S1 + d * S2, (u < U)[None, :], other=0.0)
                dy = dy * x[:, None].to(tl.float32) + y
                y = y * x[:, None].to(tl.float32) + a
            dx = tl.sum(tl.where(mask, g * dy, 0.0), 1)
    else:
        if not LOG:
            g *= out
        g = tl.where(mask, g, 0.0)
        p = tl.load(P0 + f * S0 + u, u < U, other=0.0) if S1 == 1 else tl.load(P0 + f * S0 + u * S1, u < U, other=0.0)
        if KIND == "gaussian":
            sd = tl.load(P1 + f * T0 + u * T1, u < U, other=1.0)
            delta = x[:, None].to(tl.float32) - p[None, :]
            dm = delta / (sd * sd)[None, :]
            if WANT0:
                tl.store(G0 + f * U + u, tl.sum(tl.where(mask, g * dm, 0.0), 0), u < U)
            if WANT1:
                z = delta / sd[None, :]
                ds = (z * z - 1.0) / sd[None, :]
                tl.store(G1 + f * U + u, tl.sum(tl.where(mask, g * ds, 0.0), 0), u < U)
            if WANT2:
                tl.store(G2 + f * U + u, tl.sum(g, 0), u < U)
            if WANTX:
                dx = tl.sum(tl.where(mask, -g * dm, 0.0), 1)
        else:
            if KIND == "binomial_probs":
                safe = tl.minimum(tl.maximum(p, 1.1920928955078125e-7), 1.0 - 1.1920928955078125e-7)
                derivative = (x[:, None].to(tl.int64).to(tl.float32) - COUNT * safe[None, :]) / (safe * (1.0 - safe))[None, :]
                derivative = tl.where(((p >= 1.1920928955078125e-7) & (p <= 1.0 - 1.1920928955078125e-7))[None, :], derivative, 0.0)
            else:
                derivative = x[:, None].to(tl.int64).to(tl.float32) - COUNT * tl.sigmoid(p)[None, :]
            if WANT0:
                tl.store(G0 + f * U + u, tl.sum(g * derivative, 0), u < U)
    if WANTX:
        variable = tl.load(Routes + f).to(tl.int32)
        tl.atomic_add(GX + b * V + variable, dx, valid, sem="relaxed")


@tr.jit
def _lookup_backward(
    X,
    Routes,
    P0,
    Grad,
    G0,
    B: tl.constexpr,
    V: tl.constexpr,
    U: tl.constexpr,
    C: tl.constexpr,
    KIND: tl.constexpr,
    LOG: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    NC: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    f = tl.program_id(2)
    u = tl.program_id(0) * N + tl.arange(0, N)
    c = tl.program_id(1) * NC + tl.arange(0, NC)
    kk = tl.arange(0, K)
    acc = tl.full((NC, N), 0.0, tl.float32)
    counts = tl.full((NC,), 0, tl.int32)
    for panel in range(tr.cdiv(B, K)):
        b = panel * K + kk
        cat, valid = _raw(X, Routes, f, b, B, V)
        cat = cat.to(tl.int64)
        cat = tl.where(cat < 0, cat + C, cat)
        indicator = (c[:, None] == cat[None, :]) & valid[None, :] & (c < C)[:, None]
        counts += tl.sum(indicator.to(tl.int32), 1)
        g = tl.load(Grad + f * GS0 + b[:, None] * GS1 + u[None, :] * GS2, (b < B)[:, None] & (u < U)[None, :], other=0.0)
        acc = tl.dot(indicator.to(tl.float32), g, acc, input_precision="tf32x3")
    mask = (c < C)[:, None] & (u < U)[None, :]
    p = tl.load(P0 + f * S0 + u[None, :] * S1 + c[:, None] * S2, mask, other=1.0)
    if KIND == "categorical_probs":
        if LOG:
            acc = acc / p
        else:
            acc *= tl.exp(tl.log(p)) / p
    elif KIND == "categorical_logits":
        if not LOG:
            acc = tl.where(counts[:, None] > 0, acc * tl.exp(p), 0.0)
    elif LOG:
        acc = tl.where(counts[:, None] > 0, acc / p, 0.0)
    tl.store(G0 + (f * U + u[None, :]) * C + c[:, None], acc, mask)
