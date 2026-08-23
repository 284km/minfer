# PAIN.md — what dense float compute forced, and what it refuted

## P1 — the optimizer was part of the language (mere v0.1.315)

The find of the arc. At `-O2`, clang contracts `a*b+c` into fused
multiply-add by default on arm64 — one rounding where the interpreter (and
LLVM and Wasm backends) round twice — so this program's outputs differed
between `mere run` and its own `-O2` binary by an ulp on 4 of 10 outputs.
Every upstream gate compiled at `-O0`, and existing float parity compared
formatted output or quantized image bytes; the divergence lived exactly
where nobody looked. Mere's emitted C now carries
`#pragma STDC FP_CONTRACT OFF`, and an upstream gate compiles a dense dot
product at `-O0` and `-O2` demanding identical bits from both and from the
interpreter.

## P2 — the missing flat float array was not missing

The roadmap assumed a dense float buffer type would have to be added.
`Vec[R, float]` already compiles to `{ double* data; ... }` — a plain dense
array. The premise died on first contact with the code, before any design
work; checking the ledger before building remains cheaper than building.

## P3 — the bounds-check bill never arrived (a result)

The expected cost of this arc: `vec_get`/`vec_set` bounds-check every
element in the matmul inner loop. Measured at `-O2` against the same
network handwritten in C over plain arrays: indistinguishable (±10% run
noise, both directions across runs). No unchecked-access primitive is
needed on this evidence.

## P4 — rediscoveries, not discoveries

Two walls this program hit were already on the books: an int literal above
2^62 kills the interpreter with a bare `Failure("int_of_string")` (Q-037's
second boundary — interp int is 63-bit OCaml native against the compiled
backends' 64), and a trailing `exit 0` is the known open LLVM main-type
hole. Both worked around (a 48-bit LCG whose low bits survive both int
widths; a unit main). One near-miss: an inner closure over a polymorphic
vec was suspected of the known LLVM monomorphization gap and "fixed" —
then measured innocent (the ascribed element type monomorphizes fine).
The suspicion was a guess, and the file keeps the closure.

## What worked without friction

Loading 100K doubles through `read_file_bytes` + `float_of_bits` assembly:
~830 KB of weights, no new primitive, all four backends. The forward pass
reads like the math.
