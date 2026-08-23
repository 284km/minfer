# minfer

A 784→128→10 MLP forward pass in [Mere](https://github.com/merelang/mere),
over dense float vectors — and the checks that make "it computes the right
numbers" a claim about bits.

```
minfer <weights.bin>          the 10 outputs, as raw IEEE-754 bits (hi lo)
minfer <weights.bin> bench N  N forward passes, then the time in ms
```

`gen_weights.py` generates the deterministic weight file and computes the
same forward pass in plain Python — same accumulation order, same doubles —
so `verify.sh` can require every output bit to match, with no tolerance.
Then it runs the identical program on all four Mere backends (interpreter,
C, LLVM, Wasm) and requires one transcript, and times the C build against
the same network handwritten in C over plain arrays.

Printing bits instead of digits keeps string formatting out of a claim that
is about arithmetic — which is how this program found a real one: at `-O2`,
clang's default fma contraction changed 4 of the 10 outputs by an ulp
relative to the interpreter, invisible to every formatted comparison. Mere
pins `FP_CONTRACT OFF` in its emitted C since v0.1.315 because of it.

The measured speed: indistinguishable from the handwritten C reference
(±10% run noise, both directions) — `Vec[R, float]` is a plain dense
`double*` at runtime, and the bounds-checked reads in the hot loop cost
nothing clang's optimizer keeps.

## License

MIT
