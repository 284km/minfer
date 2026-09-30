#!/bin/sh
# verify.sh — the network's answer is bits, and every runner must produce them.
#
#   MERE=/path/to/mere.exe sh verify.sh   (or MERE=<mere checkout>)
#
# Three claims:
#
#   oracle    gen_weights.py generates the weight file AND computes the
#             forward pass in plain Python, same accumulation order, and the
#             -O2 native binary must reproduce every output BIT (this is the
#             claim that found fp-contract: clang's default fma made 4 of 10
#             outputs differ by an ulp until mere pinned FP_CONTRACT OFF).
#   parity    interp, C, LLVM and Wasm all print the same transcript.
#   speed     the same forward pass, handwritten in C over plain double
#             arrays, timed against minfer's -O2 binary. The ratio is
#             REPORTED (bounds-checked vec_get in the hot loop is the cost
#             being measured); PAIN.md keeps the number.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
# MERE is the compiler (the convention most verify.sh files follow) or a mere
# checkout; either works. MERE_ROOT is the checkout when one can be found.
[ -n "${MERE:-}" ] || { echo "usage: MERE=/path/to/mere.exe (or a mere checkout) sh verify.sh" >&2; exit 2; }
if [ -d "$MERE" ]; then MERE_ROOT="$MERE"; M="$MERE/_build/default/bin/mere.exe"
else M="$MERE"; MERE_ROOT="$(cd "$(dirname "$MERE")/../../.." 2>/dev/null && pwd)"; fi
[ -x "$M" ] || { echo "verify: $M not found (dune build?)" >&2; exit 2; }
CC="${CC:-cc}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

python3 "$DIR/gen_weights.py" "$TMP/weights.bin"
python3 "$DIR/gen_weights.py" "$TMP/weights.bin" --oracle > "$TMP/want10"
# A unit main prints nothing since mere v0.1.494 (Q-136); this expected a
# trailing "()" line from before that, and every backend has been "wrong"
# against it since.
cp "$TMP/want10" "$TMP/want"

# --- oracle: the -O2 native binary reproduces every bit ------------------------
"$M" -c "$DIR/minfer.mere" > "$TMP/minfer.c" 2>"$TMP/err" \
  || { echo "FAIL verify: mere -c"; cat "$TMP/err"; exit 1; }
"$CC" -O2 -w "$TMP/minfer.c" -lm -o "$TMP/minfer_c" || { echo "FAIL verify: cc"; exit 1; }
"$TMP/minfer_c" "$TMP/weights.bin" > "$TMP/got_c"
diff "$TMP/want" "$TMP/got_c" > /dev/null \
  || { echo "FAIL verify: C output differs from the Python oracle"; diff "$TMP/want" "$TMP/got_c" | head -5; exit 1; }
echo "  ok | C -O2: all 10 outputs bit-identical to the oracle"

# --- parity: four backends, one transcript -------------------------------------
"$M" "$DIR/minfer.mere" "$TMP/weights.bin" > "$TMP/got_i" 2>&1
diff "$TMP/want" "$TMP/got_i" > /dev/null \
  || { echo "FAIL verify: interp differs"; diff "$TMP/want" "$TMP/got_i" | head -5; exit 1; }

# the LLVM backend has no native now_ms; the bench mode's extern is satisfied
# by a three-line shim (measurement apparatus, not part of the program)
cat > "$TMP/now_ms.c" <<'C'
#include <time.h>
long long now_ms(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}
C
"$M" -ll "$DIR/minfer.mere" > "$TMP/minfer.ll" 2>"$TMP/ll.err" \
  || { echo "FAIL verify: mere -ll"; cat "$TMP/ll.err"; exit 1; }
"$CC" -O2 -w "$TMP/minfer.ll" "$TMP/now_ms.c" -lm -o "$TMP/minfer_ll" \
  || { echo "FAIL verify: llvm build"; exit 1; }
"$TMP/minfer_ll" "$TMP/weights.bin" > "$TMP/got_ll"
diff "$TMP/want" "$TMP/got_ll" > /dev/null \
  || { echo "FAIL verify: LLVM differs"; diff "$TMP/want" "$TMP/got_ll" | head -5; exit 1; }

if command -v wat2wasm >/dev/null 2>&1 && command -v node >/dev/null 2>&1; then
  "$M" -w "$DIR/minfer.mere" > "$TMP/minfer.wat" 2>"$TMP/w.err" \
    || { echo "FAIL verify: mere -w"; cat "$TMP/w.err"; exit 1; }
  wat2wasm --enable-tail-call "$TMP/minfer.wat" -o "$TMP/minfer.wasm" \
    || { echo "FAIL verify: wat2wasm"; exit 1; }
  node "$MERE_ROOT/scripts/run_wasm.js" "$TMP/minfer.wasm" "$TMP/weights.bin" > "$TMP/got_w" 2>&1
  diff "$TMP/want" "$TMP/got_w" > /dev/null \
    || { echo "FAIL verify: Wasm differs"; diff "$TMP/want" "$TMP/got_w" | head -5; exit 1; }
  echo "  ok | interp / C / LLVM / Wasm: one transcript"
else
  echo "  ok | interp / C / LLVM: one transcript (wasm toolchain absent)"
fi

# --- speed: the bounds-check bill, measured ------------------------------------
cat > "$TMP/ref.c" <<'C'
/* the same forward pass over plain double arrays -- the floor minfer's
   vec_get/vec_push hot loop is measured against */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define IN 784
#define HID 128
#define OUT 10
static long long ms(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}
int main(int argc, char** argv) {
  int n = atoi(argv[2]);
  FILE* f = fopen(argv[1], "rb");
  static double w1[HID*IN], b1[HID], w2[OUT*HID], b2[OUT], x[IN], h[HID], y[OUT];
  if (fread(w1, 8, HID*IN, f) + fread(b1, 8, HID, f) + fread(w2, 8, OUT*HID, f)
      + fread(b2, 8, OUT, f) + fread(x, 8, IN, f) != HID*IN + HID + OUT*HID + OUT + IN)
    return 1;
  fclose(f);
  double sink = 0.0;
  long long t0 = ms();
  for (int r = 0; r < n; r++) {
    for (int j = 0; j < HID; j++) {
      double acc = b1[j];
      for (int i = 0; i < IN; i++) acc = acc + w1[j*IN+i] * x[i];
      h[j] = acc > 0.0 ? acc : 0.0;
    }
    for (int k = 0; k < OUT; k++) {
      double acc = b2[k];
      for (int j = 0; j < HID; j++) acc = acc + w2[k*HID+j] * h[j];
      y[k] = acc;
    }
    sink += y[0];
  }
  long long t1 = ms();
  printf("%lld ms\n", t1 - t0);
  fprintf(stderr, "%g\n", sink);
  return 0;
}
C
"$CC" -O2 -w -ffp-contract=off "$TMP/ref.c" -o "$TMP/ref"
N=2000
mere_ms="$("$TMP/minfer_c" "$TMP/weights.bin" bench $N | head -1 | awk '{print $1}')"
ref_ms="$("$TMP/ref" "$TMP/weights.bin" $N 2>/dev/null | awk '{print $1}')"
[ "${ref_ms:-0}" -gt 0 ] || ref_ms=1
ratio=$(( (mere_ms * 10 + ref_ms / 2) / ref_ms ))
echo "  ok | bench: $N passes -- minfer ${mere_ms} ms vs handwritten C ${ref_ms} ms (x$(( ratio / 10 )).$(( ratio % 10 )))"

echo "verify: ok"
