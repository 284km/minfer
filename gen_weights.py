#!/usr/bin/env python3
"""Deterministic weights for minfer — and the oracle that must agree with it.

The same LCG generates the weights here and nowhere else; minfer READS the
file, so the two implementations share bytes, not code. `forward` below is
the oracle: plain Python floats (IEEE doubles), accumulated in exactly the
order minfer accumulates, so agreement is bit-for-bit or a finding.

usage: gen_weights.py <weights.bin>          write the weight file
       gen_weights.py <weights.bin> --oracle print the expected output bits
"""
import struct
import sys

IN, HID, OUT = 784, 128, 10


def lcg(seed):
    s = seed
    while True:
        s = (s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        # a small float in [-0.5, 0.5), from the top bits
        yield ((s >> 33) / float(1 << 31)) - 0.5


def weights():
    g = lcg(20260823)
    w1 = [next(g) for _ in range(HID * IN)]
    b1 = [next(g) for _ in range(HID)]
    w2 = [next(g) for _ in range(OUT * HID)]
    b2 = [next(g) for _ in range(OUT)]
    x = [next(g) for _ in range(IN)]
    return w1, b1, w2, b2, x


def forward(w1, b1, w2, b2, x):
    h = []
    for j in range(HID):
        acc = b1[j]
        for i in range(IN):
            acc = acc + w1[j * IN + i] * x[i]
        h.append(acc if acc > 0.0 else 0.0)
    y = []
    for k in range(OUT):
        acc = b2[k]
        for j in range(HID):
            acc = acc + w2[k * HID + j] * h[j]
        y.append(acc)
    return y


def main():
    path = sys.argv[1]
    w1, b1, w2, b2, x = weights()
    if len(sys.argv) > 2 and sys.argv[2] == "--oracle":
        for v in forward(w1, b1, w2, b2, x):
            bits = struct.unpack("<Q", struct.pack("<d", v))[0]
            print(f"{bits >> 32} {bits % (1 << 32)}")
        return
    with open(path, "wb") as f:
        for arr in (w1, b1, w2, b2, x):
            for v in arr:
                f.write(struct.pack("<d", v))


if __name__ == "__main__":
    main()
