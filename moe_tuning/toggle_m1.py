#!/usr/bin/env python3
"""toggle_m1.py -- enable/disable the m==1 branch in the container's
layers/utils.py, to bisect whether that patch is what breaks MM=1 (vision on).

  python3 toggle_m1.py off   # remove the branch (keep a backup)
  python3 toggle_m1.py on    # restore it
  python3 toggle_m1.py st    # show current state

Purely textual, marker-delimited, re-parsed with ast before writing.
"""
import ast
import os
import shutil
import sys

F = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py"
BAK = F + ".m1bak"

BEGIN = "    # A scalar projection (m == 1) is a dot product, not a GEMM."
ENDMARK = "        return acc.to(x.dtype).reshape(*x.shape[:-1], weight.shape[0])\n"


def read():
    return open(F).read()


def state(s):
    return "ON " if BEGIN in s else "OFF"


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "st"
    s = read()
    print("current:", state(s))
    if cmd == "st":
        return
    if cmd == "off":
        if BEGIN not in s:
            print("already off"); return
        if not os.path.exists(BAK):
            shutil.copy2(F, BAK)
            print("backup ->", BAK)
        i = s.index(BEGIN)
        j = s.index(ENDMARK, i) + len(ENDMARK)
        # also swallow the blank line we inserted after the block
        while j < len(s) and s[j] == "\n" and s[j - 1] == "\n":
            j += 1
        new = s[:i] + s[j:]
    elif cmd == "on":
        if BEGIN in s:
            print("already on"); return
        if not os.path.exists(BAK):
            print("no backup to restore from"); return
        new = open(BAK).read()
        print("restored from backup")
    else:
        print("usage: toggle_m1.py on|off|st"); return
    ast.parse(new)                      # never write a file that does not parse
    open(F, "w").write(new)
    print("after  :", state(read()), "| m==1 occurrences:", read().count("m == 1"))


if __name__ == "__main__":
    main()
