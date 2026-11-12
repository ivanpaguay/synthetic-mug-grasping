# =========================================================================
# export_houdini_code  v1   (Houdini: Python Shell or shelf tool)
# Pulls all hand-written code out of a .hip, for this repository.
# READ ONLY: it changes no node and does not save the .hip.
#
# It walks every node in the scene and exports each non-empty code parameter:
#   snippet  (SOP/DOP/LOP/COP wrangles, VEX)              -> .vfl
#   python   (Python SOP)                                  -> .py
#   script / pythonscript / code (TOPs and others, Python) -> .py
# It writes to OUT_DIR:
#   <network>/<node>__<parm>.vfl|.py   with a header (path, type, inputs)
#   index.csv          path, type, parm, file, lines, absolute paths
#   hdas.txt           loaded HDAs that are NOT from SideFX (your own)
#   absolute_paths.txt lines with absolute paths (I:\..., C:\...) to review
#
# Usage: set OUT_DIR, open the pipeline .hip, then in Houdini's Python Shell
# (Windows > Python Shell):  exec(open(r"<repo>/tools/export_houdini_code.py").read())
# Repeat for each .hip.
# =========================================================================
import csv
import os
import re

import hou

OUT_DIR = r"C:\path\to\repo_export\houdini_code"   # set before running

CODE_PARMS = {"snippet": ".vfl", "python": ".py", "script": ".py", "pythonscript": ".py", "code": ".py"}
ABS_PATH = re.compile(r"(?i)\b[a-z]:[\\/]")


def safe(s):
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip("/"))
    return s[:120] or "root"


def export():
    hip = os.path.splitext(os.path.basename(hou.hipFile.path()))[0]
    base = os.path.join(OUT_DIR, safe(hip))
    os.makedirs(base, exist_ok=True)
    rows, abs_lines = [], []
    for node in hou.node("/").allSubChildren(recurse_in_locked_nodes=False):
        for pname, ext in CODE_PARMS.items():
            p = node.parm(pname)
            if p is None:
                continue
            try:
                code = p.unexpandedString()
            except Exception:
                continue
            if not code or not code.strip() or "\n" not in code and len(code) < 20:
                continue
            parent = node.parent().path() if node.parent() else "/"
            d = os.path.join(base, safe(parent))
            os.makedirs(d, exist_ok=True)
            fn = os.path.join(d, "%s__%s%s" % (safe(node.name()), pname, ext))
            ins = ", ".join(i.path() for i in node.inputs() if i is not None)
            cm = "//" if ext == ".vfl" else "#"
            header = ("%s node: %s\n%s type:  %s\n%s parm:  %s\n%s inputs: %s\n%s exported from %s.hip\n\n"
                      % (cm, node.path(), cm, node.type().nameWithCategory(), cm, pname, cm, ins or "-", cm, hip))
            with open(fn, "w", encoding="utf-8") as fh:
                fh.write(header + code.rstrip() + "\n")
            n_abs = 0
            for k, line in enumerate(code.splitlines(), 1):
                if ABS_PATH.search(line):
                    n_abs += 1
                    abs_lines.append("%s:%d  %s" % (node.path(), k, line.strip()))
            rows.append({"node": node.path(), "type": node.type().nameWithCategory(), "parm": pname,
                         "file": os.path.relpath(fn, base), "lines": code.count("\n") + 1, "abs_paths": n_abs})
    with open(os.path.join(base, "index.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["node", "type", "parm", "file", "lines", "abs_paths"])
        w.writeheader()
        w.writerows(rows)
    hfs = os.path.normcase(os.path.abspath(hou.text.expandString("$HFS")))
    own = sorted({f for f in hou.hda.loadedFiles()
                  if f and not os.path.normcase(os.path.abspath(hou.text.expandString(f))).startswith(hfs)})
    with open(os.path.join(base, "hdas.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(own) + ("\n" if own else ""))
    with open(os.path.join(base, "absolute_paths.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(abs_lines) + ("\n" if abs_lines else ""))
    print("export_houdini_code: %d code blocks, %d own HDAs, %d lines with absolute paths -> %s"
          % (len(rows), len(own), len(abs_lines), base))


export()
