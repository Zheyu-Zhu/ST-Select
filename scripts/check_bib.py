"""Audit main.tex citations against main.bib.

Written as a file, not a heredoc: LaTeX control sequences inside a <<'PY'
heredoc get eaten by Python's string escaping (\\u in \\usepackage is a
truncated unicode escape).

Entries are split on the blank line between them rather than on "\\n}", because
that pattern swallowed the newline terminating each entry's final field -- which
is almost always `year`, so the first version of this script reported 20 of 22
entries as missing a year they in fact had.
"""
import collections
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
D = r"C:\Users\zheyu\OneDrive\Desktop\SPIE2027"
tex = open(D + r"\main.tex", encoding="utf-8").read()
bib = open(D + r"\main.bib", encoding="utf-8").read()

cited = []
for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
    cited += [k.strip() for k in m.group(1).split(",")]
keys = re.findall(r"@\w+\{([^,]+),", bib)
ncite = len(re.findall(r"[\\]cite", tex))
print(f"cite commands {ncite}, distinct keys cited {len(set(cited))}, "
      f"bib entries {len(keys)}")
print("duplicate bib keys :", [k for k, c in collections.Counter(keys).items() if c > 1] or "none")
print("cited but missing  :", sorted(set(cited) - set(keys)) or "none")
print("bib but never cited:", sorted(set(keys) - set(cited)) or "none")

VENUE = ("journal", "booktitle", "publisher", "howpublished", "note", "school",
         "institution", "organization", "series")

print("\n=== entry hygiene ===")
bad = 0
for chunk in re.split(r"\n(?=@)", bib):
    m = re.match(r"@(\w+)\{([^,]+),", chunk)
    if not m:
        continue
    typ, key = m.group(1), m.group(2)
    f = {a.lower(): b for a, b in
         re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*$", chunk, re.S | re.M)}
    miss = [x for x in ("title", "author", "year") if x not in f]
    venue = next((f[v] for v in VENUE if v in f), None)
    if miss or not venue:
        print(f"  {key:26s} type={typ:14s} missing={miss or '-'} "
              f"venue={'NONE' if not venue else 'ok'}")
        bad += 1
    if re.search(r"contentReference|oaicite", chunk):
        print(f"  {key:26s} !! leftover artifact text")
        bad += 1
print(f"  {bad} entries need attention" if bad
      else f"  all {len(keys)} entries carry title/author/year/venue")

print("\n=== preamble ===")
pre = tex[:tex.index("[b]egin{document}".replace("[b]", "\\b"))]
pk = re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}", pre, re.S)
flat = [p.strip() for g in pk for p in g.split(",")]
print("  packages:", len(flat), "| duplicated:",
      sorted({p for p, c in collections.Counter(flat).items() if c > 1}) or "none")
print("  geometry present:", "geometry" in flat)
