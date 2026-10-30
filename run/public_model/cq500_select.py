"""Series selection identical to the locked CQ500 evaluation (research_cq500_extract.py, choose())."""
def choose(z):
    ser = {}
    for n in z.namelist():
        if n.endswith('.dcm'):
            ser.setdefault(n.rsplit('/', 1)[0], []).append(n)
    def nm(k): return k.rsplit('/', 1)[-1].upper()
    ok = {k: v for k, v in ser.items() if not any(w in nm(k) for w in ['POST', 'CONTRAST']) or 'PRE' in nm(k)}
    bone = [k for k in ok if 'BONE' in nm(k)]
    soft = [k for k in ok if 'BONE' not in nm(k)]
    def score(k):
        t = nm(k); return (('THIN' in t), abs(len(ok[k]) - 32))
    soft.sort(key=score)
    bone.sort(key=lambda k: (('THIN' in nm(k)), abs(len(ok[k]) - 32)))
    return (soft[0] if soft else None), (bone[0] if bone else None), {k: len(v) for k, v in ser.items()}, ok
