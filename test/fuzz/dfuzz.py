#!/usr/bin/env python3
"""
dfuzz.py - mutation fuzzer for mcx: mutated test/builtin inputs and command-line options

usage:
    dfuzz.py --bin MCX [options]                 crash/hang/sanitizer fuzzing of one binary
    dfuzz.py --bin MCX --ref REFMCX [options]    differential: compare against a reference build

Each case mutates a builtin JSON input field by field (type swaps, extreme numbers, wrong
array shapes, deleted keys) and in ways that reach the output accumulation (output types,
pattern and RF sources, time gates), adds random command-line flags, and runs it. A case is
reported when a binary crashes, hangs or prints a sanitizer error; in differential mode also
when the two binaries exit differently, or, in deterministic runs (-A 0, fixed thread count,
same photons), print different absorbed fractions or output totals (> --tol relative).
Reported cases are saved under OUT/caseNNNN with the input, the arguments and both logs.

Build the binaries with -fsanitize=address,undefined to catch memory errors, e.g.
    make CPPFLAGS='-g -Wall -fsanitize=address,undefined' \\
         CUGENCODE='-arch=sm_75 -Xcompiler -fsanitize=address -Xcompiler -fsanitize=undefined' \\
         USERLINKOPT='lib/libzmat.a -lpthread -fsanitize=address,undefined'
"""
import argparse, os, json, random, subprocess, glob, shutil, re, base64, zlib

ap = argparse.ArgumentParser(description='mutation fuzzer for mcx')
ap.add_argument('--bin', required=True, help='mcx binary to fuzz')
ap.add_argument('--ref', help='reference mcx binary (differential mode)')
ap.add_argument('--builtin', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'builtin'),
                help='folder of seed inputs [../builtin]')
ap.add_argument('-n', '--cases', type=int, default=100, help='number of cases [100]')
ap.add_argument('--seed', type=int, default=1, help='fuzzer RNG seed [1]')
ap.add_argument('--out', default='fuzzout', help='output folder [fuzzout]')
ap.add_argument('--timeout', type=float, default=90, help='seconds per run [90]')
ap.add_argument('--tol', type=float, default=1e-3, help='relative tolerance of output totals [1e-3]')
opt = ap.parse_args()

rng = random.Random(opt.seed)
os.makedirs(opt.out, exist_ok=True)
env = dict(os.environ, ASAN_OPTIONS=os.environ.get('ASAN_OPTIONS', 'protect_shadow_gap=0:detect_leaks=0'),
           UBSAN_OPTIONS=os.environ.get('UBSAN_OPTIONS', 'print_stacktrace=1'))
seeds = [f for f in sorted(glob.glob(os.path.join(opt.builtin, '*.json'))) if os.path.getsize(f) < 3e6 and 'atlas' not in f]
BINS = {'bin': os.path.abspath(opt.bin)}

if opt.ref:
    BINS['ref'] = os.path.abspath(opt.ref)

ODD = [0, -1, 1, 1e-30, -1e-30, 1e30, -1e30, 2 ** 31, 2 ** 32 + 1, 2 ** 63, 0.5, 1e-6, 1e6, 3]
OTYPES = ['flux', 'fluence', 'energy', 'jacobian', 'wp', 'dcs', 'rf', 'length', 'rfmus', 'wltof', 'wptof', 'x', 'f', 'e',
          'adjoint', 'garbage', 7]
FLAGS = [['-O', list('xfejprlstbadcuvwq') + ['XX', '9']], ['-X', ['0', '1', '2']], ['-A', ['0', '1', '2']],
         ['-t', ['0', '64', '1000', '4096', '-1', '1e9']], ['-T', ['1', '32', '64', '128', '0']], ['-b', ['0', '1']],
         ['-B', ['aaaaaa', 'rrrrrr', 'mmmmmm', 'cccccc', '__abc_', 'x']], ['-u', ['1', '0.005', '1e-6', '1e6', '-1']],
         ['-e', ['0', '0.5', '0.9', '1e-9']], ['-r', ['1', '2', '-2', '0']], ['-d', ['0', '1']],
         ['-w', ['DP', 'DSPMXVW', 'M', 'zz']], ['-U', ['0', '1']], ['-q', ['0', '1']], ['-V', ['0', '1']],
         ['-k', ['0', '1']], ['-Y', ['0', '-1', '1']], ['-S', ['0', '1']]]


def paths(o, p=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from paths(v, p + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o[:8]):
            yield from paths(v, p + (i,))
    yield p


def get(o, p):
    for k in p:
        o = o[k]
    return o


def mutate_json(d):
    for _ in range(rng.randint(1, 3)):
        ps = [p for p in paths(d) if p]
        if not ps:
            break
        p = rng.choice(ps)
        try:
            v, par, r = get(d, p), get(d, p[:-1]), rng.random()
            if r < 0.3 and isinstance(v, (int, float)) and not isinstance(v, bool):
                par[p[-1]] = rng.choice(ODD) if rng.random() < 0.5 else v * rng.choice([0, -1, 2, 10, 0.1, 1e3])
            elif r < 0.45:
                par[p[-1]] = rng.choice(['abc', '', [], {}, None, True, [1, 2], -5])
            elif r < 0.6 and isinstance(v, list):
                par[p[-1]] = v[:rng.randint(0, len(v))] if v and rng.random() < 0.5 else v + [rng.choice(ODD)]
            elif r < 0.7 and isinstance(par, dict):
                del par[p[-1]]
        except Exception:
            pass

    sess = d.get('Session') if isinstance(d.get('Session'), dict) else None
    optode = d.get('Optode') if isinstance(d.get('Optode'), dict) else {}
    src = optode.get('Source') if isinstance(optode.get('Source'), dict) else None

    if sess is not None and rng.random() < 0.3:
        sess['OutputType'] = rng.choice(OTYPES)

    if src is not None and rng.random() < 0.25:
        src['Frequency'] = rng.choice([1e8, 1e9, 0, -1e8, 1e12])

    if src is not None and rng.random() < 0.15:
        n = rng.choice([1, 2, 3])
        src.update({'Type': rng.choice(['pattern', 'pattern3d']), 'Param1': [20, 0, 0, 2], 'Param2': [0, 20, 0, 2], 'SrcNum': n,
                    'Pattern': {'Nx': 2, 'Ny': 2, 'Nz': 1, 'Data': [rng.choice([0, 1, 2, 255, -1, 1e6]) for _ in range(4 * n)]}})

    if isinstance(d.get('Forward'), dict) and rng.random() < 0.15:
        d['Forward'].update(rng.choice([{'T1': 1e-6}, {'Dt': 1e-12}, {'T1': 5e-11, 'Dt': 1e-11}, {'T0': 1e-9}]))

    return d


def mutate_cli(det):
    a = []

    for _ in range(rng.randint(0, 4)):
        f, vals = rng.choice(FLAGS)

        if not (det and f in ('-A', '-t', '-T', '-X')):
            a += [f, rng.choice(vals)]

    return a


def total(fn):
    """sum of the saved volume, None if absent or undecodable"""
    try:
        import numpy as np
        a = json.load(open(fn))['NIFTIData']
        dt = {'single': np.float32, 'double': np.float64}[a['_ArrayType_']]
        return float(np.frombuffer(zlib.decompress(base64.b64decode(a['_ArrayZipData_'])), dtype=dt).astype(np.float64).sum())
    except Exception:
        return None


def run(exe, args, wd):
    try:
        p = subprocess.run([exe] + args, cwd=wd, env=env, capture_output=True, timeout=opt.timeout)
        txt, rc = (p.stdout + p.stderr).decode('latin1'), p.returncode
    except subprocess.TimeoutExpired:
        txt, rc = 'TIMEOUT', 'timeout'

    txt = re.sub(r'\x1b\[[0-9;]*m', '', txt)
    return {'rc': rc, 'san': [l for l in txt.splitlines() if 'Sanitizer' in l or 'runtime error' in l],
            'err': [e[0] + e[1].split(' in unit')[0][:80] for e in re.findall(r'MCX ERROR\(?(-?\d+)\)?:?([^\n]*)', txt)],
            'abs': re.findall(r'absorbed:\s*([0-9.]+)%', txt), 'sum': total(os.path.join(wd, 'o.jnii')), 'log': txt}


def bad(r):
    return r['rc'] == 'timeout' or (isinstance(r['rc'], int) and r['rc'] < 0) or bool(r['san'])


stats = {}

for c in range(opt.cases):
    seedfile = rng.choice(seeds)
    d = mutate_json(json.load(open(seedfile)))
    det = rng.random() < 0.6
    args = ['-f', 'in.json', '-n', rng.choice(['1e3', '1e4', '3e4']), '-E', '1648335518', '-F', 'jnii', '-s', 'o'] + mutate_cli(det)

    if det:
        args += ['-A', '0', '-t', '4096', '-T', '64']

    res = {}

    for k, exe in BINS.items():
        wd = os.path.join(opt.out, 'w_' + k)
        shutil.rmtree(wd, ignore_errors=True)
        os.makedirs(wd)
        json.dump(d, open(os.path.join(wd, 'in.json'), 'w'))
        res[k] = run(exe, args, wd)

    B, R, found = res['bin'], res.get('ref'), []

    if bad(B):
        found.append(('crash/hang/sanitizer' + (' (also in ref)' if R and bad(R) else '')) + ': rc=%s %s' % (B['rc'], B['san'][:2]))

    if R:
        saveref = '-X' in args and args[args.index('-X') + 1] != '0'    # -X is not reproducible run to run

        if (B['rc'] == 0) != (R['rc'] == 0) or B['err'] != R['err']:
            found.append('exit/error differs: ref %s %s | bin %s %s' % (R['rc'], R['err'][:1], B['rc'], B['err'][:1]))

        if det and not saveref and B['rc'] == 0 and R['rc'] == 0:
            if B['abs'] != R['abs']:
                found.append('absorbed differs: ref %s | bin %s' % (R['abs'][:2], B['abs'][:2]))

            if B['sum'] is not None and R['sum'] is not None and abs(B['sum'] - R['sum']) > opt.tol * max(abs(R['sum']), 1e-30):
                found.append('output total differs: ref %.6g | bin %.6g' % (R['sum'], B['sum']))

    key = found[0].split(':')[0] if found else 'ok'
    stats[key] = stats.get(key, 0) + 1

    if found:
        cd = os.path.join(opt.out, 'case%04d' % c)
        os.makedirs(cd, exist_ok=True)
        json.dump(d, open(os.path.join(cd, 'in.json'), 'w'))
        open(os.path.join(cd, 'args.txt'), 'w').write(' '.join(args) + '\n' + os.path.basename(seedfile) + '\n' + '\n'.join(found) + '\n')

        for k, r in res.items():
            open(os.path.join(cd, k + '.log'), 'w').write(r['log'][-4000:])

    print('%4d %-45s %-4s %s' % (c, os.path.basename(seedfile)[:45], 'det' if det else 'auto', '; '.join(found) or 'ok'), flush=True)

print('SUMMARY', json.dumps(stats))
