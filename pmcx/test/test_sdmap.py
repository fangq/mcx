"""cfg['sdmap'] selects the channels of the adjoint Jacobians: each row [srcid, detid]
(1-based) becomes one channel, in that order, instead of all Ns*Nd pairs, and only the
sources and detectors of those channels are simulated; cfg['detdir']
may have 3 columns (no focal length) or 4."""

import re
import subprocess
import sys
import unittest

SCRIPT = r"""
import numpy as np
import pmcx as pmcx
try:
    if not pmcx.gpuinfo():
        print("SKIP")
        raise SystemExit(0)
except Exception:
    print("SKIP")
    raise SystemExit(0)
N = 30
cfg = dict(nphoton=3000000, vol=np.ones((N, N, N), np.uint8), tstart=0, tend=5e-9, tstep=5e-9,
           prop=np.array([[0, 0, 1, 1], [0.01, 1, 0.01, 1.37]]), issrcfrom0=1, seed=1234,
           srcpos=[[15, 15, 0], [10, 15, 0]], srcdir=[[0, 0, 1], [0, 0, 1]],
           detpos=[[15, 10, 0, 2], [15, 20, 0, 2], [20, 15, 0, 2]], detdir=np.tile([0, 0, 1, 0], (3, 1)),
           outputtype="adjoint")
full = pmcx.run(cfg)["jmua"]
sdmap = np.array([[2, 3], [1, 1], [2, 1]])
cfg["sdmap"] = sdmap
sub = pmcx.run(cfg)["jmua"]
print("SHAPES %d %d" % (full.shape[3], sub.shape[3]))
err = 0.0
for c, (s, d) in enumerate(sdmap):
    a = full[..., (s - 1) * 3 + d - 1]
    err = max(err, np.linalg.norm(a - sub[..., c]) / np.linalg.norm(a))
print("RELERR %g" % err)
cfg["sdmap"] = [[3, 1]]
try:
    pmcx.run(cfg)
    print("RANGE no error")
except Exception:
    print("RANGE error")
# a 3-column detdir has a 0 focal length, the same as the 4-column one above
del cfg["sdmap"]
cfg["detdir"] = np.tile([0, 0, 1], (3, 1))
j3 = pmcx.run(cfg)["jmua"]
print("DETDIR3 %g" % (np.linalg.norm(j3 - full) / np.linalg.norm(full)))
cfg["detdir"] = np.tile([0, 0], (3, 1))
try:
    pmcx.run(cfg)
    print("DETDIR2 no error")
except Exception:
    print("DETDIR2 error")
"""


class TestSDMap(unittest.TestCase):
    def test_sdmap_channels(self):
        out = subprocess.run(
            [sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=600
        )
        if re.search(r"^SKIP$", out.stdout, re.M):
            self.skipTest("no GPU available")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("SHAPES 6 3", out.stdout)
        # the sdmap run only simulates the slots of its sources and detectors, so the photons
        # of the two runs differ: they agree within the Monte Carlo noise
        self.assertLess(float(re.search(r"RELERR (\S+)", out.stdout).group(1)), 0.05)
        self.assertIn("RANGE error", out.stdout)
        # 3-column detdir: same Jacobians as [vx vy vz 0], up to run-to-run noise
        self.assertLess(float(re.search(r"DETDIR3 (\S+)", out.stdout).group(1)), 0.05)
        self.assertIn("DETDIR2 error", out.stdout)


if __name__ == "__main__":
    unittest.main()
