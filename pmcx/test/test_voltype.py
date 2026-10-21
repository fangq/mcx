"""Run the same label volume stored as every supported numpy dtype, with
detectors, several times in one process. uint32 and float32 volumes used to
pass an invalid media format to the kernel (NaN fluence or a hang)."""

import re
import subprocess
import sys
import unittest

SCRIPT = r'''
import numpy as np
import pmcx as pmcx
try:
    if not pmcx.gpuinfo():
        print("SKIP")
        raise SystemExit(0)
except Exception:
    print("SKIP")
    raise SystemExit(0)
N = 20
lab = np.ones((N, N, N), np.uint8)
lab[:, :, 10:] = 2
prop = np.array([[0, 0, 1, 1], [0.01, 1, 0.9, 1.37], [0.02, 2, 0.8, 1.37]], np.float32)
sums = []
for dtype in ["uint8", "uint16", "uint32", "int32", "float32", "float64"]:
    cfg = dict(nphoton=100000, vol=lab.astype(dtype), tstart=0, tend=5e-9, tstep=5e-9,
               srcpos=[10, 10, 0], srcdir=[0, 0, 1], prop=prop, detpos=[[12, 10, 0, 2], [8, 10, 0, 2]],
               isreflect=1, seed=11, autopilot=1)
    res = pmcx.run(cfg)
    flux = np.asarray(res["flux"])
    assert not np.isnan(flux).any(), dtype + " returns NaN"
    sums.append(flux.sum())
sums = np.array(sums)
print("RELDIFF %g" % (np.abs(sums - sums[0]).max() / sums[0]))
'''


class TestVolumeDtypes(unittest.TestCase):
    def test_label_dtypes_with_detectors(self):
        out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=600)
        if re.search(r"^SKIP$", out.stdout, re.M):
            self.skipTest("no GPU available")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        reldiff = float(re.search(r"RELDIFF (\S+)", out.stdout).group(1))
        self.assertLess(reldiff, 0.02)


if __name__ == "__main__":
    unittest.main()
