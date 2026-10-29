"""Unit tests of the pmcx binding interface (run, gpuinfo, version and the mcxlab
front end): the cfg parsing, the errors, the output keys, shapes and types, and short
absorbed-fraction end-to-end checks. Each simulation is small (a 20^3 domain, 1e4-1e5
photons), so the suite takes a few seconds on a GPU; it is skipped without a device.

Run with: python3 -m unittest discover -s pmcx/test
"""

import contextlib
import os
import subprocess
import sys
import unittest

import numpy as np

import pmcx as pmcx

BINDING = pmcx.__name__  # the same file tests pmcxcl, with the import above changed

try:
    HAS_BINARY = isinstance(
        pmcx.version(), str
    )  # the noarch pmcx wheel has no simulator
except Exception:
    HAS_BINARY = False

try:
    HAS_DEVICE = HAS_BINARY and len(pmcx.gpuinfo()) > 0
except Exception:
    HAS_DEVICE = False

N = 20
PROP = np.array([[0, 0, 1, 1], [0.01, 1, 0.9, 1.37]], np.float32)
PROP2 = np.array([[0, 0, 1, 1], [0.01, 1, 0.9, 1.37], [0.05, 5, 0.8, 1.37]], np.float32)

# absorbed fractions of the cases in TestEndToEnd, from 1e7 photons with pmcx and pmcxcl
# (they agree within 5e-5); the 1e5-photon runs below are within 0.01 of these
REF_HOMOGENEOUS = 0.2440
REF_TWOLAYER = 0.3972


def cfg(**kw):
    """a 20^3 homogeneous cube with a pencil beam; keywords override or add keys"""
    c = dict(
        nphoton=10000,
        vol=np.ones((N, N, N), np.uint8),
        tstart=0,
        tend=5e-9,
        tstep=5e-9,
        srcpos=[10, 10, 0],
        srcdir=[0, 0, 1],
        prop=PROP,
        issrcfrom0=1,
        seed=12345,
        flog="NUL" if sys.platform.startswith("win") else "/dev/null",
    )
    c.update(kw)
    return c


def absorbed(res):
    return res["stat"]["energyabs"] / res["stat"]["energytot"]


def assert_same_flux(test, a, b, rtol=1e-3):
    """the same simulation, up to the float rounding of the atomic accumulation order"""
    test.assertEqual(a.shape, b.shape)
    a, b = a.astype(np.float64), b.astype(np.float64)
    test.assertAlmostEqual(
        a.sum() / b.sum(), 1, delta=1e-4
    )  # another photon set differs by ~1e-2
    test.assertLessEqual(np.abs(a - b).sum(), rtol * np.abs(a).sum())


@unittest.skipUnless(HAS_BINARY, "the binary module is not installed")
class TestInfo(unittest.TestCase):
    def test_version(self):
        ver = pmcx.version()
        self.assertIsInstance(ver, str)
        self.assertTrue(ver.startswith("v"), ver)

    @unittest.skipUnless(HAS_DEVICE, "no device")
    def test_gpuinfo(self):
        info = pmcx.gpuinfo()
        self.assertIsInstance(info, list)
        for key in (
            "name",
            "id",
            "devcount",
            "major",
            "minor",
            "globalmem",
            "core",
            "autothread",
            "maxgate",
        ):
            self.assertIn(key, info[0])
        self.assertEqual(info[0]["devcount"], len(info))

    def test_docstrings(self):
        for name in ("run", "gpuinfo", "version"):
            self.assertIn("Returns", getattr(pmcx, name).__doc__ or "", name)


@unittest.skipUnless(HAS_DEVICE, "no device")
class TestErrors(unittest.TestCase):
    """invalid settings raise Python exceptions, not crashes or silent runs"""

    def run_without(self, key):
        c = cfg()
        del c[key]
        return pmcx.run(c)

    def test_missing_vol(self):
        with self.assertRaises(ValueError):
            self.run_without("vol")

    def test_missing_prop(self):
        with self.assertRaises((ValueError, RuntimeError)):
            self.run_without("prop")

    def test_bad_values(self):
        bad = [
            (TypeError, dict(vol=np.ones((N, N, N), np.complex64))),
            (ValueError, dict(prop=[[0, 0, 1]])),
            (ValueError, dict(debuglevel="Q")),
            (ValueError, dict(gpuid="")),
            (ValueError, dict(outputtype="xyz")),
            (ValueError, dict(srctype="xyz")),
            (
                ValueError,
                dict(seed=np.zeros((3, 4), np.uint8)),
            ),  # a replay seed of the wrong size
        ]
        for err, kw in bad:
            with self.subTest(kw=list(kw)):
                with self.assertRaises(err):
                    pmcx.run(cfg(**kw))


@unittest.skipUnless(HAS_DEVICE, "no device")
class TestInterface(unittest.TestCase):
    def test_dict_and_keywords(self):
        """run(cfg) and run(**cfg) are the same simulation"""
        a = pmcx.run(cfg(autopilot=0))["flux"]
        b = pmcx.run(**cfg(autopilot=0))["flux"]
        assert_same_flux(self, a, b)

    def test_array_order(self):
        """C- and Fortran-ordered volumes give the same result, vol[x, y, z] being voxel (x, y, z)"""
        vol = np.ones((N, N + 2, N + 4), np.uint8)
        vol[:, :, 8:] = 2
        a = pmcx.run(cfg(vol=np.ascontiguousarray(vol), prop=PROP2, autopilot=0))[
            "flux"
        ]
        b = pmcx.run(cfg(vol=np.asfortranarray(vol), prop=PROP2, autopilot=0))["flux"]
        self.assertEqual(a.shape, (N, N + 2, N + 4, 1))
        assert_same_flux(self, a, b)
        # the beam enters at z=0, so the fluence decays along the 3rd axis
        self.assertGreater(a[10, 10, 2, 0], a[10, 10, 15, 0])

    def test_label_dtypes(self):
        """a label volume stored as any supported dtype is the same simulation"""
        ref = pmcx.run(cfg(autopilot=0))["flux"]
        for dtype in (
            "int8",
            "uint16",
            "int16",
            "uint32",
            "int32",
            "float32",
            "float64",
        ):
            with self.subTest(dtype=dtype):
                f = pmcx.run(cfg(vol=np.ones((N, N, N), dtype), autopilot=0))["flux"]
                assert_same_flux(self, ref, f)

    def test_cfg_unchanged(self):
        c = cfg(detpos=[[10, 8, 0, 2]])
        before = {
            k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in c.items()
        }
        pmcx.run(c)
        self.assertEqual(set(c), set(before))
        for k in c:
            np.testing.assert_array_equal(c[k], before[k], err_msg=k)

    def test_time_gates_and_stat(self):
        res = pmcx.run(cfg(tstep=1e-9))
        self.assertEqual(res["flux"].shape, (N, N, N, 5))
        self.assertEqual(res["flux"].dtype, np.float32)
        for key in (
            "runtime",
            "nphoton",
            "energytot",
            "energyabs",
            "normalizer",
            "unitinmm",
            "workload",
        ):
            self.assertIn(key, res["stat"])
        self.assertEqual(res["stat"]["nphoton"], 10000)

    def test_energy_output_is_absorbed_fraction(self):
        """the normalized 'energy' output adds up to energyabs/energytot"""
        res = pmcx.run(cfg(outputtype="energy"))
        self.assertAlmostEqual(res["flux"].sum(), absorbed(res), places=4)

    def test_detected_photons(self):
        """detp rows follow savedetflag; seeds and detp have one column per photon"""
        res = pmcx.run(cfg(detpos=[[10, 8, 0, 2]], issaveseed=1, savedetflag="dspmxvw"))
        ncol = 1 + 3 * 1 + 3 + 3 + 1  # detid, nscat/ppath/mom per medium, p, v, w0
        ndet = res["detp"].shape[1]
        self.assertGreater(ndet, 0)
        self.assertEqual(res["detp"].shape, (ncol, ndet))
        self.assertEqual(res["seeds"].dtype, np.uint8)
        self.assertEqual(res["seeds"].shape[1], ndet)
        np.testing.assert_array_equal(res["detp"][0], 1)  # detector id
        self.assertTrue(np.all(res["detp"][2] > 0))  # partial path in medium 1
        # default 'dp': detid and the partial path
        self.assertEqual(pmcx.run(cfg(detpos=[[10, 8, 0, 2]]))["detp"].shape[0], 2)

    def test_no_fluence(self):
        res = pmcx.run(cfg(issave2pt=0, detpos=[[10, 8, 0, 2]]))
        self.assertNotIn("flux", res)
        self.assertIn("detp", res)

    def test_trajectories(self):
        res = pmcx.run(cfg(nphoton=100, debuglevel="M"))
        traj = res["traj"]
        self.assertEqual(traj.shape[0], 6)
        self.assertGreater(
            len(np.unique(traj[0].view(np.uint32))), 1
        )  # photon ids, as uint32 bits
        pos = traj[1:4]
        self.assertTrue(np.all(pos >= -1e-3) and np.all(pos <= N + 1e-3))

    def test_reflectance_and_mask(self):
        shapes = '{"Shapes":[{"Sphere":{"O":[10,10,10],"R":4,"Tag":2}}]}'
        vol = np.ones((N, N, N), np.uint8)
        vol[
            :, :, 0
        ] = 0  # the reflectance is recorded in the background voxels next to the domain
        res = pmcx.run(
            cfg(vol=vol, issaveref=1, dumpmask=True, shapes=shapes, prop=PROP2)
        )
        self.assertEqual(res["dref"].shape, res["flux"].shape)
        self.assertGreater(res["dref"].sum(), 0)
        self.assertEqual(res["vol"].dtype, np.uint32)
        self.assertEqual(
            res["vol"][10, 10, 10], 2
        )  # the sphere is drawn into the label volume
        self.assertEqual(res["vol"][1, 1, 1], 1)

    def test_rf_is_complex(self):
        res = pmcx.run(cfg(omega=2 * np.pi * 1e8))
        self.assertTrue(np.iscomplexobj(res["flux"]))
        self.assertEqual(res["flux"].shape, (N, N, N, 1))

    def test_pattern_sources(self):
        """several patterns are interleaved along the 1st axis of the raw output"""
        pat = np.ones((2, 4, 4), np.float32)
        pat[1] = 0.5
        res = pmcx.run(
            cfg(
                srctype="pattern",
                srcpos=[5, 5, -0.01],
                srcparam1=[8, 0, 0, 4],
                srcparam2=[0, 8, 0, 4],
                srcpattern=pat,
                srcnum=2,
            )
        )
        self.assertEqual(res["flux"].shape, (2 * N, N, N, 1))

    def test_replay(self):
        """replaying the detected photons of a run gives a Jacobian"""
        base = pmcx.run(cfg(nphoton=20000, detpos=[[10, 8, 0, 2]], issaveseed=1))
        ndet = base["detp"].shape[1]
        jac = pmcx.run(
            cfg(
                seed=base["seeds"],
                detphotons=base["detp"],
                outputtype="jacobian",
                detpos=[[10, 8, 0, 2]],
            )
        )
        self.assertEqual(jac["flux"].shape, (N, N, N, 1))
        self.assertTrue(np.all(np.isfinite(jac["flux"])))
        self.assertGreater(np.abs(jac["flux"]).sum(), 0)
        self.assertGreater(ndet, 0)

    def test_flog_null_device_is_silent(self):
        """'NUL' and '/dev/null' are the null device on every OS: nothing is printed, not even the
        cfg echo or the progress bar, and no file is written, also from a read-only folder
        (fangq/mcx#280); without flog, the log is printed"""
        import tempfile

        def run(flog):
            script = (
                "import sys, numpy as np; sys.path[:0] = %r; import %s as p; "
                "c = dict(nphoton=1000, vol=np.ones((10, 10, 10), np.uint8), tend=5e-9, srcpos=[5, 5, 0], "
                "srcdir=[0, 0, 1], prop=[[0, 0, 1, 1], [0.01, 1, 0.9, 1.37]], debuglevel='P'); "
                "c.update(%r); p.run(c)"
                % ([os.path.abspath(d) for d in sys.path], BINDING, flog)
            )
            with tempfile.TemporaryDirectory() as tmp:
                os.chmod(tmp, 0o555)
                try:
                    out = subprocess.run(
                        [sys.executable, "-c", script],
                        capture_output=True,
                        text=True,
                        timeout=300,
                        cwd=tmp,
                    )
                    files = os.listdir(tmp)
                finally:
                    os.chmod(tmp, 0o755)
            self.assertEqual(out.returncode, 0, out.stderr)
            return out.stdout + out.stderr, files

        for name in ("NUL", "nul", "/dev/null"):
            with self.subTest(flog=name):
                text, files = run({"flog": name})
                self.assertEqual(text.strip(), "")
                self.assertEqual(files, [])
        self.assertNotEqual(run({})[0].strip(), "")  # the default still prints


@unittest.skipUnless(HAS_DEVICE, "no device")
class TestMcxlab(unittest.TestCase):
    """the post-processing of the mcxlab front end"""

    def test_named_outputs(self):
        res = pmcx.mcxlab(
            cfg(
                detpos=[[10, 8, 0, 2]],
                issaveseed=1,
                savedetflag="dspmxvw",
                debuglevel="M",
                nphoton=2000,
            )
        )
        detp = res["detp"]
        ndet = detp["detid"].shape[0]
        for key, shape in (
            ("nscat", (ndet, 1)),
            ("ppath", (ndet, 1)),
            ("mom", (ndet, 1)),
            ("p", (ndet, 3)),
            ("v", (ndet, 3)),
        ):
            self.assertEqual(detp[key].shape, shape, key)
        np.testing.assert_array_equal(detp["prop"], PROP)
        self.assertEqual(res["seeds"].shape[1], ndet)
        # trajectories sorted by photon id, with the positions as rows
        self.assertTrue(np.all(np.diff(res["traj"]["id"].astype(np.int64)) >= 0))
        self.assertEqual(res["traj"]["pos"].shape[1], 3)

    def test_patterns_in_last_dimension(self):
        """mcxlab moves the patterns to the last dimension, without mixing up the voxels"""
        pat = np.ones((2, 4, 4), np.float32)
        pat[1, :, :2] = 0  # the 2nd pattern is dark for y < 9
        res = pmcx.mcxlab(
            cfg(
                nphoton=50000,
                srctype="pattern",
                srcpos=[5, 5, -0.01],
                srcparam1=[8, 0, 0, 4],
                srcparam2=[0, 8, 0, 4],
                srcpattern=pat,
                srcnum=2,
                issaveref=1,
            )
        )
        flux = res["flux"]
        self.assertEqual(flux.shape, (N, N, N, 1, 2))
        self.assertEqual(res["dref"].shape, flux.shape)
        lo = flux[:, :10, ..., 1].sum() / flux[:, :10, ..., 0].sum()
        hi = flux[:, 10:, ..., 1].sum() / flux[:, 10:, ..., 0].sum()
        self.assertLess(lo, hi)  # the dark half of pattern 2 is at low y, not at low x
        lox = flux[:10, ..., 1].sum() / flux[:10, ..., 0].sum()
        hix = flux[10:, ..., 1].sum() / flux[10:, ..., 0].sum()
        self.assertAlmostEqual(lox / hix, 1, delta=0.05)

    def test_info_commands(self):
        self.assertTrue(pmcx.mcxlab("version").startswith("v"))
        self.assertIsInstance(pmcx.mcxlab("gpuinfo"), list)


@unittest.skipUnless(HAS_DEVICE, "no device")
class TestFeatures(unittest.TestCase):
    """every source type and the less common settings run and return sensible outputs"""

    P2 = dict(srcparam1=[8, 0, 0, 4], srcparam2=[0, 8, 0, 4], srcpos=[6, 6, 0])

    def test_source_types(self):
        cases = {
            "isotropic": dict(srcpos=[10, 10, 10]),
            "arcsine": dict(srcpos=[10, 10, 10]),
            "cone": dict(srcparam1=[0.5, 0, 0, 0]),
            "zgaussian": dict(srcparam1=[0.2, 0, 0, 0]),
            "gaussian": dict(srcparam1=[3, 0, 0, 0]),
            "hyperboloid": dict(srcparam1=[2, 10, 5, 0]),
            "disk": dict(srcparam1=[4, 0, 0, 0]),
            "ring": dict(srcparam1=[5, 2, 0, 0]),
            "planar": self.P2,
            "pattern": dict(self.P2, srcpattern=np.ones((4, 4), np.float32)),
            "pattern3d": dict(
                srcpos=[0, 0, 0],
                srcparam1=[N, N, 5, 0],
                srcpattern=np.ones((N, N, 5), np.float32),
            ),
            "fourier": dict(
                srcparam1=[8, 0, 0, 2], srcparam2=[0, 8, 0, 1], srcpos=[6, 6, 0]
            ),
            "fourierx": dict(
                srcparam1=[8, 0, 0, 8], srcparam2=[1, 1, 0, 0.5], srcpos=[6, 6, 0]
            ),
            "fourierx2d": dict(
                srcparam1=[8, 0, 0, 8], srcparam2=[1, 1, 0, 0], srcpos=[6, 6, 0]
            ),
            "pencilarray": self.P2,
            "line": dict(srcpos=[5, 10, 10], srcparam1=[10, 0, 0, 0]),
            "slit": dict(srcpos=[5, 10, 0], srcparam1=[10, 0, 0, 0]),
        }
        for srctype, kw in cases.items():
            with self.subTest(srctype=srctype):
                res = pmcx.run(cfg(srctype=srctype, **kw))
                self.assertTrue(np.all(np.isfinite(res["flux"])))
                self.assertGreater(absorbed(res), 0.15)
                self.assertLess(absorbed(res), 0.30)

    def test_isotropic_lambertian_launch(self):
        """srcdir[3] = nan or -inf works for every source type, the pencil beam included"""
        pencil = absorbed(pmcx.run(cfg(nphoton=100000)))
        lambert = absorbed(pmcx.run(cfg(nphoton=100000, srcdir=[0, 0, 1, -np.inf])))
        disk = absorbed(
            pmcx.run(
                cfg(
                    nphoton=100000,
                    srctype="disk",
                    srcparam1=[0.01, 0, 0, 0],
                    srcdir=[0, 0, 1, -np.inf],
                )
            )
        )
        self.assertLess(lambert, pencil - 0.01)  # oblique launches escape sooner
        self.assertAlmostEqual(lambert, disk, delta=0.01)
        center = dict(nphoton=100000, srcpos=[10, 10, 10])
        iso = absorbed(pmcx.run(cfg(srcdir=[0, 0, 1, np.nan], **center)))
        self.assertAlmostEqual(
            iso, absorbed(pmcx.run(cfg(srctype="isotropic", **center))), delta=0.01
        )

    def test_polarized(self):
        """polarized runs need no srciquv (it was left uninitialized, failing at random)"""
        c = cfg(polprop=[[0.01, 0.5, 0.1, 1.59, 1.33]])
        c["lambda"] = 633
        for _ in range(3):
            res = pmcx.run(c)
            self.assertEqual(res["prop"].shape, (4, 2))
            self.assertTrue(np.all(np.isfinite(res["flux"])))

    def test_trajectory_only(self):
        res = pmcx.run(cfg(nphoton=100, debuglevel="T"))
        self.assertIn("traj", res)
        self.assertNotIn("flux", res)

    def test_raw_output_normalizer(self):
        """without normalization, stat['normalizer'] is 1 (it was uninitialized)"""
        self.assertEqual(pmcx.run(cfg(isnormalized=0))["stat"]["normalizer"], 1)
        res = pmcx.run(cfg(tstep=1e-9, maxgate=2))
        self.assertEqual(res["flux"].shape, (N, N, N, 5))
        self.assertGreater(res["stat"]["normalizer"], 0)

    def test_multiple_sources(self):
        two = dict(
            srcpos=[[6, 10, 0], [14, 10, 0]],
            srcdir=[[0, 0, 1], [0, 0, 1]],
            nphoton=40000,
        )

        def left(f):
            f = f.reshape(N, N, N, -1)
            return f[:10].sum() / f.sum()

        self.assertEqual(pmcx.run(cfg(**two))["flux"].shape, (N, N, N, 1))
        self.assertAlmostEqual(left(pmcx.run(cfg(**two))["flux"]), 0.5, delta=0.05)
        each = pmcx.run(cfg(srcid=-1, **two))["flux"]
        self.assertEqual(each.shape[-1], 2)
        self.assertGreater(left(each[..., 0]), 0.6)
        self.assertLess(left(each[..., 1]), 0.4)
        self.assertLess(left(pmcx.run(cfg(srcid=2, **two))["flux"]), 0.4)

    def test_adjoint(self):
        det = dict(
            detpos=[[10, 6, 0, 1], [10, 14, 0, 1]], detdir=[[0, 0, 1], [0, 0, 1]]
        )
        res = pmcx.run(cfg(outputtype="adjoint", **det))
        self.assertEqual(res["jmua"].shape, (N, N, N, 2))
        self.assertEqual(
            res["flux"].shape, (N, N, N, 1, 3)
        )  # the source and the 2 detectors
        res = pmcx.run(cfg(outputtype="adjoint_mua_d", omega=2 * np.pi * 1e8, **det))
        self.assertTrue(np.iscomplexobj(res["jmua"]) and np.iscomplexobj(res["jd"]))

    def test_boundary_conditions(self):
        res = pmcx.run(cfg(bc="______001000"))  # detect photons leaving at z=0
        self.assertGreater(res["detp"].shape[1], 0)
        self.assertGreater(
            absorbed(pmcx.run(cfg(bc="cc____", srcpos=[1, 10, 0]))),
            REF_HOMOGENEOUS + 0.03,
        )
        self.assertAlmostEqual(absorbed(pmcx.run(cfg(bc="mmmmmm"))), 1.0, delta=1e-3)

    def test_replay_each_detector(self):
        det = dict(detpos=[[10, 6, 0, 2], [10, 14, 0, 2]])
        base = pmcx.run(cfg(nphoton=20000, issaveseed=1, **det))
        jac = pmcx.run(
            cfg(
                seed=base["seeds"],
                detphotons=base["detp"],
                replaydet=-1,
                outputtype="jacobian",
                **det
            )
        )
        self.assertEqual(jac["flux"].shape, (N, N, N, 1, 2))

    def test_misc_settings(self):
        self.assertEqual(pmcx.run(cfg(respin=2))["stat"]["nphoton"], 20000)
        self.assertEqual(
            pmcx.run(cfg(vol=np.ones((N, 1, N), np.uint8), srcpos=[10, 0, 0]))[
                "flux"
            ].shape,
            (N, 1, N, 1),
        )
        many = (
            (np.arange(N**3) % 300 + 1).reshape(N, N, N).astype(np.uint16)
        )  # more than 255 labels
        res = pmcx.run(cfg(vol=many, prop=np.vstack([PROP[:1]] + [PROP[1:]] * 300)))
        self.assertAlmostEqual(absorbed(res), REF_HOMOGENEOUS, delta=0.02)
        self.assertGreater(
            pmcx.run(cfg(detpos=[[10, 8, 0, 3]], maxdetphoton=10))["detp"].shape[1], 0
        )
        self.assertEqual(
            pmcx.run(cfg(detpos=[[10, 8, 0, 3]], maxdetphoton=10))["detp"].shape[1], 10
        )


@unittest.skipUnless(HAS_DEVICE, "no device")
class TestEndToEnd(unittest.TestCase):
    """short absorbed-fraction checks against 1e7-photon references"""

    def test_homogeneous_cube(self):
        self.assertAlmostEqual(
            absorbed(pmcx.run(cfg(nphoton=100000))), REF_HOMOGENEOUS, delta=0.01
        )

    def test_two_layers_index_matched(self):
        vol = np.ones((N, N, N), np.uint8)
        vol[:, :, 8:] = 2
        res = pmcx.run(cfg(nphoton=100000, vol=vol, prop=PROP2, isreflect=0))
        self.assertAlmostEqual(absorbed(res), REF_TWOLAYER, delta=0.01)

    def test_per_voxel_media(self):
        """every per-voxel format of the homogeneous medium matches the label volume, with
        mua/mus in 1/mm scaled by unitinmm once, whatever the key order"""
        mua, mus = PROP[1, 0], PROP[1, 1]
        ones = np.ones((N, N, N), np.float32)
        formats = {
            "mua_float": np.stack([mua * ones]),
            "muamus_float": np.stack([mua * ones, mus * ones]),
            "asgn_float": np.stack([mua * ones, mus * ones, 0.9 * ones, 1.37 * ones]),
            "labelplus_mua": np.stack(
                [mua * ones, 0 * ones, ones]
            ),  # replaces the mua of label 1
            "labelplus_mus": np.stack([mus * ones, 1 * ones, ones]),  # replaces its mus
        }
        props = {
            "labelplus_mua": [[0, 0, 1, 1], [0.05, 1, 0.9, 1.37]],
            "labelplus_mus": [[0, 0, 1, 1], [0.01, 5, 0.9, 1.37]],
        }
        ref = absorbed(pmcx.run(cfg(nphoton=50000, unitinmm=2)))
        for name, vol in formats.items():
            prop = props.get(name, PROP)
            first = dict(unitinmm=2, vol=vol)
            first.update(
                {k: v for k, v in cfg(nphoton=50000, prop=prop).items() if k != "vol"}
            )
            for order, c in (
                ("unitinmm last", cfg(nphoton=50000, vol=vol, prop=prop, unitinmm=2)),
                ("unitinmm first", first),
            ):
                with self.subTest(format=name, order=order):
                    self.assertAlmostEqual(absorbed(pmcx.run(c)), ref, delta=0.01)


if __name__ == "__main__":
    unittest.main()
