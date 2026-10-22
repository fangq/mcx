# Copyright (c) 2023-2025 Qianqian Fang (q.fang <at> neu.edu)
# Copyright (c) 2023 Fan-Yu (Ivy) Yen (yen.f at northeastern.edu)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Created on Fri Jul 28 15:44:04 2023

@author: ivyyen
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pmcx.utils import (
    detweight,
    meanpath,
    dettpsf,
    dettime,
    tddiffusion,
    getdistance,
    detphoton,
    mcxlab,
)

import numpy as np


class TestFunctions(unittest.TestCase):
    def test_detweight_with_valid_input(self):
        detp = {
            "detid": np.array([1, 2, 1]),
            "ppath": np.array([[4.0361433, 0], [40.828487, 0], [47.87069, 0]]),
            "prop": np.array([[0, 0, 1, 1], [0.005, 1, 0, 1.37], [0.1, 10, 0.9, 1]]),
            "data": np.array([[1, 2, 1], [4.0361433, 40.828487, 47.87069], [0, 0, 0]]),
        }
        expected_output = np.array([0.9800216, 0.81534624, 0.78713661])
        actual_output = detweight(detp)
        np.testing.assert_array_almost_equal(actual_output, expected_output)

    def test_detweight_with_w0(self):
        ppath = np.array([[4.0361433, 0], [40.828487, 0], [47.87069, 0]])
        prop = np.array([[0, 0, 1, 1], [0.005, 1, 0, 1.37], [0.1, 10, 0.9, 1]])
        atten = np.array([0.9800216, 0.81534624, 0.78713661])
        w0 = np.array([0.5, 1.0, 0.25])

        # w0 as returned by pmcx, a (nphoton, 1) column, and as a 1-D array
        for w in (w0[:, np.newaxis], w0):
            actual_output = detweight({"ppath": ppath, "prop": prop, "w0": w})
            self.assertEqual(actual_output.shape, (3,))
            np.testing.assert_array_almost_equal(actual_output, w0 * atten)

        # one initial weight column per source (pattern sources)
        w2 = np.stack([w0, 2 * w0], axis=1)
        actual_output = detweight({"ppath": ppath, "prop": prop, "w0": w2})
        self.assertEqual(actual_output.shape, (3, 2))
        np.testing.assert_array_almost_equal(actual_output, w2 * atten[:, np.newaxis])

    def test_detweight_with_missing_prop(self):
        detp = {
            "ppath": np.array([[4.0361433, 0], [40.828487, 0], [47.87069, 0]]),
        }
        with self.assertRaises(ValueError):
            detweight(detp)

    def test_detweight_with_invalid_prop(self):
        detp = {"prop": np.array([[1.0]]), "ppath": np.array([[1.0, 1.0], [2.0, 2.0]])}
        with self.assertRaises(ValueError):
            detweight(detp, detp["prop"])

    def test_meanpath_with_valid_input(self):
        detp = {
            "ppath": np.array([[4.0361433, 0], [40.828487, 0], [47.87069, 0]]),
            "prop": np.array([[0, 0, 1, 1], [0.005, 1, 0, 1.37], [0.1, 10, 0.9, 1]]),
        }
        expected_output = np.array([29.01278056, 0])
        actual_output = meanpath(detp, detp["prop"])
        np.testing.assert_array_almost_equal(actual_output, expected_output)

    def test_meanpath_with_missing_prop(self):
        detp = {"ppath": np.array([[4.0361433, 0], [40.828487, 0], [47.87069, 0]])}
        with self.assertRaises(ValueError):
            meanpath(detp)

    def test_dettpsf_with_valid_input(self):
        detp = {
            "detid": np.array([1, 2, 1]),
            "ppath": np.array([[4.0361433, 0], [40.828487, 0], [47.87069, 0]]),
            "prop": np.array([[0, 0, 1, 1], [0.005, 1, 0, 1.37], [0.1, 10, 0.9, 1]]),
            "data": np.array([[1, 2, 1], [4.0361433, 40.828487, 47.87069], [0, 0, 0]]),
        }
        detnum = 1
        tstart = 0
        tstep = 5e-9
        tend = 5e-9
        time = np.array([tstart, tstep, tend])
        expected_output = 1.767158
        actual_output = dettpsf(detp, detnum, detp["prop"], time)
        np.testing.assert_array_almost_equal(actual_output, expected_output)

    def test_dettime_with_valid_input(self):
        detp = {
            "detid": np.array([1, 2, 1]),
            "ppath": np.array([[4.0361433, 0], [40.828487, 0], [47.87069, 0]]),
            "prop": np.array([[0, 0, 1, 1], [0.005, 1, 0, 1.37], [0.1, 10, 0.9, 1]]),
            "data": np.array([[1, 2, 1], [4.0361433, 40.828487, 47.87069], [0, 0, 0]]),
        }
        expected_output = np.array([[1.84444811e-11, 1.86579167e-10, 2.18760825e-10]])
        actual_output = dettime(detp)
        np.testing.assert_array_almost_equal(actual_output, expected_output)

    def test_tddiffusion_with_valid_input(self):
        cfg = {
            "tstart": 0,
            "tend": 5e-9,
            "tstep": 5e-9,
            "srcpos": [29, 29, 0],
            "prop": np.array([[0, 0, 1, 1], [0.005, 1, 0, 1.37], [0.1, 10, 0.9, 1]]),
        }
        c0 = 299792458000
        twin = np.arange(cfg["tstart"] + cfg["tstep"] / 2, cfg["tend"], cfg["tstep"])

        mua = cfg["prop"][1, 0]
        musp = cfg["prop"][1, 1] * (1 - cfg["prop"][1, 2])
        v = c0
        Reff = 0
        srcpos = cfg["srcpos"]
        detpos = np.array([[29, 13, 8], [0, 0, 0]])
        t = twin

        expected_output = np.array([[1888.13933534], [92.31054244]])
        actual_output = tddiffusion(mua, musp, v, Reff, srcpos, detpos + 1, t)
        np.testing.assert_array_almost_equal(actual_output, expected_output)

    def test_getdistance_with_valid_input(self):
        srcpos = np.array([[1, 5, 7], [43, 62, 4]])
        detpos = np.array([[1, 3, 8], [40, 60, 2]])
        expected_output = np.array(
            [[2.23606798, 72.53275122], [67.60917098, 4.12310563]]
        )
        actual_output = getdistance(srcpos, detpos)
        np.testing.assert_array_almost_equal(actual_output, expected_output)

    def test_detphoton_with_valid_input(self):
        detp = np.array([[1, 2, 1], [4.0361433, 40.828487, 47.87069], [0, 0, 0]])
        medianum = 1
        flag = ["DP"]
        expected_output = {
            "detid": np.array([1, 2, 1]),
            "ppath": np.array([[4.0361433], [40.828487], [47.87069]]),
        }
        actual_output = detphoton(detp, medianum, *flag)
        self.assertEqual(actual_output.keys(), expected_output.keys())
        for key in actual_output.keys():
            if isinstance(actual_output[key], np.ndarray):
                np.testing.assert_array_equal(actual_output[key], expected_output[key])
            else:
                self.assertEqual(actual_output[key], expected_output[key])

    # check mcxlab nested output

    def get_first_three_digits(self, num):
        return np.array([int(digit) for digit in str(num)[:3]])

    def check_outputs(self, actual, expected, path="", decimal=6):
        # Check if both items are dictionaries
        if isinstance(actual, dict) and isinstance(expected, dict):
            self.assertEqual(
                set(actual.keys()),
                set(expected.keys()),
                f"Keys mismatch at path: {path}",
            )

            for key in actual.keys():
                new_path = f"{path}.{key}" if path else key

                # Special handling for flux
                if key == "flux":
                    actual_val = self.get_first_three_digits(np.sum(actual[key]))
                    expected_val = self.get_first_three_digits(np.sum(actual[key]))

                    np.testing.assert_array_equal(actual_val, expected_val)
                else:
                    self.check_outputs(actual[key], expected[key], new_path, decimal)

        # Check if both items are numpy arrays
        elif isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
            try:
                np.testing.assert_array_equal(actual, expected)
            except AssertionError:
                np.testing.assert_array_almost_equal(
                    actual, expected, decimal, f"Array mismatch at path: {path}"
                )

        # Check if both items are other types (stat)
        # else:
        # self.assertEqual(actual, expected, f"Mismatch at path: {path}")

    def test_mcxlab_with_valid_input(self):
        # with the default autopilot, photons are dispatched to the GPU threads dynamically and
        # two runs with the same seed differ; autopilot=0 selects the static split, under which a
        # fixed seed reproduces the same photons on the same device
        try:
            if not mcxlab("gpuinfo"):
                self.skipTest("no GPU available")
        except ImportError:
            self.skipTest("neither pmcx nor pmcxcl binary extension is available")

        cfg = {
            "nphoton": 1e3,
            "vol": np.ones([2, 2, 2], dtype="uint8"),
            "tstart": 0,
            "tend": 5e-9,
            "tstep": 5e-9,
            "srcpos": [1, 1, 1],
            "srcdir": [0, 0, 1],
            "prop": np.array([[0, 0, 1, 1], [0.005, 1, 0, 1.37], [0.1, 10, 0.9, 1]]),
            "detpos": [[1, 1, 0, 1]],  # to detect photons, one must first define detectors
            "issavedet": 1,  # cfg.issavedet must be set to 1 or True in order to save detected photons
            "issrcfrom0": 1,  # set this flag to ensure src/det coordinates align with voxel space
            "autopilot": 0,
            "seed": 12345,
        }

        res = [mcxlab(dict(cfg)) for _ in range(2)]

        for out in res:
            detp, flux = out["detp"], np.asarray(out["flux"])
            self.assertEqual(sorted(detp.keys())[:2], ["data", "detid"])
            self.assertGreater(len(detp["detid"]), 0)
            self.assertTrue(np.all(np.asarray(detp["detid"]) == 1))
            self.assertTrue(np.all(np.asarray(detp["ppath"])[:, 0] > 0))
            self.assertEqual(flux.shape[:3], (2, 2, 2))
            self.assertTrue(np.all(np.isfinite(flux)) and np.all(flux > 0))
            self.assertEqual(out["stat"]["nphoton"], 1000)

        # the static split reproduces the detected photons exactly, but they are stored in the
        # order the threads finish, and the fluence only up to the order of the float additions
        recs = [np.asarray(out["detp"]["data"]).T for out in res]
        recs = [r[np.lexsort(r.T[::-1])] for r in recs]
        np.testing.assert_array_equal(recs[0], recs[1])
        np.testing.assert_allclose(res[0]["flux"], res[1]["flux"], rtol=1e-5)

if __name__ == "__main__":
    unittest.main()
