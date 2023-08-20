#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 15:44:04 2023

@author: ivyyen
"""

import unittest
import sys

sys.path.append("../pmcx/")

from utils import (
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
        cfg = {
            "nphoton": 1e2,
            "vol": np.ones([2, 2, 2], dtype="uint8"),
            "tstart": 0,
            "tend": 5e-9,
            "tstep": 5e-9,
            "srcpos": [1, 1, 1],
            "srcdir": [0, 0, 1],
            "prop": np.array([[0, 0, 1, 1], [0.005, 1, 0, 1.37], [0.1, 10, 0.9, 1]]),
        }
        cfg["detpos"] = [
            [1, 1, 0, 1],
            [1, 1, 0, 1],
        ]  # to detect photons, one must first define detectors
        cfg[
            "issavedet"
        ] = 1  # cfg.issavedet must be set to 1 or True in order to save detected photons
        cfg[
            "issrcfrom0"
        ] = 1  # set this flag to ensure src/det coordinates align with voxel space

        # expected output data
        detp = {
            "detid": np.array([1, 1, 1, 1, 1]),
            "ppath": np.array(
                [
                    [1.059002, 0.0],
                    [1.9772522, 0.0],
                    [2.069064, 0.0],
                    [2.8588557, 0.0],
                    [2.5556042, 0.0],
                ]
            ),
            "prop": np.array(
                [
                    [0.00e00, 0.00e00, 1.00e00, 1.00e00],
                    [5.00e-03, 1.00e00, 0.00e00, 1.37e00],
                    [1.00e-01, 1.00e01, 9.00e-01, 1.00e00],
                ]
            ),
            "data": np.array(
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.059002, 1.9772522, 2.069064, 2.8588557, 2.5556042],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        }
        flux = np.array(
            [
                [[[2.5131630e07], [3.6653404e07]], [[4.1204576e07], [4.7903040e07]]],
                [[[2.6910164e07], [3.5469168e07]], [[2.5065972e07], [1.6582197e08]]],
            ]
        )
        stat = {
            "runtime": 2,
            "nphoton": 100,
            "energytot": 100.0,
            "energyabs": 1.0103997588157654,
            "normalizer": 2000000.0,
            "unitinmm": 1.0,
            "workload": [3584.0],
        }
        expected_output = {"detp": detp, "flux": flux, "stat": stat}

        actual_output = mcxlab(cfg)
        self.check_outputs(actual_output, expected_output)


if __name__ == "__main__":
    unittest.main()
