# Copyright (c) 2023-2025 Qianqian Fang (q.fang <at> neu.edu)
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

import unittest
import tempfile
import os
import sys
import json
import struct
import numpy as np
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestPMCXUtils(unittest.TestCase):
    """Unit tests for pmcx.utils module"""

    def setUp(self):
        """Set up test data"""
        self.basic_prop = np.array(
            [
                [0.0, 0.0, 1.0, 1.0],  # background
                [0.01, 10.0, 0.9, 1.37],  # tissue 1
                [0.02, 8.0, 0.8, 1.4],  # tissue 2
            ]
        )

        self.detp_data = {
            "detid": np.array([1, 1, 2, 2]),
            "ppath": np.array(
                [[1.0, 2.0], [1.5, 1.8], [0.8, 2.5], [1.2, 2.2]], dtype=np.float32
            ),
            "nscat": np.array([[5, 3], [4, 2], [6, 4], [3, 5]], dtype=np.float32),
            "prop": self.basic_prop,
            "unitinmm": 0.1,
            "w0": np.array([1.0, 0.9, 0.8, 0.7]),
        }

    def test_detweight_basic(self):
        """Test basic detweight calculation"""
        from pmcx.utils import detweight

        # Test with basic parameters
        weights = detweight(self.detp_data, self.basic_prop, unitinmm=0.1)

        self.assertEqual(len(weights), 4)
        self.assertTrue(all(w > 0 for w in weights))
        self.assertTrue(all(w <= 1 for w in weights))

    def test_detweight_with_w0(self):
        """Test detweight with initial weights"""
        from pmcx.utils import detweight

        weights = detweight(self.detp_data, self.basic_prop, unitinmm=0.1)

        # Check that weights are affected by path lengths and optical properties
        expected_weight_0 = 1.0 * np.exp(-0.01 * 1.0 * 0.1) * np.exp(-0.02 * 2.0 * 0.1)
        self.assertAlmostEqual(weights[0], expected_weight_0, places=6)

    def test_detweight_errors(self):
        """Test detweight error conditions"""
        from pmcx.utils import detweight

        # Test with empty property list
        empty_prop = np.array([[0.0, 0.0, 1.0, 1.0]])
        with self.assertRaises(ValueError):
            detweight(self.detp_data, empty_prop)

        # Test with missing prop in detp and no prop provided
        detp_no_prop = self.detp_data.copy()
        del detp_no_prop["prop"]
        with self.assertRaises(ValueError):
            detweight(detp_no_prop)

    def test_meanpath(self):
        """Test meanpath calculation"""
        from pmcx.utils import meanpath

        avg_path = meanpath(self.detp_data, self.basic_prop)

        self.assertEqual(len(avg_path), 2)  # Two tissue types
        self.assertTrue(all(p > 0 for p in avg_path))

    def test_meanscat(self):
        """Test meanscat calculation"""
        from pmcx.utils import meanscat

        avg_nscat = meanscat(self.detp_data, self.basic_prop)

        self.assertTrue(avg_nscat > 0)
        self.assertIsInstance(avg_nscat, (float, np.floating))

    def test_dettime(self):
        """Test dettime calculation"""
        from pmcx.utils import dettime

        times = dettime(self.detp_data, self.basic_prop, unitinmm=0.1)

        self.assertEqual(times.shape, (1, 4))
        self.assertTrue(all(t > 0 for t in times.flatten()))

    def test_cwdref(self):
        """Test cwdref calculation"""
        from pmcx.utils import cwdref

        cfg = {
            "unitinmm": 0.1,
            "prop": self.basic_prop,
            "detpos": np.array([[0, 0, 0, 1.0], [5, 5, 0, 1.0]]),
            "nphoton": 1000000,
        }

        dref = cwdref(self.detp_data, cfg)

        self.assertEqual(len(dref), 2)  # Two detectors
        self.assertTrue(all(d >= 0 for d in dref))

    def test_getdistance(self):
        """Test getdistance calculation"""
        from pmcx.utils import getdistance

        src_pos = np.array([[0, 0, 0], [10, 10, 10]])
        det_pos = np.array([[5, 0, 0], [0, 5, 0]])

        distances = getdistance(src_pos, det_pos)

        self.assertEqual(distances.shape, (2, 2))  # 2 detectors, 2 sources

        # Test known distance
        expected_dist_00 = np.sqrt(25)  # distance from (0,0,0) to (5,0,0)
        self.assertAlmostEqual(distances[0, 0], expected_dist_00, places=6)

    def test_tddiffusion(self):
        """Test tddiffusion analytical solution"""
        from pmcx.utils import tddiffusion

        mua = 0.01
        musp = 1.0
        v = 2.998e8  # speed of light in mm/s
        Reff = 0.493
        srcpos = np.array([0, 0, 0])
        detpos = np.array([[10, 0, 0]])
        t = np.array([1e-9, 2e-9, 3e-9])

        phi = tddiffusion(mua, musp, v, Reff, srcpos, detpos, t)

        self.assertEqual(phi.size, 3)  # Three time points
        self.assertTrue(np.all(phi > 0))

    def test_detphoton(self):
        """Test detphoton data separation"""
        from pmcx.utils import detphoton

        # Create test data array
        medianum = 2
        nphot = 4

        # Create combined data array [detid, nscat1, nscat2, ppath1, ppath2, p1, p2, p3, w0]
        combined_data = np.array(
            [
                [1, 2, 1, 2],  # detid
                [5, 4, 6, 3],  # nscat tissue 1
                [3, 2, 4, 5],  # nscat tissue 2
                [1.0, 1.5, 0.8, 1.2],  # ppath tissue 1
                [2.0, 1.8, 2.5, 2.2],  # ppath tissue 2
                [0, 1, 2, 3],  # p x
                [0, 1, 2, 3],  # p y
                [0, 1, 2, 3],  # p z
                [1.0, 0.9, 0.8, 0.7],  # w0
            ]
        )

        savedetflag = "dspxw"
        newdetp = detphoton(combined_data, medianum, savedetflag)

        self.assertIn("detid", newdetp)
        self.assertIn("nscat", newdetp)
        self.assertIn("ppath", newdetp)
        self.assertIn("p", newdetp)
        self.assertIn("w0", newdetp)

        self.assertEqual(newdetp["detid"].shape, (4,))
        self.assertEqual(newdetp["nscat"].shape, (4, 2))
        self.assertEqual(newdetp["ppath"].shape, (4, 2))

    def test_cwdiffusion(self):
        """Test cwdiffusion analytical solution"""
        from pmcx.utils import cwdiffusion

        mua = 0.01
        musp = 1.0
        Reff = 0.493
        srcpos = np.array([0, 0, 0])
        detpos = np.array([[10, 0, 0], [20, 0, 0]])

        phi, r = cwdiffusion(mua, musp, Reff, srcpos, detpos)

        self.assertEqual(phi.shape, (2, 1))  # 2 detectors, 1 source
        self.assertEqual(r.shape, (2, 1))
        self.assertTrue(all(p > 0 for p in phi.flatten()))
        self.assertTrue(all(r_val > 0 for r_val in r.flatten()))

    def test_cwfluxdiffusion(self):
        """Test cwfluxdiffusion calculation"""
        from pmcx.utils import cwfluxdiffusion

        mua = 0.01
        musp = 1.0
        Reff = 0.493
        srcpos = np.array([0, 0, 0])
        detpos = np.array([[10, 0, 0]])

        flux = cwfluxdiffusion(mua, musp, Reff, srcpos, detpos)

        self.assertGreater(flux[0, 0], 0)

    def test_mcxcreate(self):
        """Test mcxcreate benchmark creation"""
        from pmcx.utils import mcxcreate

        # Test getting list of benchmarks
        bench_list = mcxcreate()
        self.assertIsInstance(bench_list, list)
        self.assertIn("cube60", bench_list)
        self.assertIn("cube60b", bench_list)

        # Test creating a specific benchmark
        cfg = mcxcreate("cube60")
        self.assertIsInstance(cfg, dict)
        self.assertIn("nphoton", cfg)
        self.assertIn("vol", cfg)
        self.assertIn("prop", cfg)

        # Test with custom parameters
        cfg_custom = mcxcreate("cube60", nphoton=2000000, seed=12345)
        self.assertEqual(cfg_custom["nphoton"], 2000000)
        self.assertEqual(cfg_custom["seed"], 12345)

        # Test error for unsupported benchmark
        with self.assertRaises(ValueError):
            mcxcreate("nonexistent_benchmark")

    def test_dettpsf(self):
        """Test dettpsf calculation"""
        from pmcx.utils import dettpsf

        detnum = 1
        time_config = [0, 5e-9, 1e-10]

        tpsf = dettpsf(self.detp_data, detnum, self.basic_prop, time_config)

        self.assertEqual(tpsf.shape[1], 1)  # Single column output
        self.assertGreater(tpsf.shape[0], 0)  # Has time bins

    @patch("pmcx.utils.loadmch")
    def test_dcsg1_with_file(self, mock_loadmch):
        """Test dcsg1 with file input"""
        from pmcx.utils import dcsg1

        # Mock file loading
        mock_mch_data = np.array(
            [
                [
                    1,
                    0,
                    1.0,
                    2.0,
                    0.1,
                    0.2,
                ],  # detid, reserved, ppath1, ppath2, mom1, mom2
                [1, 0, 1.5, 1.8, 0.15, 0.18],
            ]
        )
        mock_header = {"medianum": 2}
        mock_loadmch.return_value = (mock_mch_data, mock_header)

        # Mock jdata.load for config
        with patch("jdata.load") as mock_jload:
            mock_cfg = {
                "Domain": {
                    "Media": [
                        {"mua": 0.0, "mus": 0.0, "g": 1.0, "n": 1.0},
                        {"mua": 0.01, "mus": 10.0, "g": 0.9, "n": 1.37},
                        {"mua": 0.02, "mus": 8.0, "g": 0.8, "n": 1.4},
                    ]
                }
            }
            mock_jload.return_value = mock_cfg

            tau, g1 = dcsg1("test.mch")

            self.assertEqual(len(tau), 200)  # Default tau length
            self.assertGreater(g1.shape[0], 0)


class TestPMCXIO(unittest.TestCase):
    """Unit tests for pmcx.io module"""

    def setUp(self):
        """Set up test data"""
        self.test_dim = [10, 10, 10, 5]
        self.test_data = np.random.rand(*self.test_dim).astype(np.float32)

    def create_temp_mc2_file(self, data, format_type="float"):
        """Helper to create temporary MC2 file"""
        temp_file = tempfile.NamedTemporaryFile(suffix=".mc2", delete=False)
        temp_file.write(data.tobytes())
        temp_file.close()
        return temp_file.name

    def test_loadmc2_basic(self):
        """Test basic loadmc2 functionality"""
        from pmcx.io import loadmc2

        # Create test file
        temp_file = self.create_temp_mc2_file(self.test_data)

        try:
            loaded_data = loadmc2(temp_file, self.test_dim, "float")
            np.testing.assert_array_equal(loaded_data, self.test_data)
        finally:
            os.unlink(temp_file)

    def test_loadmc2_different_formats(self):
        """Test loadmc2 with different data formats"""
        from pmcx.io import loadmc2

        # Test with uint8
        test_data_uint8 = np.random.randint(0, 256, size=(5, 5, 5), dtype=np.uint8)
        temp_file = self.create_temp_mc2_file(test_data_uint8)

        try:
            loaded_data = loadmc2(temp_file, [5, 5, 5], "uint8")
            np.testing.assert_array_equal(loaded_data, test_data_uint8)
        finally:
            os.unlink(temp_file)

    def test_loadmc2_with_offset(self):
        """Test loadmc2 with byte offset"""
        from pmcx.io import loadmc2

        # Create file with header + data
        header = b"HEADER" + b"\x00" * 10
        temp_file = tempfile.NamedTemporaryFile(suffix=".mc2", delete=False)
        temp_file.write(header + self.test_data.tobytes())
        temp_file.close()

        try:
            loaded_data = loadmc2(temp_file.name, self.test_dim, "float", offset=16)
            np.testing.assert_array_equal(loaded_data, self.test_data)
        finally:
            os.unlink(temp_file.name)

    def test_loadmc2_errors(self):
        """Test loadmc2 error conditions"""
        from pmcx.io import loadmc2

        # Test unsupported format
        temp_file = self.create_temp_mc2_file(self.test_data)
        try:
            with self.assertRaises(ValueError):
                loadmc2(temp_file, self.test_dim, "unsupported_format")
        finally:
            os.unlink(temp_file)

        # Test non-existent file
        with self.assertRaises(IOError):
            loadmc2("nonexistent_file.mc2", self.test_dim)

    def create_temp_mch_file(self):
        """Helper to create temporary MCH file with proper format"""
        temp_file = tempfile.NamedTemporaryFile(suffix=".mch", delete=False)

        # MCH file format: magic header + header data + photon data
        magic = b"MCXH"

        # Header: version, medianum, detnum, recordnum, totalphoton, detectedphoton, savedphoton
        header = struct.pack("<7I", 1, 2, 2, 6, 1000, 100, 4)  # 7 uint32 values

        # Additional header fields
        unitmm = struct.pack("<f", 0.1)  # float
        seedbyte = struct.pack("<I", 0)  # uint32
        normalizer = struct.pack("<f", 1000000.0)  # float
        respin = struct.pack("<i", 1)  # int32
        srcnum = struct.pack("<I", 1)  # uint32
        savedetflag = struct.pack("<I", 0)  # uint32
        totalsource = struct.pack("<I", 1)  # uint32
        junk = struct.pack("<I", 0)  # uint32

        # Sample photon data (4 photons, 6 columns each)
        photon_data = np.array(
            [
                [1, 5, 3, 1.0, 2.0, 1.0],  # detid, nscat1, nscat2, ppath1, ppath2, w0
                [1, 4, 2, 1.5, 1.8, 0.9],
                [2, 6, 4, 0.8, 2.5, 0.8],
                [2, 3, 5, 1.2, 2.2, 0.7],
            ],
            dtype=np.float32,
        )

        # Write to file
        temp_file.write(magic)
        temp_file.write(header)
        temp_file.write(unitmm)
        temp_file.write(seedbyte)
        temp_file.write(normalizer)
        temp_file.write(respin)
        temp_file.write(srcnum)
        temp_file.write(savedetflag)
        temp_file.write(totalsource)
        temp_file.write(junk)
        temp_file.write(photon_data.tobytes())

        temp_file.close()
        return temp_file.name

    def test_loadmch_basic(self):
        """Test basic loadmch functionality"""
        from pmcx.io import loadmch

        temp_file = self.create_temp_mch_file()

        try:
            data, header = loadmch(temp_file)

            self.assertEqual(data.shape[0], 4)  # 4 photons
            self.assertEqual(data.shape[1], 6)  # 6 columns

            self.assertIn("version", header)
            self.assertIn("medianum", header)
            self.assertIn("detnum", header)
            self.assertEqual(header["version"], 1)
            self.assertEqual(header["medianum"], 2)

        finally:
            os.unlink(temp_file)

    def test_loadmch_errors(self):
        """Test loadmch error conditions"""
        from pmcx.io import loadmch

        # Test non-existent file
        with self.assertRaises(IOError):
            loadmch("nonexistent_file.mch")

        # Test file without proper magic header
        temp_file = tempfile.NamedTemporaryFile(suffix=".mch", delete=False)
        temp_file.write(b"INVALID_HEADER")
        temp_file.close()

        try:
            with self.assertRaises(RuntimeError):
                loadmch(temp_file.name)
        finally:
            os.unlink(temp_file.name)

    def test_loadfile_mc2(self):
        """Test loadfile with MC2 format"""
        from pmcx.io import loadfile

        temp_file = self.create_temp_mc2_file(self.test_data)

        try:
            data, header = loadfile(temp_file, self.test_dim, "float")

            # Data should be log10 transformed for MC2 files
            expected_data = np.log10(self.test_data)
            np.testing.assert_array_almost_equal(data, expected_data)

            self.assertIn("format", header)
            self.assertIn("scale", header)
            self.assertEqual(header["scale"], "log10")

        finally:
            os.unlink(temp_file)

    def test_loadfile_mch(self):
        """Test loadfile with MCH format"""
        from pmcx.io import loadfile

        temp_file = self.create_temp_mch_file()

        try:
            data, header = loadfile(temp_file)

            self.assertEqual(data.shape[0], 4)  # 4 photons
            self.assertIn("version", header)

        finally:
            os.unlink(temp_file)

    def test_mcx2json_basic(self):
        """Test basic mcx2json functionality"""
        from pmcx.io import mcx2json

        cfg = {
            "nphoton": 1000000,
            "vol": np.ones((10, 10, 10), dtype=np.uint8),
            "srcpos": [5, 5, 0],
            "srcdir": [0, 0, 1],
            "tstart": 0,
            "tend": 5e-9,
            "tstep": 5e-9,
            "prop": [[0, 0, 1, 1], [0.005, 1, 0.01, 1.37]],
            "detpos": [[5, 3, 0, 1], [5, 7, 0, 1]],
            "isreflect": 1,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            filestub = os.path.join(temp_dir, "test")
            mcx2json(cfg, filestub)

            # Check that JSON file was created
            json_file = filestub + ".json"
            self.assertTrue(os.path.exists(json_file))

            # Load and verify JSON content
            with open(json_file, "r") as f:
                json_data = json.load(f)

            self.assertIn("Session", json_data)
            self.assertIn("Forward", json_data)
            self.assertIn("Optode", json_data)
            self.assertIn("Domain", json_data)

            # Check specific values
            self.assertEqual(json_data["Session"]["Photons"], 1000000)
            self.assertEqual(json_data["Forward"]["T0"], 0)
            self.assertEqual(json_data["Forward"]["T1"], 5e-9)

    def test_json2mcx_basic(self):
        """Test basic json2mcx functionality"""
        from pmcx.io import json2mcx

        # Create test JSON data
        json_data = {
            "Session": {"Photons": 1000000, "RNGSeed": 12345, "DoMismatch": 1},
            "Forward": {"T0": 0, "T1": 5e-9, "Dt": 5e-9},
            "Optode": {
                "Source": {"Pos": [5, 5, 0], "Dir": [0, 0, 1], "Type": "pencil"},
                "Detector": [{"Pos": [5, 3, 0], "R": 1}, {"Pos": [5, 7, 0], "R": 1}],
            },
            "Domain": {
                "Media": [
                    {"mua": 0, "mus": 0, "g": 1, "n": 1},
                    {"mua": 0.005, "mus": 1, "g": 0.01, "n": 1.37},
                ],
                "Dim": [10, 10, 10],
            },
        }

        cfg = json2mcx(json_data)

        self.assertEqual(cfg["nphoton"], 1000000)
        self.assertEqual(cfg["seed"], 12345)
        self.assertEqual(cfg["isreflect"], 1)
        self.assertEqual(cfg["tstart"], 0)
        self.assertEqual(cfg["tend"], 5e-9)
        self.assertEqual(cfg["tstep"], 5e-9)
        self.assertEqual(cfg["srcpos"], [5, 5, 0])
        self.assertEqual(cfg["srcdir"], [0, 0, 1])
        self.assertEqual(len(cfg["prop"]), 2)
        self.assertEqual(len(cfg["detpos"]), 2)

    def test_json2mcx_with_file(self):
        """Test json2mcx with file input"""
        from pmcx.io import json2mcx

        json_data = {
            "Session": {"Photons": 500000},
            "Forward": {"T0": 0, "T1": 1e-9, "Dt": 1e-9},
            "Domain": {"Media": [{"mua": 0, "mus": 0, "g": 1, "n": 1}]},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            temp_file = f.name

        try:
            with patch("jdata.load") as mock_load:
                mock_load.return_value = json_data
                cfg = json2mcx(temp_file)
                self.assertEqual(cfg["nphoton"], 500000)
        finally:
            os.unlink(temp_file)


class TestIntegration(unittest.TestCase):
    """Integration tests for both modules"""

    def test_detweight_dettime_integration(self):
        """Test integration between detweight and dettime"""
        from pmcx.utils import detweight, dettime

        detp_data = {
            "detid": np.array([1, 1, 2]),
            "ppath": np.array([[1.0, 2.0], [1.5, 1.8], [0.8, 2.5]]),
            "prop": np.array(
                [[0.0, 0.0, 1.0, 1.0], [0.01, 10.0, 0.9, 1.37], [0.02, 8.0, 0.8, 1.4]]
            ),
            "unitinmm": 0.1,
        }

        weights = detweight(detp_data)
        times = dettime(detp_data)

        self.assertEqual(len(weights), len(times.flatten()))
        self.assertTrue(all(w > 0 for w in weights))
        self.assertTrue(all(t > 0 for t in times.flatten()))

    def test_mc2_mch_workflow(self):
        """Test workflow from MC2 to MCH file processing"""
        from pmcx.io import loadmc2, loadfile
        from pmcx.utils import detweight

        # Create MC2 test data
        test_data = np.random.rand(5, 5, 5, 2).astype(np.float32)
        temp_mc2 = tempfile.NamedTemporaryFile(suffix=".mc2", delete=False)
        temp_mc2.write(test_data.tobytes())
        temp_mc2.close()

        try:
            # Test direct loading
            mc2_data = loadmc2(temp_mc2.name, [5, 5, 5, 2], "float")
            np.testing.assert_array_equal(mc2_data, test_data)

            # Test through loadfile
            file_data, header = loadfile(temp_mc2.name, [5, 5, 5, 2], "float")
            expected_log_data = np.log10(test_data)
            np.testing.assert_array_almost_equal(file_data, expected_log_data)

        finally:
            os.unlink(temp_mc2.name)


if __name__ == "__main__":
    unittest.main()
