/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2018
**
**  \section sref Reference:
**  \li \c (\b Fang2009) Qianqian Fang and David A. Boas, 
**          <a href="http://www.opticsinfobase.org/abstract.cfm?uri=oe-17-22-20178">
**          "Monte Carlo Simulation of Photon Migration in 3D Turbid Media Accelerated 
**          by Graphics Processing Units,"</a> Optics Express, 17(22) 20178-20190 (2009).
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**          "Scalable and massively parallel Monte Carlo photon transport
**           simulations for heterogeneous computing platforms," J. Biomed. Optics, 23(1), 010504, 2018.
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    mcx_bench.h

@brief   MCX builtin benchmarks
*******************************************************************************/
                                                                                         
#ifndef _MCEXTREME_BENCHMARK_H
#define _MCEXTREME_BENCHMARK_H

#define MSTR(...) #__VA_ARGS__

const char *benchname[]={"cube60","cube60b","cube60planar","skinvessel",""};
const char *benchjson[]={
MSTR(
{
    "Session": {
	"ID":       "cube60",
	"Photons":  1e6,
	"RNGSeed":  1648335518,
	"DoMismatch": 0
    },
    "Domain": {
        "Dim":    [60,60,60],
        "OriginType": 1,
        "Media": [
             {"mua": 0.00, "mus": 0.0, "g": 1.00, "n": 1.0},
             {"mua": 0.005,"mus": 1.0, "g": 0.01, "n": 1.37},
             {"mua": 0.002,"mus": 5.0, "g": 0.90, "n": 1.0}
        ]
    },
    "Shapes": [
        {"Name":     "cubic60"},
        {"Origin":   [0,0,0]},
        {"Grid":     {"Tag":1, "Size":[60,60,60]}}
    ],
    "Forward": {
	"T0": 0.0e+00,
	"T1": 5.0e-09,
	"Dt": 5.0e-09
    },
    "Optode": {
	"Source": {
	    "Type":"pencil",
	    "Pos": [29.0, 29.0, 0.0],
	    "Dir": [0.0, 0.0, 1.0]
	},
	"Detector": [
	    {
		"Pos": [29.0,  19.0,  0.0],
		"R": 1.0
	    },
            {
                "Pos": [29.0,  39.0,  0.0],
                "R": 1.0
            },
            {
                "Pos": [19.0,  29.0,  0.0],
                "R": 1.0
            },
            {
                "Pos": [39.0,  29.0,  0.0],
                "R": 1.0
            }
	]
    }
}),


MSTR(
{
    "Session": {
	"ID":       "cube60b",
	"Photons":  1e6,
	"RNGSeed":  1648335518,
	"DoMismatch": 1
    },
    "Domain": {
        "Dim":    [60,60,60],
        "OriginType": 1,
        "Media": [
             {"mua": 0.00, "mus": 0.0, "g": 1.00, "n": 1.0},
             {"mua": 0.005,"mus": 1.0, "g": 0.01, "n": 1.37},
             {"mua": 0.002,"mus": 5.0, "g": 0.90, "n": 1.0}
        ]
    },
    "Shapes": [
        {"Name":     "cube60b"},
        {"Origin":   [0,0,0]},
        {"Grid":     {"Tag":1, "Size":[60,60,60]}}
    ],
    "Forward": {
	"T0": 0.0e+00,
	"T1": 5.0e-09,
	"Dt": 5.0e-09
    },
    "Optode": {
	"Source": {
	    "Type":"pencil",
	    "Pos": [29.0, 29.0, 0.0],
	    "Dir": [0.0, 0.0, 1.0]
	},
	"Detector": [
	    {
		"Pos": [29.0,  19.0,  0.0],
		"R": 1.0
	    },
            {
                "Pos": [29.0,  39.0,  0.0],
                "R": 1.0
            },
            {
                "Pos": [19.0,  29.0,  0.0],
                "R": 1.0
            },
            {
                "Pos": [39.0,  29.0,  0.0],
                "R": 1.0
            }
	]
    }
}),


MSTR(
{
    "Session": {
	"ID":       "cube60planar",
	"Photons":  1e6,
	"RNGSeed":  1648335518,
	"DoMismatch": 1
    },
    "Domain": {
        "Dim":    [60,60,60],
        "OriginType": 1,
        "Media": [
             {"mua": 0.00, "mus": 0.0, "g": 1.00, "n": 1.0},
             {"mua": 0.005,"mus": 1.0, "g": 0.01, "n": 1.37},
             {"mua": 0.002,"mus": 5.0, "g": 0.90, "n": 1.0}
        ]
    },
    "Shapes": [
        {"Name":     "cube60planar"},
        {"Origin":   [0,0,0]},
        {"Grid":     {"Tag":1, "Size":[60,60,60]}}
    ],
    "Forward": {
	"T0": 0.0e+00,
	"T1": 5.0e-09,
	"Dt": 5.0e-09
    },
    "Optode": {
	"Source": {
	    "Type":"planar",
	    "Pos": [10.0, 10.0, -10.0],
	    "Dir": [0.0, 0.0, 1.0],
	    "Param1": [40.0, 0.0, 0.0, 0.0],
	    "Param2": [0.0, 40.0, 0.0, 0.0]
	},
	"Detector": [
	    {
		"Pos": [29.0,  19.0,  0.0],
		"R": 1.0
	    },
            {
                "Pos": [29.0,  39.0,  0.0],
                "R": 1.0
            },
            {
                "Pos": [19.0,  29.0,  0.0],
                "R": 1.0
            },
            {
                "Pos": [39.0,  29.0,  0.0],
                "R": 1.0
            }
	]
    }
}),

MSTR(
{
	"Session": {
		"ID": "skinvessel",
		"DoMismatch": 1,
		"DoAutoThread": 1,
		"Photons": 10000000
	},
	"Forward": {
		"T0": 0,
		"T1": 5e-08,
		"Dt": 5e-08
	},
	"Optode": {
		"Source": {
			"Pos": [100,100,20],
			"Dir": [0,0,1],
			"Param1": [60,0,0,0],
			"Type": "disk"
		}
	},
	"Domain": {
		"OriginType": 1,
		"LengthUnit": 0.005,
		"Media": [
			{
				"mua": 1e-05,
				"mus": 0,
				"g": 1,
				"n": 1.37
			},
			{
				"mua": 3.564e-05,
				"mus": 1,
				"g": 1,
				"n": 1.37
			},
			{
				"mua": 23.05426549,
				"mus": 9.398496241,
				"g": 0.9,
				"n": 1.37
			},
			{
				"mua": 0.04584957865,
				"mus": 35.65405549,
				"g": 0.9,
				"n": 1.37
			},
			{
				"mua": 1.657237447,
				"mus": 37.59398496,
				"g": 0.9,
				"n": 1.37
			}
		],
	        "Dim":    [200,200,200]
	},
        "Shapes": [
	       {"Grid": {"Tag":1, "Size":[200,200,200]}},
	       {"ZLayers":[[1,20,1],[21,32,4],[33,200,3]]}, 
	       {"Cylinder": {"Tag":2, "C0": [0,100.5,100.5], "C1": [200,100.5,100.5], "R": 20}}
        ]
})
};

#endif
