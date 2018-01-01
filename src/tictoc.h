/*******************************************************************************
**
**  \mainpage Monte Carlo eXtreme (MCX)  - GPU accelerated 3D Monte Carlo transport simulation
**
**  \author Qianqian Fang <q.fang at neu.edu>
**
**  \section sref Reference:
**  \li \c (\b Fang2009) Qianqian Fang and David A. Boas, 
**          <a href="http://www.opticsinfobase.org/abstract.cfm?uri=oe-17-22-20178">
**          "Monte Carlo Simulation of Photon Migration in 3D Turbid Media Accelerated 
**          by Graphics Processing Units,"</a> Optics Express, 17(22) 20178-20190 (2009).
**  
**  \section slicense License
**        GNU General Public License v3, see LICENSE.txt for details
**
*******************************************************************************/

/***************************************************************************//**
\file    tictoc.h

@brief   Timing function header file
*******************************************************************************/

#ifndef GETTIMEOFDAY_H
#define GETTIMEOFDAY_H

#ifdef	__cplusplus
extern "C" {
#endif

unsigned int StartTimer ();
unsigned int GetTimeMillis ();
void sleep_ms(int milliseconds);

#ifdef	__cplusplus
}
#endif

#endif /* GETTIMEOFDAY_H */

