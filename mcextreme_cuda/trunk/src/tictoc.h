#ifndef GETTIMEOFDAY_H
#define GETTIMEOFDAY_H

static unsigned int timerRes;
#ifndef _WIN32
#include <unistd.h>
#include <sys/time.h>
#include <string.h>
void SetupMillisTimer(void) {}
void CleanupMillisTimer(void) {}
int GetTime (void) {
  struct timeval tv;
  timerRes = 1000;
  gettimeofday(&tv,NULL);
  int temp = tv.tv_usec;
  temp+=tv.tv_sec*1000000;
  return temp;
}
unsigned int GetTimeMillis () {
  return (unsigned int)(GetTime ()/1000);
}

#else
#include <windows.h>

/*
 * GetTime --
 *
 *      Returns the curent time (from some uninteresting origin) in usecs
 *      based on the performance counters.
 */

int GetTime(void)
{
   static double cycles_per_usec;
   LARGE_INTEGER counter;

   if (cycles_per_usec == 0) {
      static LARGE_INTEGER lFreq;
      if (!QueryPerformanceFrequency(&lFreq)) {
         fprintf(stderr, "Unable to read the performance counter frquency!\n");
         return 0;
      }

      cycles_per_usec = 1000000 / ((double) lFreq.QuadPart);
   }

   if (!QueryPerformanceCounter(&counter)) {
      fprintf(stderr,"Unable to read the performance counter!\n");
      return 0;
   }

   return ((int) (((double) counter.QuadPart) * cycles_per_usec));
}

// Tim is evil...
#pragma comment(lib,"winmm")

unsigned int GetTimeMillis(void) {
  return (unsigned int)timeGetTime();
}

//  By default in 2000/XP, the timeGetTime call is set to some resolution
// between 10-15 ms query for the range of value periods and then set timer
// to the lowest possible.  Note: MUST make call to corresponding
// CleanupMillisTimer
void SetupMillisTimer(void) {

  TIMECAPS timeCaps;
  timeGetDevCaps(&timeCaps, sizeof(TIMECAPS)); 

  if (timeBeginPeriod(timeCaps.wPeriodMin) == TIMERR_NOCANDO) {
    fprintf(stderr,"WARNING: Cannot set timer precision.  Not sure what precision we're getting!\n");
  }
  else {
    timerRes = timeCaps.wPeriodMin;
    fprintf(stderr,"(* Set timer resolution to %d ms. *)\n",timeCaps.wPeriodMin);
  }

}

void CleanupMillisTimer(void) {
  if (timeEndPeriod(timerRes) == TIMERR_NOCANDO) {
    fprintf(stderr,"WARNING: bad return value of call to timeEndPeriod.\n");
  }
}

#endif
#endif /* GETTIMEOFDAY_H */

