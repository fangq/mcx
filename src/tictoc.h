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

