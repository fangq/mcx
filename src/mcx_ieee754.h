#ifndef MCX_IEEE754_H
#define MCX_IEEE754_H

#define MCX_LITTLE_ENDIAN 0x41424344UL 
#define MCX_BIG_ENDIAN    0x44434241UL
#define MCX_PDP_ENDIAN    0x42414443UL
#define MCX_BYTE_ORDER    ('ABCD') 

union ieee754_double
  {
    double d;

    /* This is the IEEE 754 double-precision format.  */
    struct
      {
#if	MCX_BYTE_ORDER == MCX_BIG_ENDIAN
	unsigned int negative:1;
	unsigned int exponent:11;
	/* Together these comprise the mantissa.  */
	unsigned int mantissa0:20;
	unsigned int mantissa1:32;
#else
	/* Together these comprise the mantissa.  */
	unsigned int mantissa1:32;
	unsigned int mantissa0:20;
	unsigned int exponent:11;
	unsigned int negative:1;
#endif				/* Little endian.  */
      } ieee;

    /* This format makes it easier to see if a NaN is a signalling NaN.  */
    struct
      {
#if	MCX_BYTE_ORDER == MCX_BIG_ENDIAN
	unsigned int negative:1;
	unsigned int exponent:11;
	unsigned int quiet_nan:1;
	/* Together these comprise the mantissa.  */
	unsigned int mantissa0:19;
	unsigned int mantissa1:32;
#else
	/* Together these comprise the mantissa.  */
	unsigned int mantissa1:32;
	unsigned int mantissa0:19;
	unsigned int quiet_nan:1;
	unsigned int exponent:11;
	unsigned int negative:1;
#endif
      } ieee_nan;
  };

#define IEEE754_DOUBLE_BIAS	0x3ff /* Added to exponent.  */

#endif
