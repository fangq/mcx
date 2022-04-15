#ifndef UBJ_INTERNAL_H
#define UBJ_INTERNAL_H

#include "ubj.h"
#include <stdlib.h>
#include <string.h>

#if _MSC_VER
#define inline __inline
#endif


static const uint8_t UBJI_TYPEC_convert[UBJ_NUM_TYPES] = "\x00ZNTFCSHiUIulmLMhdD[{";

static const int UBJI_TYPE_size[UBJ_NUM_TYPES] =
	{ -1,	 //MIXED
	0,	 //NULLTYPE
	0,	 //NOOP
	0,   //BOOL_TRUE
	0,   //BOOL_FALSE
	1,   //CHAR
	sizeof(const char*), //STRING
	sizeof(const char*), //high-precision
	1,					//INT8
	1,					//UINT8
	2,					//int16
	2,					//uint16
	4,					//int32
	4,					//uint32
	8,					//int64
	8,					//uint64
	2,					//float16
	4,					//float32
	8,					//float64
	-1,					//array
	-1					//object
	};

static const size_t UBJR_TYPE_localsize[UBJ_NUM_TYPES] =
{
	sizeof(ubjr_dynamic_t),	 //MIXED
	0,	 //NULLTYPE
	0,	 //NOOP
	0,   //BOOL_TRUE
	0,   //BOOL_FALSE
	sizeof(ubjr_string_t),   //CHAR
	sizeof(ubjr_string_t), //STRING
	sizeof(ubjr_string_t), //high-precision
	sizeof(int8_t),					//INT8
	sizeof(uint8_t),					//UINT8
	sizeof(int16_t),					//int16
	sizeof(uint16_t),					//int16
	sizeof(int32_t),					//int32
	sizeof(uint32_t),					//int32
	sizeof(int64_t),					//int64
	sizeof(uint64_t),					//int64
	sizeof(short),					//float16
	sizeof(float),					//float32
	sizeof(double),					//float64
	sizeof(ubjr_array_t),					//array
	sizeof(ubjr_object_t)					//object
};

static inline void _to_bigendian16(uint8_t* outbuffer, uint16_t input)
{
	*outbuffer++ = (input >> 8); // Get top order byte (guaranteed endian-independent since machine registers)
	*outbuffer++ = input & 0xFF; // Get bottom order byte
}
static inline void _to_bigendian32(uint8_t* outbuffer, uint32_t input)
{
	_to_bigendian16(outbuffer, (uint16_t)(input >> 16)); // Get top order 2 bytes
	_to_bigendian16(outbuffer + 2, (uint16_t)(input & 0xFFFF)); // Get bottom order 2 bytes
}
static inline void _to_bigendian64(uint8_t* outbuffer, uint64_t input)
{
	_to_bigendian32(outbuffer, (uint32_t)(input >> 32));
	_to_bigendian32(outbuffer + 4, (uint32_t)(input & 0xFFFFFFFF));
}

static inline uint8_t _is_bigendian()
{
	int i = 1;
	char *low = (char*)&i;
	return *low ? 0 : 1;
}

#define BUF_BIG_ENDIAN_SWAP(type,func,ptr,num)  \
	{											\
		size_t i;type* d = (type*)ptr; 					\
		for (i = 0; i < num; i++)				\
		{										\
			func((uint8_t*)&d[i], d[i]);		\
		}										\
	}											\

static inline void buf_endian_swap(uint8_t* buf, size_t sz, size_t n, int isbjdata)
{
	if (isbjdata == _is_bigendian())
	{
		switch (sz)
		{
		case 1:
		case 0:
			break;
		case 2:
			BUF_BIG_ENDIAN_SWAP(uint16_t, _to_bigendian16,buf,n);
			break;
		case 4:
			BUF_BIG_ENDIAN_SWAP(uint32_t, _to_bigendian32,buf,n);
			break;
		case 8:
			BUF_BIG_ENDIAN_SWAP(uint64_t, _to_bigendian64,buf,n);
			break;
		};
	}
}

//warning...null-terminated strings are assumed...when this is not necessarily valid. FIXED: we don't use null-terminated strings in the reader (NOT FIXED...string type is awkward)
static inline ubjr_dynamic_t priv_ubjr_pointer_to_dynamic(UBJ_TYPE typ, const void* dat)
{
	ubjr_dynamic_t outdyn;
	outdyn.type = typ;
	switch (typ)
	{
	case UBJ_NULLTYPE:
	case UBJ_NOOP:
		break;
	case UBJ_BOOL_TRUE:
	case UBJ_BOOL_FALSE:
		outdyn.boolean = (typ == UBJ_BOOL_TRUE ? 1 : 0);
		break;
	case UBJ_HIGH_PRECISION:
	case UBJ_STRING:
	case UBJ_CHAR://possibly if char allocate, otherwise don't
		outdyn.string = *(const ubjr_string_t*)dat;
		break;
	case UBJ_INT8:
		outdyn.integer = *(const int8_t*)dat;
		break;
	case UBJ_UINT8:
		outdyn.integer = *(const uint8_t*)dat;
		break;
	case UBJ_INT16:
		outdyn.integer = *(const int16_t*)dat;
		break;
	case UBJ_UINT16:
		outdyn.integer = *(const uint16_t*)dat;
		break;
	case UBJ_INT32:
		outdyn.integer = *(const int32_t*)dat;
		break;
	case UBJ_UINT32:
		outdyn.integer = *(const uint32_t*)dat;
		break;
	case UBJ_INT64:
		outdyn.integer = *(const int64_t*)dat;
		break;
	case UBJ_UINT64:
		outdyn.integer = *(const uint64_t*)dat;
		break;
	case UBJ_FLOAT16:
		outdyn.half = *(const uint16_t*)dat;
		break;
	case UBJ_FLOAT32:
		outdyn.real = *(const float*)dat;
		break;
	case UBJ_FLOAT64:
		outdyn.real = *(const double*)dat;
		break;
	case UBJ_ARRAY:
		outdyn.container_array = *(const ubjr_array_t*)dat;
		break;
	case UBJ_OBJECT:
		outdyn.container_object = *(const ubjr_object_t*)dat;
		break;
	case UBJ_MIXED:
		outdyn = *(const ubjr_dynamic_t*)dat;
        default:
                {}
	};
	return outdyn;
}

#endif