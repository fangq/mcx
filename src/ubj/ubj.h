#ifndef UBJ_H
#define UBJ_H

#ifdef __cplusplus
extern "C" {
#endif

#include<inttypes.h>
#include<stdio.h>

typedef enum
{
	UBJ_MIXED=0,			//NOT a type...or the type is mixed

	UBJ_NULLTYPE,
	UBJ_NOOP,
	UBJ_BOOL_TRUE,
	UBJ_BOOL_FALSE,
	
	UBJ_CHAR,
	UBJ_STRING,
	UBJ_HIGH_PRECISION,

	UBJ_INT8,
	UBJ_UINT8 ,
	UBJ_INT16,
	UBJ_UINT16,
	UBJ_INT32,
	UBJ_UINT32,
	UBJ_INT64,
	UBJ_UINT64,
	UBJ_FLOAT16,
	UBJ_FLOAT32,
	UBJ_FLOAT64,

	UBJ_ARRAY,
	UBJ_OBJECT,

	UBJ_NUM_TYPES				//this is the size of how many types there are (chris' trick)
} UBJ_TYPE;

typedef union ubj_float16
{
    uint8_t  byte[2];
    uint16_t val;
} bjd_half;

//////////here is the declarations for the writer API////////////////////////////////////



/////////////////////////////////////////////////////////////////////////////////////////

struct ubjw_context_t_s;
typedef struct ubjw_context_t_s ubjw_context_t;

ubjw_context_t* ubjw_open_callback(void* userdata,
	size_t(*write_cb)(const void* data, size_t size, size_t count, void* userdata),
	int(*close_cb)(void* userdata),
	void(*error_cb)(const char* error_msg)
	);
ubjw_context_t* ubjw_open_file(FILE*);
ubjw_context_t* ubjw_open_memory(uint8_t* dst_b, uint8_t* dst_e);

size_t ubjw_close_context(ubjw_context_t* ctx);

void ubjw_write_string(ubjw_context_t* dst, const char* out);
void ubjw_write_char(ubjw_context_t* dst, char out);

void ubjw_write_int8(ubjw_context_t* dst, int8_t out);
void ubjw_write_uint8(ubjw_context_t* dst, uint8_t out);
void ubjw_write_int16(ubjw_context_t* dst, int16_t out);
void ubjw_write_uint16(ubjw_context_t* dst, uint16_t out);
void ubjw_write_int32(ubjw_context_t* dst, int32_t out);
void ubjw_write_uint32(ubjw_context_t* dst, uint32_t out);
void ubjw_write_int64(ubjw_context_t* dst, int64_t out);
void ubjw_write_uint64(ubjw_context_t* dst, uint64_t out);
void ubjw_write_high_precision(ubjw_context_t* dst, const char* hp);

void ubjw_write_integer(ubjw_context_t* dst, int64_t out);
UBJ_TYPE ubjw_min_integer_type(int64_t in);

void ubjw_write_float16(ubjw_context_t* dst, uint16_t out);
void ubjw_write_float32(ubjw_context_t* dst, float out);
void ubjw_write_float64(ubjw_context_t* dst, double out);

void ubjw_write_floating_point(ubjw_context_t* dst, double out);

void ubjw_write_noop(ubjw_context_t* dst);
void ubjw_write_null(ubjw_context_t* dst);
void ubjw_write_bool(ubjw_context_t* dst, uint8_t out);

void ubjw_begin_array(ubjw_context_t* dst, UBJ_TYPE type, size_t count);

void ubjw_begin_object(ubjw_context_t* dst, UBJ_TYPE type, size_t count);
void ubjw_write_key(ubjw_context_t* dst, const char* key);
void ubjw_end(ubjw_context_t* dst);

//output an efficient buffer of types
void ubjw_write_buffer(ubjw_context_t* dst, const uint8_t* data, UBJ_TYPE type, size_t count);

//Proposal for N-D arrays
void ubjw_begin_ndarray(ubjw_context_t* dst, UBJ_TYPE type, const size_t* dims, uint8_t ndims);
void ubjw_write_ndbuffer(ubjw_context_t* dst,const uint8_t* data, UBJ_TYPE type, const size_t* dims, uint8_t ndims);


//////////here is the declarations for the reader API////////////////////////////////////



/////////////////////////////////////////////////////////////////////////////////////////

struct ubjr_context_t_s;
typedef struct ubjr_context_t_s ubjr_context_t;

//Open up a reader context for reading using a custom calllback
ubjr_context_t* ubjr_open_callback(void* userdata,
	size_t(*read_cb)(void* data, size_t size, size_t count, void* userdata),
	int(*peek_cb)(void* userdata),
	int(*close_cb)(void* userdata),
	void(*error_cb)(const char* error_msg)
	);

//Open a context initialized to a UBJ file
ubjr_context_t* ubjr_open_file(FILE*);

//Open up a context initialized to a memory dump of a UBJ file (or a segment of a UBJ file)
ubjr_context_t* ubjr_open_memory(const uint8_t* dst_b, const uint8_t* dst_e);

//Close a reader context 
size_t ubjr_close_context(ubjr_context_t* ctx);

typedef char* ubjr_string_t;

//An array that you read from the stream
typedef struct ubjr_array_t_s
{
	uint8_t originally_sized;
	UBJ_TYPE type;	
	size_t size;	//total number of elements
	void* values;
	uint8_t num_dims;
	size_t* dims;	//this could be faster if it was constant size, but would also make the size of the dynamic object a LOT bigger

} ubjr_array_t;

//a map that you read from the stream
typedef struct ubjr_object_t_s
{
	uint8_t originally_sized;
	UBJ_TYPE type;
	size_t size;
	void* values;
	ubjr_string_t* keys;
	void* metatable;		//don't use this..only useful for computing object_lookup
} ubjr_object_t;

//a dynamic type that you parsed.
typedef struct ubjr_dynamic_t_s
{
	UBJ_TYPE type;
	union
	{
		uint8_t boolean;
		uint16_t half;
		double real;
		int64_t integer;
		ubjr_string_t string;
		ubjr_array_t container_array;
		ubjr_object_t container_object;
	};
} ubjr_dynamic_t;

//Parse a dynamic object from the stream
ubjr_dynamic_t ubjr_read_dynamic(ubjr_context_t* ctx);
void ubjr_cleanup_dynamic(ubjr_dynamic_t* dyn);

ubjr_dynamic_t ubjr_object_lookup(ubjr_object_t* obj, const char* key);
size_t ubjr_local_type_size(UBJ_TYPE typ);//should be equivalent to sizeof()
size_t ubjr_ndarray_index(const ubjr_array_t* arr, const size_t* indices);


//output an efficient buffer of types
///void ubjr_read_buffer(struct ubjr_context_t* dst, const uint8_t* data, UBJ_TYPE type, size_t count);

void ubjr_cleanup_dynamic(ubjr_dynamic_t* dyn);
void ubjr_cleanup_array(ubjr_array_t* arr);
void ubjr_cleanup_object(ubjr_object_t* obj);



///////UBJ_RW api

void ubjrw_write_dynamic(ubjw_context_t* ctx, ubjr_dynamic_t dobj,uint8_t optimize);
//ubjrw_append_object(ubjw_context_t* ctx, ubjr_dynamic_t dobj);
//ubjrw_append_array(ubjw_context_t* ctx, ubjr_dynamic_t dobj);

#ifdef __cplusplus
}

#include<iostream>

static size_t write_os(const void* data, size_t size, size_t count, void* userdata)
{
	size_t n = size*count;
	reinterpret_cast<std::ostream*>(userdata)->write(data, n);
	return n;
}
static void close_os(void* userdata)
{
	reinterpret_cast<std::ostream*>(userdata)->close();
}

static size_t read_is(void* data, size_t size, size_t count, void* userdata)
{
	size_t n = size*count;
	reinterpret_cast<std::istream*>(userdata)->read(data, n);
	return n;
}
static int peek_is(void* userdata)
{
	return reinterpret_cast<std::istream*>(userdata)->peek();
}
static void close_is(void* userdata)
{
	reinterpret_cast<std::istream*>(userdata)->close();
}

static ubjw_context_t* ubjw_open_stream(std::ostream& outstream)
{
	return ubjw_open_callback((void*)&outstream, write_os, close_os, NULL);
}

static ubjr_context_t* ubjr_open_stream(std::istream& instream)
{
	return ubjr_open_callback((void*)&instream, read_is, peek_is, close_is, NULL);
}



#endif

#endif
