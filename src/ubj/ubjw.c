#include "ubj.h"
#include "ubj_internal.h"

#define CONTAINER_IS_SIZED		0x1
#define CONTAINER_IS_TYPED		0x2
#define CONTAINER_IS_UBJ_ARRAY		0x4
#define CONTAINER_IS_UBJ_OBJECT		0x8

#define CONTAINER_EXPECTS_KEY	0x10

#define CONTAINER_STACK_MAX		64
#define BUFFER_OUT_SIZE			1024

#define MAX_DIMS	8


struct priv_ubjw_container_t
{
	uint8_t flags;
	UBJ_TYPE type;
	size_t elements_remaining;
};

struct ubjw_context_t_s
{
	size_t(*write_cb)(const void* data, size_t size, size_t count, void* userdata);
	int(*close_cb)(void* userdata);
	void (*error_cb)(const char* error_msg);
	
	void* userdata;

	struct priv_ubjw_container_t container_stack[CONTAINER_STACK_MAX];
	struct priv_ubjw_container_t* head;

	uint8_t ignore_container_flags;

	uint16_t last_error_code;

	size_t total_written;
};



ubjw_context_t* ubjw_open_callback(void* userdata,
	size_t(*write_cb)(const void* data, size_t size, size_t count, void* userdata),
	int(*close_cb)(void* userdata),
	void (*error_cb)(const char* error_msg)
 					)
{
	ubjw_context_t* ctx = (ubjw_context_t*)malloc(sizeof(ubjw_context_t));
	ctx->userdata = userdata;
	ctx->write_cb = write_cb;
	ctx->close_cb = close_cb;
	ctx->error_cb = error_cb;
	
	ctx->head = ctx->container_stack;
	ctx->head->flags = 0;
	ctx->head->type = UBJ_MIXED;
	ctx->head->elements_remaining = 0;
	//ctx->head->num_dims=1;

	ctx->ignore_container_flags = 0;

	ctx->last_error_code = 0;

	ctx->total_written = 0;
	return ctx;
}
ubjw_context_t* ubjw_open_file(FILE* fd)
{
	return ubjw_open_callback(fd, (void*)fwrite,(void*)fclose,NULL);
}

struct mem_w_fd
{
	uint8_t *begin,*current, *end;
};

static int memclose(void* mfd)
{
	free(mfd);
	return 0;
}
static size_t memwrite(const void* data, size_t size, size_t count, struct mem_w_fd* fp)
{
	size_t n = size*count;
	size_t lim = fp->end - fp->current;
	if (lim < n)
	{
		n = lim;
	}
	memcpy(fp->current, data, n);
	fp->current += n;
	return n;
}

ubjw_context_t* ubjw_open_memory(uint8_t* be, uint8_t* en)
{
	struct mem_w_fd* mfd = (struct mem_w_fd*)malloc(sizeof(struct mem_w_fd));
	mfd->current = be;
	mfd->begin = be;
	mfd->end = en;
	return ubjw_open_callback(mfd, (void*)memwrite, (void*)memclose,NULL);
}

static inline void priv_ubjw_context_append(ubjw_context_t* ctx, uint8_t a)
{
	ctx->total_written += 1;
	ctx->write_cb(&a, 1, 1, ctx->userdata);
}

static inline void priv_disassembly_begin(ubjw_context_t* ctx)
{
#ifdef UBJW_DISASSEMBLY_MODE
	priv_ubjw_context_append(ctx, (uint8_t)'[');
#endif
}
static inline void priv_disassembly_end(ubjw_context_t* ctx)
{
#ifdef UBJW_DISASSEMBLY_MODE
	priv_ubjw_context_append(ctx, (uint8_t)']');
#endif
}
static inline void priv_disassembly_indent(ubjw_context_t* ctx)
{
#ifdef UBJW_DISASSEMBLY_MODE
	int n = ctx->head - ctx->container_stack;
	int i;
	priv_ubjw_context_append(ctx, (uint8_t)'\n');
	for (i = 0; i < n; i++)
	{
		priv_ubjw_context_append(ctx, (uint8_t)'\t');
	}
#endif
}

static inline void priv_ubjw_context_finish_container(ubjw_context_t* ctx, struct priv_ubjw_container_t* head)
{
	if (head->flags & CONTAINER_IS_SIZED)
	{
		if (head->elements_remaining > 0)
		{
			//error not all elements written
		}
	}
	else
	{
		priv_disassembly_begin(ctx);
		if (head->flags & CONTAINER_IS_UBJ_ARRAY)
		{
			priv_ubjw_context_append(ctx, (uint8_t)']');
		}
		else if (head->flags & CONTAINER_IS_UBJ_OBJECT)
		{
			priv_ubjw_context_append(ctx, (uint8_t)'}');
		}
		priv_disassembly_end(ctx);
	}
}

static inline void priv_ubjw_container_stack_push(ubjw_context_t* ctx, const struct priv_ubjw_container_t* cnt)
{
	size_t height = ctx->head-ctx->container_stack+1;
	if(height < CONTAINER_STACK_MAX)
	{
		*(++(ctx->head))=*cnt;
	}
	else
	{
		//todo::error
	}
}
static inline struct priv_ubjw_container_t priv_ubjw_container_stack_pop(ubjw_context_t* ctx)
{
	return *ctx->head--;
}

size_t ubjw_close_context(ubjw_context_t* ctx)
{
	while (ctx->head > ctx->container_stack)
	{
		struct priv_ubjw_container_t cnt = priv_ubjw_container_stack_pop(ctx);
		priv_ubjw_context_finish_container(ctx, &cnt);
	};
	size_t n = ctx->total_written;
	if (ctx->close_cb)
		ctx->close_cb(ctx->userdata);
	free(ctx);
	return n;
}


static inline size_t priv_ubjw_context_write(ubjw_context_t* ctx, const uint8_t* data, size_t sz)
{
	ctx->total_written += sz;
	return ctx->write_cb(data, 1, sz, ctx->userdata);
}

static inline void priv_ubjw_tag_public(ubjw_context_t* ctx, UBJ_TYPE tid)
{
	struct priv_ubjw_container_t* ch = ctx->head;
	if (!ctx->ignore_container_flags)
	{

		/*if (
			(!(ch->flags & (CONTAINER_IS_UBJ_ARRAY | CONTAINER_IS_UBJ_OBJECT))) &&
			(tid != UBJ_ARRAY && tid !=UBJ_OBJECT))
		{
			//error, only array and object can be first written
		}*/

		if (ch->flags & CONTAINER_IS_UBJ_OBJECT)
		{
			if (ch->flags & CONTAINER_EXPECTS_KEY)
			{
				//error,a key expected
				return;
			}
			ch->flags |= CONTAINER_EXPECTS_KEY; //set key expected next time in this context
		}
		else
		{
			priv_disassembly_indent(ctx);
		}

		if (ch->flags & CONTAINER_IS_SIZED)
		{
			ch->elements_remaining--; //todo: error if elements remaining is 0;
		}

		if ((ch->flags & CONTAINER_IS_TYPED) && ch->type == tid)
		{
			return;
		}
	}
	priv_disassembly_begin(ctx);
	priv_ubjw_context_append(ctx, UBJI_TYPEC_convert[tid]);
	priv_disassembly_end(ctx);
}

static inline void priv_ubjw_write_raw_string(ubjw_context_t* ctx, const char* out)//TODO: possibly use a safe string
{
	size_t n = strlen(out);
	ctx->ignore_container_flags = 1; 
	ubjw_write_integer(ctx, (int64_t)n);
	ctx->ignore_container_flags = 0;
	priv_disassembly_begin(ctx);
	priv_ubjw_context_write(ctx, (const uint8_t*)out, n);
	priv_disassembly_end(ctx);
}
void ubjw_write_string(ubjw_context_t* ctx, const char* out)
{
	priv_ubjw_tag_public(ctx,UBJ_STRING);
	priv_ubjw_write_raw_string(ctx, out);
}

static inline void priv_ubjw_write_raw_char(ubjw_context_t* ctx, char out)
{
	priv_disassembly_begin(ctx);
	priv_ubjw_context_append(ctx, (uint8_t)out);
	priv_disassembly_end(ctx);
}
void ubjw_write_char(ubjw_context_t* ctx, char out)
{
	priv_ubjw_tag_public(ctx,UBJ_CHAR);
	priv_ubjw_write_raw_char(ctx, out);
}

#ifndef min
static inline size_t min(size_t x,size_t y)
{
	return x < y ? x : y;
}
#endif

#ifdef UBJW_DISASSEMBLY_MODE
#include <stdarg.h>  
#define DISASSEMBLY_PRINT_BUFFER_SIZE 1024

static inline void priv_disassembly_print(ubjw_context_t* ctx, const char* format,...)
{
	char buffer[DISASSEMBLY_PRINT_BUFFER_SIZE];
	va_list args; 
	va_start(args, format);
	int n=vsnprintf(buffer, DISASSEMBLY_PRINT_BUFFER_SIZE, format, args);
	n = min(n, DISASSEMBLY_PRINT_BUFFER_SIZE);
	priv_ubjw_context_write(ctx, buffer,n);
	va_end(args);
}
#endif

static inline void priv_ubjw_write_raw_uint8(ubjw_context_t* ctx, uint8_t out)
{
	priv_disassembly_begin(ctx);
#ifndef UBJW_DISASSEMBLY_MODE
	priv_ubjw_context_append(ctx, out);
#else
	priv_disassembly_print(ctx, "%hhu", out);
#endif
	priv_disassembly_end(ctx);
}
void ubjw_write_uint8(ubjw_context_t* ctx, uint8_t out)
{
	priv_ubjw_tag_public(ctx,UBJ_UINT8);
	priv_ubjw_write_raw_uint8(ctx, out);
}

static inline void priv_ubjw_write_raw_int8(ubjw_context_t* ctx, int8_t out)
{
	priv_disassembly_begin(ctx);
#ifndef UBJW_DISASSEMBLY_MODE
	priv_ubjw_context_append(ctx, *(uint8_t*)&out);
#else
	priv_disassembly_print(ctx, "%hhd", out);
#endif
	priv_disassembly_end(ctx);
}
void ubjw_write_int8(ubjw_context_t* ctx, int8_t out)
{
	priv_ubjw_tag_public(ctx,UBJ_INT8);
	priv_ubjw_write_raw_int8(ctx, out);
}

static inline void priv_ubjw_write_raw_int16(ubjw_context_t* ctx, int16_t out)
{
	priv_disassembly_begin(ctx);
#ifndef UBJW_DISASSEMBLY_MODE
	uint8_t buf[2];
	_to_bigendian16(buf, *(uint16_t*)&out);
	priv_ubjw_context_write(ctx, buf, 2);
#else
	priv_disassembly_print(ctx, "%hd", out);
#endif
	priv_disassembly_end(ctx);
}
void ubjw_write_int16(ubjw_context_t* ctx, int16_t out)
{
	priv_ubjw_tag_public(ctx,UBJ_INT16);
	priv_ubjw_write_raw_int16(ctx, out);
}
static inline void priv_ubjw_write_raw_int32(ubjw_context_t* ctx, int32_t out)
{
	priv_disassembly_begin(ctx);
#ifndef UBJW_DISASSEMBLY_MODE
	uint8_t buf[4];
	_to_bigendian32(buf, *(uint32_t*)&out);
	priv_ubjw_context_write(ctx, buf, 4);
#else
	priv_disassembly_print(ctx, "%ld", out);
#endif
	priv_disassembly_end(ctx);
}
void ubjw_write_int32(ubjw_context_t* ctx, int32_t out)
{
	priv_ubjw_tag_public(ctx,UBJ_INT32);
	priv_ubjw_write_raw_int32(ctx, out);
}
static inline void priv_ubjw_write_raw_int64(ubjw_context_t* ctx, int64_t out)
{
	priv_disassembly_begin(ctx);
#ifndef UBJW_DISASSEMBLY_MODE
	uint8_t buf[8];
	_to_bigendian64(buf, *(uint64_t*)&out);
	priv_ubjw_context_write(ctx, buf, 8);
#else
	priv_disassembly_print(ctx, "%lld", out);
#endif
	priv_disassembly_end(ctx);
}
void ubjw_write_int64(ubjw_context_t* ctx, int64_t out)
{
	priv_ubjw_tag_public(ctx,UBJ_INT64);
	priv_ubjw_write_raw_int64(ctx, out);
}

void ubjw_write_high_precision(ubjw_context_t* ctx, const char* hp)
{
	priv_ubjw_tag_public(ctx,UBJ_HIGH_PRECISION);
	priv_ubjw_write_raw_string(ctx, hp);
}
UBJ_TYPE ubjw_min_integer_type(int64_t in)
{
	uint64_t mc = llabs(in);
	if (mc < 0x80)
	{
		return UBJ_INT8;
	}
	else if (in > 0 && mc < 0x100)
	{
		return UBJ_UINT8;
	}
	else if (mc < 0x8000)
	{
		return UBJ_INT16;
	}
	else if (mc < 0x80000000)
	{
		return UBJ_INT32;
	}
	else
	{
		return UBJ_INT64;
	}
}

void ubjw_write_integer(ubjw_context_t* ctx, int64_t out)
{
	switch (ubjw_min_integer_type(out))
	{
	case UBJ_INT8:
		ubjw_write_int8(ctx, (int8_t)out);
		break;
	case UBJ_UINT8:
		ubjw_write_uint8(ctx, (uint8_t)out);
		break;
	case UBJ_INT16:
		ubjw_write_int16(ctx, (int16_t)out);
		break;
	case UBJ_INT32:
		ubjw_write_int32(ctx, (int32_t)out);
		break;
	default:
		ubjw_write_int64(ctx, out);
		break;
	};
}

static inline void priv_ubjw_write_raw_float32(ubjw_context_t* ctx, float out)
{
	priv_disassembly_begin(ctx);
#ifndef UBJW_DISASSEMBLY_MODE
	uint32_t fout = *(uint32_t*)&out;
	uint8_t outbuf[4];
	_to_bigendian32(outbuf, fout);
	priv_ubjw_context_write(ctx, outbuf, 4);
#else
	priv_disassembly_print(ctx, "%g", out);
#endif
	priv_disassembly_end(ctx);

}
void ubjw_write_float32(ubjw_context_t* ctx, float out)
{
	priv_ubjw_tag_public(ctx,UBJ_FLOAT32);
	priv_ubjw_write_raw_float32(ctx, out);
}
static inline void priv_ubjw_write_raw_float64(ubjw_context_t* ctx, double out)
{
	priv_disassembly_begin(ctx);
#ifndef UBJW_DISASSEMBLY_MODE
	uint64_t fout = *(uint64_t*)&out;
	uint8_t outbuf[8];
	_to_bigendian64(outbuf, fout);
	priv_ubjw_context_write(ctx, outbuf, 8);
#else
	priv_disassembly_print(ctx, "%g", out);
#endif
	priv_disassembly_end(ctx);
}
void ubjw_write_float64(ubjw_context_t* ctx, double out)
{
	priv_ubjw_tag_public(ctx,UBJ_FLOAT64);
	priv_ubjw_write_raw_float64(ctx, out);
}

void ubjw_write_floating_point(ubjw_context_t* ctx, double out)
{
	//this may not be possible to implement correctly...for now we just write it as a float64'
	ubjw_write_float64(ctx,out);
}

void ubjw_write_noop(ubjw_context_t* ctx)
{
	priv_ubjw_tag_public(ctx,UBJ_NOOP);
}
void ubjw_write_null(ubjw_context_t* ctx)
{
	priv_ubjw_tag_public(ctx,UBJ_NULLTYPE);
}
void ubjw_write_bool(ubjw_context_t* ctx, uint8_t out)
{
	priv_ubjw_tag_public(ctx,(out ? UBJ_BOOL_TRUE : UBJ_BOOL_FALSE));
}

void priv_ubjw_begin_container(struct priv_ubjw_container_t* cnt, ubjw_context_t* ctx, UBJ_TYPE typ, const size_t *count, int ndims)
{
	cnt->flags=0;
	cnt->elements_remaining = *count;
	cnt->type = typ;

	if (typ != UBJ_MIXED)
	{
		if (*count == 0)
		{
			//error and return;
		}
		priv_disassembly_begin(ctx);
		priv_ubjw_context_append(ctx, '$');
		priv_disassembly_end(ctx);

		priv_disassembly_begin(ctx);
		priv_ubjw_context_append(ctx, UBJI_TYPEC_convert[typ]);
		priv_disassembly_end(ctx);

		cnt->flags |= CONTAINER_IS_TYPED;
	}
	if (*count != 0)
	{
		priv_disassembly_begin(ctx);
		priv_ubjw_context_append(ctx, '#');
		priv_disassembly_end(ctx);

		ctx->ignore_container_flags = 1;
		if(ndims==1)
		    ubjw_write_integer(ctx, (int64_t)(*count));
		else{
		    int i;
		    priv_disassembly_begin(ctx);
		    priv_ubjw_context_append(ctx, '[');
		    priv_disassembly_end(ctx);

		    for(i=0;i<ndims;i++)
		   	ubjw_write_integer(ctx, (int64_t)(count[i]));
		    priv_disassembly_begin(ctx);
		    priv_ubjw_context_append(ctx, ']');
		    priv_disassembly_end(ctx);
		}
		ctx->ignore_container_flags = 0;
		
		cnt->flags |= CONTAINER_IS_SIZED;
		cnt->elements_remaining = *count;
	}
}
void ubjw_begin_array(ubjw_context_t* ctx, UBJ_TYPE type, size_t count)
{
	priv_ubjw_tag_public(ctx, UBJ_ARRAY); //todo: should this happen before any erro potential?
	struct priv_ubjw_container_t ch;
	priv_ubjw_begin_container(&ch, ctx, type, &count, 1);
	ch.flags |= CONTAINER_IS_UBJ_ARRAY;
	priv_ubjw_container_stack_push(ctx, &ch);
}
void ubjw_begin_ndarray(ubjw_context_t* ctx, UBJ_TYPE type, const size_t* dims, uint8_t ndims)
{
	priv_ubjw_tag_public(ctx, UBJ_ARRAY); //todo: should this happen before any erro potential?
	struct priv_ubjw_container_t ch;
	priv_ubjw_begin_container(&ch, ctx, type, dims, ndims);
	ch.flags |= CONTAINER_IS_UBJ_ARRAY;
	priv_ubjw_container_stack_push(ctx, &ch);
}
void ubjw_write_ndbuffer(ubjw_context_t* dst, const uint8_t* data, UBJ_TYPE type, const size_t* dims, uint8_t ndims);

void ubjw_begin_object(ubjw_context_t* ctx, UBJ_TYPE type, size_t count)
{
	priv_ubjw_tag_public(ctx, UBJ_OBJECT);
	struct priv_ubjw_container_t ch;
	priv_ubjw_begin_container(&ch, ctx, type, &count, 1);
	ch.flags |= CONTAINER_EXPECTS_KEY | CONTAINER_IS_UBJ_OBJECT;
	priv_ubjw_container_stack_push(ctx, &ch);
}
void ubjw_write_key(ubjw_context_t* ctx, const char* key)
{
	if (ctx->head->flags & CONTAINER_EXPECTS_KEY && ctx->head->flags & CONTAINER_IS_UBJ_OBJECT)
	{
		priv_disassembly_indent(ctx);
		priv_ubjw_write_raw_string(ctx, key);
		ctx->head->flags ^= CONTAINER_EXPECTS_KEY; //turn off container 
	}
	else
	{
		//error unexpected key
	}
}
void ubjw_end(ubjw_context_t* ctx)
{
	struct priv_ubjw_container_t ch = priv_ubjw_container_stack_pop(ctx);
	if ((ch.flags & CONTAINER_IS_UBJ_OBJECT) && !(ch.flags & CONTAINER_EXPECTS_KEY))
	{
		//error expected value
	}
	priv_disassembly_indent(ctx);
	priv_ubjw_context_finish_container(ctx, &ch);
}


static inline void priv_ubjw_write_byteswap(ubjw_context_t* ctx, const uint8_t* data, int sz, size_t count)
{
	uint8_t buf[BUFFER_OUT_SIZE];

	size_t i;
	size_t nbytes = sz*count;
	for (i = 0; i < nbytes; i+=BUFFER_OUT_SIZE)
	{
		size_t npass = min(nbytes - i, BUFFER_OUT_SIZE);
		memcpy(buf, data + i, npass);
		buf_endian_swap(buf, sz, npass/sz);
		priv_ubjw_context_write(ctx, buf, npass);
	}
}
void ubjw_write_buffer(ubjw_context_t* ctx, const uint8_t* data, UBJ_TYPE type, size_t count)
{
	int typesz = UBJI_TYPE_size[type];
	if (typesz < 0)
	{
		//error cannot write an STC buffer of this type.
	}
	ubjw_begin_array(ctx, type, count);
	if (type == UBJ_STRING || type == UBJ_HIGH_PRECISION)
	{
		const char** databufs = (const char**)data;
		size_t i;
		for (i = 0; i < count; i++)
		{
			priv_ubjw_write_raw_string(ctx, databufs[i]);
		}
	}
#ifndef UBJW_DISASSEMBLY_MODE
	else if (typesz == 1 || _is_bigendian())
	{
		size_t n = typesz*count;
		priv_ubjw_context_write(ctx, data, n);
	}
	else if (typesz > 1) //and not big-endian
	{
		priv_ubjw_write_byteswap(ctx, data,typesz,count);
	}
#else
	else
	{
		size_t i;
		for (i = 0; i < count; i++)
		{
			ubjr_dynamic_t dyn = priv_ubjr_pointer_to_dynamic(type, data + i*typesz);
			ubjrw_write_dynamic(ctx, dyn, 0);
		}
	}
#endif
	ubjw_end(ctx);
}
