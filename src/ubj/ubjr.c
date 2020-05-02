#include "ubj.h"
#include "ubj_internal.h"
#include <stdlib.h>
#include <string.h>

#if _MSC_VER
#define inline __inline
#endif

typedef struct ubjr_context_t_s
{
	size_t (*read_cb )(void* data, size_t size, size_t count, void* userdata);
	int (*peek_cb )(void* userdata);
	int    (*close_cb)(void* userdata);
	void   (*error_cb)(const char* error_msg);

	void* userdata;

//	struct _ubjr_container_t container_stack[CONTAINER_STACK_MAX];
//	struct _ubjr_container_t* head;

	uint8_t ignore_container_flags;

	uint16_t last_error_code;

	size_t total_read;
} ubjr_context_t;

ubjr_context_t* ubjr_open_callback(void* userdata,
	size_t(*read_cb)(void* data, size_t size, size_t count, void* userdata),
	int(*peek_cb)(void* userdata),
	int(*close_cb)(void* userdata),
	void(*error_cb)(const char* error_msg)
	)
{
	ubjr_context_t* ctx = (ubjr_context_t*)malloc(sizeof(ubjr_context_t));
	ctx->userdata = userdata;
	ctx->read_cb = read_cb;
	ctx->peek_cb = peek_cb;
	ctx->close_cb = close_cb;
	ctx->error_cb = error_cb;


/*	ctx->head = ctx->container_stack;
	ctx->head->flags = 0;
	ctx->head->type = UBJ_MIXED;
	ctx->head->elements_remaining = 0;

	ctx->ignore_container_flags = 0;*/

	ctx->last_error_code = 0;

	ctx->total_read = 0;
	return ctx;
}

size_t ubjr_close_context(ubjr_context_t* ctx)
{
	size_t n = ctx->total_read;
	free(ctx);
	return n;
}

static inline uint8_t priv_ubjr_context_getc(ubjr_context_t* ctx)
{
	uint8_t a;
	ctx->total_read += 1;
	ctx->read_cb(&a, 1, 1, ctx->userdata);
	return a;
}

static int fpeek(void* fp)
{
    int c;
    c = fgetc(fp);
    ungetc(c, fp);

    return c;
}

ubjr_context_t* ubjr_open_file(FILE* fd)
{
	return ubjr_open_callback(fd, (void*)fread,(void*)fpeek,(void*)fclose, NULL);
}

struct mem_r_fd
{
	const uint8_t *begin, *current, *end;
};
static int memclose(void* mfd)
{
	//free(mfd);
	return 0;
}
static size_t memread(void* data, size_t size, size_t count, struct mem_r_fd* fp)
{
	size_t n = size*count;
	size_t lim = fp->end - fp->current;
	if (lim < n)
	{
		n = lim;
	}
	memcpy(data, fp->current, n);
	fp->current += n;
	return n;
}
static int mempeek(struct mem_r_fd* mfd)
{
	return *mfd->current;
}

ubjr_context_t* ubjr_open_memory(const uint8_t* be, const uint8_t* en)
{
	struct mem_r_fd* mfd = (struct mem_r_fd*)malloc(sizeof(struct mem_r_fd));
	mfd->current = be;
	mfd->begin = be;
	mfd->end = en;
	return ubjr_open_callback(mfd, (void*)memread, (void*)mempeek,(void*)memclose, NULL);
}

static inline int priv_ubjr_context_peek(ubjr_context_t* ctx)
{
	return ctx->peek_cb(ctx->userdata);
}
static inline size_t priv_ubjr_context_read(ubjr_context_t* ctx,uint8_t* dst,size_t n)
{
	size_t nr=ctx->read_cb(dst,n,1,ctx->userdata);
	ctx->total_read+=nr;
	return nr;
}

typedef struct priv_ubjr_sorted_key_t_s
{
	ubjr_string_t key;
	const uint8_t* value;

} priv_ubjr_sorted_key_t;

static int _obj_key_cmp(const void* av, const void* bv)
{
	const priv_ubjr_sorted_key_t *a, *b;
	a = (const priv_ubjr_sorted_key_t*)av;
	b = (const priv_ubjr_sorted_key_t*)bv;
	return strcmp(a->key,b->key);
}

static inline UBJ_TYPE priv_ubjr_type_from_char(uint8_t c)
{
	int i = 0;								//TODO: Benchmark this and see if it should be a switch statement where the compiler implements fastest switch e.g. binary search (17 cases might be binary search fast)
	for (i = 0; i < UBJ_NUM_TYPES && UBJI_TYPEC_convert[i] != c; i++);
	return (UBJ_TYPE)i;
}

size_t ubjr_local_type_size(UBJ_TYPE typ)
{
	return UBJR_TYPE_localsize[typ];
}


static inline priv_ubjr_sorted_key_t* priv_ubjr_object_build_sorted_keys(ubjr_object_t* obj)
{
	priv_ubjr_sorted_key_t* sorted_keysmem = malloc(obj->size*sizeof(priv_ubjr_sorted_key_t));
	size_t i;
	for (i = 0; i < obj->size; i++)
	{
		sorted_keysmem[i].key = obj->keys[i];
		sorted_keysmem[i].value = (const uint8_t*)obj->values + i*UBJR_TYPE_localsize[obj->type];
	}
	qsort(sorted_keysmem, obj->size, sizeof(priv_ubjr_sorted_key_t), _obj_key_cmp);
	return sorted_keysmem;
}

static inline uint8_t priv_ubjr_read_1b(ubjr_context_t* ctx)
{
	return priv_ubjr_context_getc(ctx);
}
static inline uint16_t priv_ubjr_read_2b(ubjr_context_t* ctx)
{
	return (uint16_t)priv_ubjr_read_1b(ctx) << 8 | (uint16_t)priv_ubjr_read_1b(ctx);
}
static inline uint32_t priv_ubjr_read_4b(ubjr_context_t* ctx)
{
	return (uint32_t)priv_ubjr_read_2b(ctx) << 16 | (uint32_t)priv_ubjr_read_2b(ctx);
}
static inline uint64_t priv_ubjr_read_8b(ubjr_context_t* ctx)
{
	return (uint64_t)priv_ubjr_read_4b(ctx) << 32 | (uint64_t)priv_ubjr_read_4b(ctx);
}

static inline int64_t priv_ubjw_read_integer(ubjr_context_t* ctx)
{
	ubjr_dynamic_t d = ubjr_read_dynamic(ctx);
	if (d.type >= UBJ_INT8 && d.type <= UBJ_INT64)
		return d.integer;
	return 0;//error
}

static inline ubjr_object_t priv_ubjr_read_raw_object(ubjr_context_t* ctx);
static inline ubjr_array_t priv_ubjr_read_raw_array(ubjr_context_t* ctx);
static inline void priv_ubjr_read_to_ptr(ubjr_context_t* ctx, uint8_t* dst, UBJ_TYPE typ)
{
	int64_t n = 1;
	char *tstr;
	switch (typ)
	{
	case UBJ_MIXED:
	{
		*(ubjr_dynamic_t*)dst = ubjr_read_dynamic(ctx);
		break;
	}
	case UBJ_STRING:
	case UBJ_HIGH_PRECISION:
	{
		n = priv_ubjw_read_integer(ctx);
	}
	case UBJ_CHAR:
	{
		tstr = malloc(n + 1);
		priv_ubjr_context_read(ctx, tstr, n);
		tstr[n] = 0;
		*(ubjr_string_t*)dst = tstr;
		break;
	}
	case UBJ_INT8:
	case UBJ_UINT8:
	{
		*dst = priv_ubjr_read_1b(ctx);
		break;
	}
	case UBJ_INT16:
	{
		*(uint16_t*)dst = priv_ubjr_read_2b(ctx);
		break;
	}
	case UBJ_INT32:
	case UBJ_FLOAT32:
	{
		*(uint32_t*)dst = priv_ubjr_read_4b(ctx);
		break;
	}
	case UBJ_INT64:
	case UBJ_FLOAT64:
	{
		*(uint64_t*)dst = priv_ubjr_read_8b(ctx);
		break;
	}
	case UBJ_ARRAY:
	{
		*(ubjr_array_t*)dst = priv_ubjr_read_raw_array(ctx);
		break;
	}
	case UBJ_OBJECT:
	{
		*(ubjr_object_t*)dst = priv_ubjr_read_raw_object(ctx);
		break;
	}
	};
}

ubjr_dynamic_t ubjr_object_lookup(ubjr_object_t* obj, const char* key)
{
	if (obj->metatable == NULL)
	{
		//memcpy(obj->sorted_keys,obj->keys)
		obj->metatable = priv_ubjr_object_build_sorted_keys(obj);
	}
	void* result=bsearch(key, obj->metatable,obj->size, sizeof(priv_ubjr_sorted_key_t),_obj_key_cmp);
	if (result == NULL)
	{
		ubjr_dynamic_t nulldyn;
		nulldyn.type = UBJ_NULLTYPE;
		return nulldyn;
	}
	const priv_ubjr_sorted_key_t* result_key = (const priv_ubjr_sorted_key_t*)result;
	return priv_ubjr_pointer_to_dynamic(obj->type,result_key->value);
}

size_t ubjr_ndarray_index(const ubjr_array_t* arr, const size_t* indices)
{
	//multi-dimensional array to linear array lookup
	size_t cstride = 1;
	size_t cdex = 0;
	uint8_t i;
	uint8_t nd = arr->num_dims;
	const size_t* dims = arr->dims;
	for (i = 0; i<nd; i++)
	{
		cdex += cstride*indices[i];
		cstride *= dims[i];
	}
	return cdex;
}



ubjr_dynamic_t ubjr_read_dynamic(ubjr_context_t* ctx)
{
	ubjr_dynamic_t scratch; //scratch memory
	UBJ_TYPE newtyp = priv_ubjr_type_from_char(priv_ubjr_context_getc(ctx));
	priv_ubjr_read_to_ptr(ctx, (uint8_t*)&scratch, newtyp);
	return priv_ubjr_pointer_to_dynamic(newtyp, &scratch);
}

static inline void priv_read_container_params(ubjr_context_t* ctx, UBJ_TYPE* typout, size_t* sizeout)
{
	int nextchar = priv_ubjr_context_peek(ctx);
	if (nextchar == '$')
	{
		priv_ubjr_context_getc(ctx);
		*typout = priv_ubjr_type_from_char(priv_ubjr_context_getc(ctx));
		nextchar = priv_ubjr_context_peek(ctx);
	}
	else
	{
		*typout = UBJ_MIXED;
	}

	if (nextchar == '#')
	{
	        int nextchar=0;
		priv_ubjr_context_getc(ctx);
		nextchar=priv_ubjr_context_peek(ctx);
		if(nextchar!='[' && nextchar!='@')
		    *sizeout = priv_ubjw_read_integer(ctx);
		else
		    *sizeout = 0;
	}
	else
	{
		*sizeout = 0;
	}
}
//TODO: This can be reused for object

static inline ubjr_array_t priv_ubjr_read_raw_array(ubjr_context_t* ctx)
{
	ubjr_array_t myarray;
	priv_read_container_params(ctx,&myarray.type,&myarray.size);
	myarray.num_dims = 1;
	myarray.dims = NULL;
	if (myarray.type != UBJ_MIXED && myarray.size==0) //params detected this is a typed array but no size was detected..possibly an N-D array?
	{
	        int nextchar=priv_ubjr_context_peek(ctx);
		if (nextchar == '@')
		{
			uint8_t dselect;
			priv_ubjr_context_getc(ctx);//skip over the '@' marker
			myarray.num_dims = priv_ubjr_context_getc(ctx);//since max is 8, no type indicator needed...always a int7 type
			myarray.dims = malloc(sizeof(size_t)*myarray.num_dims);
			myarray.size = 1;
			for (dselect = 0; dselect < myarray.num_dims; dselect++)
			{
				size_t d = priv_ubjw_read_integer(ctx);
				myarray.dims[dselect] = d;
				myarray.size *= d;
			}
		}else if(nextchar == '['){
			uint8_t dselect;
			ubjr_array_t dims;
			priv_ubjr_context_getc(ctx);//skip over the '[' marker
			dims = priv_ubjr_read_raw_array(ctx); // recursive call
			myarray.num_dims = dims.size;
			myarray.size = 1;
			myarray.dims = malloc(sizeof(size_t)*myarray.num_dims);
			for (dselect = 0; dselect < myarray.num_dims; dselect++)
			{
				size_t d=0; 
				memcpy(&d, (uint8_t*)dims.values+dselect*ubjr_local_type_size(dims.type), ubjr_local_type_size(dims.type));
				myarray.dims[dselect] = d;
				myarray.size *= d;
			}
			ubjr_cleanup_array(&dims);
		}
	}

	size_t ls = UBJR_TYPE_localsize[myarray.type];
	if (myarray.size == 0)
	{
		myarray.originally_sized = 0;
		size_t arrpot = 0;
		myarray.values=malloc(1*ls+1); //the +1 is for memory for the 0-size elements
		for (myarray.size = 0; priv_ubjr_context_peek(ctx) != ']'; myarray.size++)
		{
			if (myarray.size >= (1ULL << arrpot))
			{
				arrpot ++;
				myarray.values = realloc(myarray.values, (1ULL << arrpot)*ls+1);
			}
			priv_ubjr_read_to_ptr(ctx,(uint8_t*)myarray.values + ls*myarray.size,myarray.type);
		}
		priv_ubjr_context_getc(ctx); // read the closing ']'
	}
	else
	{
		myarray.originally_sized = 1;
		size_t i;
		myarray.values = malloc(ls*myarray.size+1);
		size_t sz = UBJI_TYPE_size[myarray.type];

		if (sz >= 0 && myarray.type != UBJ_STRING && myarray.type != UBJ_HIGH_PRECISION && myarray.type != UBJ_CHAR && myarray.type != UBJ_MIXED) //constant size,fastread
		{
			priv_ubjr_context_read(ctx, myarray.values, sz*myarray.size);
			buf_endian_swap(myarray.values, sz, myarray.size); //do nothing for 0-sized buffers
		}
		else
		{
			for (i = 0; i < myarray.size; i++)
			{
				priv_ubjr_read_to_ptr(ctx, (uint8_t*)myarray.values + ls*i, myarray.type);
			}
		}
	}
	if (myarray.dims == NULL)
	{
		myarray.dims = malloc(sizeof(size_t));
		myarray.dims[0] = myarray.size;
	}
	return myarray;
}

static inline ubjr_object_t priv_ubjr_read_raw_object(ubjr_context_t* ctx)
{
	ubjr_object_t myobject;
	myobject.metatable = NULL;
	priv_read_container_params(ctx, &myobject.type, &myobject.size);

	size_t ls = UBJR_TYPE_localsize[myobject.type];
	if (myobject.size == 0)
	{
		myobject.originally_sized = 0;
		size_t arrpot = 0;
		myobject.values = malloc(1 * ls + 1); //the +1 is for memory for the 0-size elements
		myobject.keys = malloc(1 * sizeof(ubjr_string_t));
		for (myobject.size = 0; priv_ubjr_context_peek(ctx) != '}'; myobject.size++)
		{
			if (myobject.size >= (1ULL << arrpot))
			{
				arrpot++;
				myobject.values = realloc(myobject.values, (1ULL << arrpot)*ls + 1);
				myobject.keys = realloc((uint8_t*)myobject.keys, (1ULL << arrpot)*sizeof(ubjr_string_t));
			}
			priv_ubjr_read_to_ptr(ctx, (uint8_t*)(myobject.keys + myobject.size), UBJ_STRING);
			priv_ubjr_read_to_ptr(ctx, (uint8_t*)myobject.values + ls*myobject.size, myobject.type);
		}
		priv_ubjr_context_getc(ctx); // read the closing '}'
	}
	else
	{
		size_t i;
		myobject.originally_sized = 1;
		myobject.values = malloc(ls*myobject.size + 1);
		myobject.keys = malloc(myobject.size * sizeof(ubjr_string_t));

		for (i = 0; i < myobject.size; i++)
		{
			priv_ubjr_read_to_ptr(ctx, (uint8_t*)(myobject.keys + i), UBJ_STRING);
			priv_ubjr_read_to_ptr(ctx, (uint8_t*)myobject.values + ls*i, myobject.type);
		}
	}
	return myobject;
}
static inline void priv_ubjr_cleanup_pointer(UBJ_TYPE typ,void* value);
static inline void priv_ubjr_cleanup_container(UBJ_TYPE type,size_t size,void* values)
{
	if(type == UBJ_MIXED || type == UBJ_ARRAY || type == UBJ_OBJECT || type == UBJ_STRING)
	{
		size_t ls=UBJR_TYPE_localsize[type];
		uint8_t *viter,*vend;
		viter=values;
		vend=viter+ls*size;
		for(;viter != vend;viter+=ls)
		{
			priv_ubjr_cleanup_pointer(type,(void*)viter);
		}
	}
	free(values);
}
static inline void priv_ubjr_cleanup_pointer(UBJ_TYPE typ,void* value)
{
	switch(typ)
	{
		case UBJ_MIXED:
		{
			ubjr_dynamic_t* dyn=(ubjr_dynamic_t*)value;
			switch(dyn->type)
			{
			case UBJ_STRING:
				priv_ubjr_cleanup_pointer(UBJ_STRING,&dyn->string);
				break;
			case UBJ_ARRAY:
				priv_ubjr_cleanup_pointer(UBJ_ARRAY,&dyn->container_array);
				break;
			case UBJ_OBJECT:
				priv_ubjr_cleanup_pointer(UBJ_OBJECT,&dyn->container_object);
				break;
			};
			break;
		}
		case UBJ_STRING:
		{
			ubjr_string_t* st=(ubjr_string_t*)value;
			free((void*)*st);
			break;
		}
		case UBJ_ARRAY:
		{
			ubjr_array_t* arr=(ubjr_array_t*)value;
			priv_ubjr_cleanup_container(arr->type,arr->size,arr->values);
			free(arr->dims);
			break;
		}
		case UBJ_OBJECT:
		{
			ubjr_object_t* obj=(ubjr_object_t*)value;
			priv_ubjr_cleanup_container(obj->type,obj->size,obj->values);
			priv_ubjr_cleanup_container(UBJ_STRING,obj->size,obj->keys);
			if(obj->metatable)
			{
				free(obj->metatable);
			}
			break;
		}
	};
}

void ubjr_cleanup_dynamic(ubjr_dynamic_t* dyn)
{
	priv_ubjr_cleanup_pointer(UBJ_MIXED,dyn);
}
void ubjr_cleanup_array(ubjr_array_t* arr)
{
	priv_ubjr_cleanup_pointer(UBJ_ARRAY,arr);
}
void ubjr_cleanup_object(ubjr_object_t* obj)
{
	priv_ubjr_cleanup_pointer(UBJ_OBJECT,obj);
}

