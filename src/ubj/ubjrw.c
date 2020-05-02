#include "ubj_internal.h"

static uint32_t compute_typemask(ubjr_dynamic_t* vals, size_t sz)
{
	uint32_t typemask = 0;
	size_t i;
	for (i = 0; i < sz; i++)
	{
		typemask |= 1UL << vals[i].type;
	}
	return typemask;
}

static inline UBJ_TYPE typemask2type(uint32_t v)
{
	unsigned int r = 0; // r will be lg(v)

	while (v >>= 1) // unroll for more speed...
	{
		r++;
	}
	return (UBJ_TYPE)r;
}
static UBJ_TYPE compute_best_integer_type(ubjr_dynamic_t* vals, size_t sz)
{
	uint32_t typemask = 0;
	size_t i;
	for (i = 0; i < sz; i++)
	{
		typemask |= 1UL << ubjw_min_integer_type(vals[i].integer);
	}
	return typemask2type(typemask);
}
static uint32_t compute_best_string_type(ubjr_dynamic_t* vals, size_t sz)
{
	size_t i;
	for (i = 0; i < sz; i++)
	{
		if (strlen(vals[i].string) > 1)
		{
			return UBJ_STRING;
		}
	}
	return UBJ_CHAR;
}
static UBJ_TYPE optimize_type(UBJ_TYPE typein,ubjr_dynamic_t* vals, size_t sz)
{
	static const uint32_t intmask = (1 << UBJ_INT8) | (1 << UBJ_UINT8) | (1 << UBJ_INT16) | (1 << UBJ_INT32) | (1 << UBJ_INT64);
	static const uint32_t stringmask = (1 << UBJ_STRING) | (1 << UBJ_CHAR);
	if (typein != UBJ_MIXED)
		return typein;
	//integer optimization can be done here...
	uint32_t tm = compute_typemask(vals, sz);
	if ((tm & intmask) == tm) //if all values are integers
	{
		return compute_best_integer_type(vals,sz);	//calculate the optimum type given the data
	}
	else if ((tm & stringmask) == tm)
	{
		return compute_best_string_type(vals,sz);
	}
	else if(tm && !(tm & (tm- 1)))  //if only one bit is set in typemask
	{
		return typemask2type(tm); //figure out which bit is set.
	}
	else
	{
		return UBJ_MIXED;
	}
}

void ubjrw_write_dynamic(ubjw_context_t* ctx, ubjr_dynamic_t dobj,uint8_t optimize)
{
	UBJ_TYPE ctyp,otyp;
	size_t csize;
	uint8_t* cvalues;
	switch (dobj.type)
	{
	case UBJ_MIXED:
		return;///error, can't be mixed
	case UBJ_NULLTYPE:
		ubjw_write_null(ctx);
		return;
	case UBJ_NOOP:
		ubjw_write_noop(ctx);
		return;
	case UBJ_BOOL_FALSE:
		ubjw_write_bool(ctx, 0);
		return;
	case UBJ_BOOL_TRUE:
		ubjw_write_bool(ctx, 1);
		return;
	case UBJ_CHAR:
		ubjw_write_char(ctx, *dobj.string);//first character of string
		return;
	case UBJ_STRING:
		ubjw_write_string(ctx, dobj.string);
		return;
	case UBJ_HIGH_PRECISION:
		ubjw_write_high_precision(ctx, dobj.string);
		return;
	case UBJ_INT8:
		ubjw_write_int8(ctx, (int8_t)dobj.integer);
		return;
	case UBJ_UINT8:
		ubjw_write_uint8(ctx, (uint8_t)dobj.integer);
		return;
	case UBJ_INT16:
		ubjw_write_int16(ctx, (int16_t)dobj.integer);
		return;
	case UBJ_INT32:
		ubjw_write_int32(ctx, (int32_t)dobj.integer);
		return;
	case UBJ_INT64:
		ubjw_write_int64(ctx, dobj.integer);
		return;
	case UBJ_FLOAT32:
		ubjw_write_float32(ctx, (float)dobj.real);
		return;
	case UBJ_FLOAT64:
		ubjw_write_float64(ctx, dobj.real);
		return;
	case UBJ_ARRAY:
		if ((dobj.container_array.originally_sized || optimize) //if we optimize an unsized array to a sized one or the original is sized
			&& dobj.container_array.type != UBJ_MIXED 
			&& dobj.container_array.type != UBJ_OBJECT 
			&& dobj.container_array.type != UBJ_ARRAY && dobj.container_array.num_dims<=1)
		{
			ubjw_write_buffer(ctx, dobj.container_array.values, dobj.container_array.type, dobj.container_array.size);
			return;
		}
		else
		{
			ctyp = dobj.container_array.type;
			csize = dobj.container_array.size;
			cvalues = dobj.container_array.values;
			otyp = optimize ? optimize_type(ctyp,(ubjr_dynamic_t*)cvalues,csize) : ctyp;
			if(dobj.container_array.num_dims<=1)
			    ubjw_begin_array(ctx, otyp, (dobj.container_array.originally_sized || optimize) ? csize : 0);
			else
			    ubjw_begin_ndarray(ctx, otyp, dobj.container_array.dims, dobj.container_array.num_dims);
			break;
		}
	case UBJ_OBJECT:
		{
			ctyp = dobj.container_object.type;
			csize = dobj.container_object.size;
			cvalues = dobj.container_object.values;
			otyp = optimize ? optimize_type(ctyp, (ubjr_dynamic_t*)cvalues, csize) : ctyp;
			ubjw_begin_object(ctx, otyp, (dobj.container_object.originally_sized || optimize) ? csize : 0);
			break;
		}
	};
	{
		size_t i;
		ubjr_dynamic_t scratch;
		size_t ls = UBJR_TYPE_localsize[ctyp];

		for (i = 0; i < csize; i++)
		{
			if (dobj.type == UBJ_OBJECT)
			{
				ubjw_write_key(ctx, dobj.container_object.keys[i]);
			}
			scratch = priv_ubjr_pointer_to_dynamic(ctyp, cvalues + ls*i);
			scratch.type = (otyp == UBJ_MIXED ? scratch.type : otyp);
			ubjrw_write_dynamic(ctx, scratch,optimize);
		}
		ubjw_end(ctx);
	}

}
