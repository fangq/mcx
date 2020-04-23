/***************************************************************************//**
**  \mainpage ZMat Library
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2019-2020
**
**  \section sref URL: http://github.com/fangq/zmat
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    zmatlib.c

@brief   A portable data compression library of zlib, lz and easylzma
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include "zmatlib.h"

#include "zlib.h"

char *zmat_errcode[]={
	"No error", 
	"input can not be empty", 
	"failed to initialize zlib", 
	"zlib error, see info.status for error flag, often a result of mismatch in compression method", 
	"easylzma error, see info.status for error flag, often a result of mismatch in compression method",
        "can not allocate output buffer",
        "lz4 error, see info.status for error flag, often a result of mismatch in compression method",
	"unsupported method"
	};

    
char * zmat_error(int id){
    if(id>=0 && id<(sizeof(zmat_errcode) / sizeof(zmat_errcode[0])))
        return zmat_errcode[id];
    else
        return "unknown error";
}

int zmat_run(const size_t inputsize, unsigned char *inputstr, size_t *outputsize, unsigned char **outputbuf, const int zipid, int *ret, const int iscompress){
       z_stream zs;
       size_t buflen[2]={0};
       *outputbuf=NULL;

       zs.zalloc = Z_NULL;
       zs.zfree = Z_NULL;
       zs.opaque = Z_NULL;

       if(inputsize==0)
	    return -1;

       if(iscompress){
            if(zipid==zmBase64){
	        *outputbuf=base64_encode((const unsigned char*)inputstr, inputsize, outputsize);
            }else if(zipid==zmZlib || zipid==zmGzip){
		if(zipid==zmZlib){
	            if(deflateInit(&zs, Z_DEFAULT_COMPRESSION) != Z_OK)
	        	return -2;
		}else{
	            if(deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 15|16, MAX_MEM_LEVEL, Z_DEFAULT_STRATEGY) != Z_OK)
	        	return -2;
		}

		buflen[0] =deflateBound(&zs,inputsize);
		*outputbuf=(unsigned char *)malloc(buflen[0]);
		zs.avail_in = inputsize; /* size of input, string + terminator*/
		zs.next_in = (Bytef *)inputstr; /* input char array*/
		zs.avail_out = buflen[0]; /* size of output*/

		zs.next_out =  (Bytef *)(*outputbuf); /*(Bytef *)(); // output char array*/

		*ret=deflate(&zs, Z_FINISH);
		*outputsize=zs.total_out;
		if(*ret!=Z_STREAM_END && *ret!=Z_OK)
	            return -3;
		deflateEnd(&zs);
#ifndef NO_LZMA
	    }else if(zipid==zmLzma || zipid==zmLzip){
	        *ret = simpleCompress((elzma_file_format)(zipid-3), (unsigned char *)inputstr,
				inputsize, outputbuf, outputsize);
		if(*ret!=ELZMA_E_OK)
		     return -4;
#endif
#ifndef NO_LZ4
            }else if(zipid==zmLz4 || zipid==zmLz4hc){
                *outputsize=LZ4_compressBound(inputsize);
                if (!(*outputbuf = (unsigned char *)malloc(*outputsize)))
                     return -5;
		if(zipid==zmLz4)
	            *outputsize = LZ4_compress_default((const char *)inputstr, (char*)(*outputbuf), inputsize, *outputsize);
		else
		    *outputsize = LZ4_compress_HC((const char *)inputstr, (char*)(*outputbuf), inputsize, *outputsize, 8);
		*ret=*outputsize;
		if(*outputsize==0)
		     return -6;
#endif
	    }else{
		return -7;
	    }
       }else{
            if(zipid==zmBase64){
	        *outputbuf=base64_decode((const unsigned char*)inputstr, inputsize, outputsize);
            }else if(zipid==zmZlib || zipid==zmGzip){
	        int count=1;
        	if(zipid==zmZlib){
	            if(inflateInit(&zs) != Z_OK)
	        	return -2;
		}else{
	            if(inflateInit2(&zs, 15|32) != Z_OK)
	        	return -2;
		}
		buflen[0] =inputsize*20;
		*outputbuf=(unsigned char *)malloc(buflen[0]);

		zs.avail_in = inputsize; /* size of input, string + terminator*/
		zs.next_in =inputstr; /* input char array*/
		zs.avail_out = buflen[0]; /* size of output*/

		zs.next_out =  (Bytef *)(*outputbuf); /*(Bytef *)(); // output char array*/

        	while((*ret=inflate(&zs, Z_SYNC_FLUSH))!=Z_STREAM_END && count<=10){
		    *outputbuf=(unsigned char *)realloc(*outputbuf, (buflen[0]<<count));
		    zs.next_out =  (Bytef *)(*outputbuf+(buflen[0]<<(count-1)));
		    zs.avail_out = (buflen[0]<<(count-1)); /* size of output*/
		    count++;
		}
		*outputsize=zs.total_out;

		if(*ret!=Z_STREAM_END && *ret!=Z_OK)
		    return -3;
		inflateEnd(&zs);
#ifndef NO_LZMA
            }else if(zipid==zmLzma || zipid==zmLzip){
	        *ret = simpleDecompress((elzma_file_format)(zipid-3), (unsigned char *)inputstr,
				inputsize, outputbuf, outputsize);
		if(*ret!=ELZMA_E_OK)
		     return -4;
#endif
#ifndef NO_LZ4
            }else if(zipid==zmLz4 || zipid==zmLz4hc){
	        int count=2;
                *outputsize=(inputsize<<count);
                if (!(*outputbuf = (unsigned char *)malloc(*outputsize))){
		     *ret=-5;
                     return *ret;
		}
        	while((*ret=LZ4_decompress_safe((const char *)inputstr, (char*)(*outputbuf), inputsize, *outputsize))<=0 && count<=10){
		     *outputsize=(inputsize<<count);
                     if (!(*outputbuf = (unsigned char *)realloc(*outputbuf, *outputsize))){
		        *ret=-5;
                         return *ret;
		     }
                     count++;
		}
		*outputsize=*ret;
		if(*ret<0)
		     return -6;
#endif
	    }else{
		return -7;
	    }
       }
       return 0;
}

int zmat_encode(const size_t inputsize, unsigned char *inputstr, size_t *outputsize, unsigned char **outputbuf, const int zipid, int *ret){
    return zmat_run(inputsize, inputstr, outputsize, outputbuf, zipid, ret, 1);
}
int zmat_decode(const size_t inputsize, unsigned char *inputstr, size_t *outputsize, unsigned char **outputbuf, const int zipid, int *ret){
    return zmat_run(inputsize, inputstr, outputsize, outputbuf, zipid, ret, 0);
}

/**
 * @brief Look up a string in a string list and return the index
 *
 * @param[in] origkey: string to be looked up
 * @param[out] table: the dictionary where the string is searched
 * @return if found, return the index of the string in the dictionary, otherwise -1.
 */

int zmat_keylookup(char *origkey, const char *table[]){
    int i=0;
    char *key=(char *)malloc(strlen(origkey)+1);
    memcpy(key,origkey,strlen(origkey)+1);
    while(key[i]){
        key[i]=tolower(key[i]);
	i++;
    }
    i=0;
    while(table[i] && table[i][0]!='\0'){
	if(strcmp(key,table[i])==0){
	        free(key);
		return i;
	}
	i++;
    }
    free(key);
    return -1;
}


/*
 * Base64 encoding/decoding (RFC1341)
 * Copyright (c) 2005-2011, Jouni Malinen <j@w1.fi>
 *
 * This software may be distributed under the terms of the BSD license.
 * See README for more details.
 */

static const unsigned char base64_table[65] =
	"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/**
 * base64_encode - Base64 encode
 * @src: Data to be encoded
 * @len: Length of the data to be encoded
 * @out_len: Pointer to output length variable, or %NULL if not used
 * Returns: Allocated buffer of out_len bytes of encoded data,
 * or %NULL on failure
 *
 * Caller is responsible for freeing the returned buffer. Returned buffer is
 * nul terminated to make it easier to use as a C string. The nul terminator is
 * not included in out_len.
 */
unsigned char * base64_encode(const unsigned char *src, size_t len,
			      size_t *out_len)
{
	unsigned char *out, *pos;
	const unsigned char *end, *in;
	size_t olen;
	int line_len;

	olen = len * 4 / 3 + 4; /* 3-byte blocks to 4-byte */
	olen += olen / 72; /* line feeds */
	olen++; /* nul termination */
	if (olen < len)
		return NULL; /* integer overflow */
	out = (unsigned char *)malloc(olen);
	if (out == NULL)
		return NULL;

	end = src + len;
	in = src;
	pos = out;
	line_len = 0;
	while (end - in >= 3) {
		*pos++ = base64_table[in[0] >> 2];
		*pos++ = base64_table[((in[0] & 0x03) << 4) | (in[1] >> 4)];
		*pos++ = base64_table[((in[1] & 0x0f) << 2) | (in[2] >> 6)];
		*pos++ = base64_table[in[2] & 0x3f];
		in += 3;
		line_len += 4;
		if (line_len >= 72) {
			*pos++ = '\n';
			line_len = 0;
		}
	}

	if (end - in) {
		*pos++ = base64_table[in[0] >> 2];
		if (end - in == 1) {
			*pos++ = base64_table[(in[0] & 0x03) << 4];
			*pos++ = '=';
		} else {
			*pos++ = base64_table[((in[0] & 0x03) << 4) |
					      (in[1] >> 4)];
			*pos++ = base64_table[(in[1] & 0x0f) << 2];
		}
		*pos++ = '=';
		line_len += 4;
	}

	if (line_len)
		*pos++ = '\n';

	*pos = '\0';
	if (out_len)
		*out_len = pos - out;
	return out;
}


/**
 * base64_decode - Base64 decode
 * @src: Data to be decoded
 * @len: Length of the data to be decoded
 * @out_len: Pointer to output length variable
 * Returns: Allocated buffer of out_len bytes of decoded data,
 * or %NULL on failure
 *
 * Caller is responsible for freeing the returned buffer.
 */
unsigned char * base64_decode(const unsigned char *src, size_t len,
			      size_t *out_len)
{
	unsigned char dtable[256], *out, *pos, block[4], tmp;
	size_t i, count, olen;
	int pad = 0;

	memset(dtable, 0x80, 256);
	for (i = 0; i < sizeof(base64_table) - 1; i++)
		dtable[base64_table[i]] = (unsigned char) i;
	dtable['='] = 0;

	count = 0;
	for (i = 0; i < len; i++) {
		if (dtable[src[i]] != 0x80)
			count++;
	}

	if (count == 0 || count % 4)
		return NULL;

	olen = count / 4 * 3;
	pos = out = (unsigned char *)malloc(olen);
	if (out == NULL)
		return NULL;

	count = 0;
	for (i = 0; i < len; i++) {
		tmp = dtable[src[i]];
		if (tmp == 0x80)
			continue;

		if (src[i] == '=')
			pad++;
		block[count] = tmp;
		count++;
		if (count == 4) {
			*pos++ = (block[0] << 2) | (block[1] >> 4);
			*pos++ = (block[1] << 4) | (block[2] >> 2);
			*pos++ = (block[2] << 6) | block[3];
			count = 0;
			if (pad) {
				if (pad == 1)
					pos--;
				else if (pad == 2)
					pos -= 2;
				else {
					/* Invalid padding */
					free(out);
					return NULL;
				}
				break;
			}
		}
	}

	*out_len = pos - out;
	return out;
}

#ifndef NO_LZMA

struct dataStream 
{
    const unsigned char * inData;
    size_t inLen;

    unsigned char * outData;
    size_t outLen;
};

static int
inputCallback(void *ctx, void *buf, size_t * size)
{
    size_t rd = 0;
    struct dataStream * ds = (struct dataStream *) ctx;
    assert(ds != NULL);
    
    rd = (ds->inLen < *size) ? ds->inLen : *size;

    if (rd > 0) {
        memcpy(buf, (void *) ds->inData, rd);
        ds->inData += rd;
        ds->inLen -= rd;
    }

    *size = rd;

    return 0;
}

static size_t
outputCallback(void *ctx, const void *buf, size_t size)
{
    struct dataStream * ds = (struct dataStream *) ctx;
    assert(ds != NULL);
    
    if (size > 0) {
        ds->outData = (unsigned char *)realloc(ds->outData, ds->outLen + size);
        memcpy((void *) (ds->outData + ds->outLen), buf, size);
        ds->outLen += size;
    }

    return size;
}

int
simpleCompress(elzma_file_format format, const unsigned char * inData,
               size_t inLen, unsigned char ** outData,
               size_t * outLen)
{
    int rc;
    elzma_compress_handle hand;

    /* allocate compression handle */
    hand = elzma_compress_alloc();
    assert(hand != NULL);

    rc = elzma_compress_config(hand, ELZMA_LC_DEFAULT,
                               ELZMA_LP_DEFAULT, ELZMA_PB_DEFAULT,
                               5, (1 << 20) /* 1mb */,
                               format, inLen);

    if (rc != ELZMA_E_OK) {
        elzma_compress_free(&hand);
        return rc;
    }    

    /* now run the compression */
    {
        struct dataStream ds;
        ds.inData = inData;
        ds.inLen = inLen;
        ds.outData = NULL;
        ds.outLen = 0;

        rc = elzma_compress_run(hand, inputCallback, (void *) &ds,
                                outputCallback, (void *) &ds,
                                NULL, NULL);
        
        if (rc != ELZMA_E_OK) {
            if (ds.outData != NULL) free(ds.outData);
            elzma_compress_free(&hand);
            return rc;
        }

        *outData = ds.outData;
        *outLen = ds.outLen;
    }

    return rc;
}

int
simpleDecompress(elzma_file_format format, const unsigned char * inData,
                 size_t inLen, unsigned char ** outData,
                 size_t * outLen)
{
    int rc;
    elzma_decompress_handle hand;
    
    hand = elzma_decompress_alloc();
    
    /* now run the compression */
    {
        struct dataStream ds;
        ds.inData = inData;
        ds.inLen = inLen;
        ds.outData = NULL;
        ds.outLen = 0;

        rc = elzma_decompress_run(hand, inputCallback, (void *) &ds,
                                  outputCallback, (void *) &ds, format);
        
        if (rc != ELZMA_E_OK) {
            if (ds.outData != NULL) free(ds.outData);
            elzma_decompress_free(&hand);
            return rc;
        }
        
        *outData = ds.outData;
        *outLen = ds.outLen;
    }

    return rc;
}

#endif
