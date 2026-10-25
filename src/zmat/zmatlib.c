/***************************************************************************//**
**  \mainpage ZMat - A portable C-library and MATLAB/Octave toolbox for inline data compression
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2019-2020
**
**  ZMat provides an easy-to-use interface for stream compression and decompression.
**
**  It can be compiled as a MATLAB/Octave mex function (zipmat.mex/zmat.m) and compresses
**  arrays and strings in MATLAB/Octave. It can also be compiled as a lightweight
**  C-library (libzmat.a/libzmat.so) that can be called in C/C++/FORTRAN etc to
**  provide stream-level compression and decompression.
**
**  Currently, zmat/libzmat supports 6 different compression algorthms, including
**     - zlib and gzip : the most widely used algorithm algorithms for .zip and .gz files
**     - lzma and lzip : high compression ratio LZMA based algorithms for .lzma and .lzip files
**     - lz4 and lz4hc : real-time compression based on LZ4 and LZ4HC algorithms
**     - base64        : base64 encoding and decoding
**
**  Depencency: ZLib library: https://www.zlib.net/
**  author: (C) 1995-2017 Jean-loup Gailly and Mark Adler
**
**  Depencency: LZ4 library: https://lz4.github.io/lz4/
**  author: (C) 2011-2019, Yann Collet,
**
**  Depencency: Original LZMA library
**  author: Igor Pavlov
**
**  Depencency: Eazylzma: https://github.com/lloyd/easylzma
**  author: Lloyd Hilaiel (lloyd)
**
**  Depencency: base64_encode()/base64_decode()
**  \copyright 2005-2011, Jouni Malinen <j@w1.fi>
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    zmatlib.c

@brief   Compression and decompression interfaces: zmat_run, zmat_encode, zmat_decode
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include "zmatlib.h"

#ifndef NO_PTHREAD
    #if defined(_MSC_VER)
        /* MSVC ships no pthreads, and the Windows wheels are built with it, so
         * map the handful of primitives used by the block-parallel codec onto
         * Win32 directly rather than disabling threading on that toolchain or
         * depending on blosc2's vendored shim being compiled in. */
        #include <windows.h>
typedef HANDLE zmat_thread_t;
typedef CRITICAL_SECTION zmat_mutex_t;
        #define zmat_mutex_init(m)    (InitializeCriticalSection(m), 0)
        #define zmat_mutex_lock(m)    EnterCriticalSection(m)
        #define zmat_mutex_unlock(m)  LeaveCriticalSection(m)
        #define zmat_mutex_destroy(m) DeleteCriticalSection(m)
        #define zmat_thread_join(t)   (WaitForSingleObject((t), INFINITE), CloseHandle(t))
    #else
        #include <pthread.h>
typedef pthread_t zmat_thread_t;
typedef pthread_mutex_t zmat_mutex_t;
        #define zmat_mutex_init(m)    pthread_mutex_init((m), NULL)
        #define zmat_mutex_lock(m)    pthread_mutex_lock(m)
        #define zmat_mutex_unlock(m)  pthread_mutex_unlock(m)
        #define zmat_mutex_destroy(m) pthread_mutex_destroy(m)
        #define zmat_thread_join(t)   pthread_join((t), NULL)
    #endif
#endif

#ifndef NO_ZLIB
    #include "zlib.h"
#else
    #include "miniz.h"
    #define GZIP_HEADER_SIZE 10
#endif

#ifndef NO_LZMA
    #include "easylzma/compress.h"
    #include "easylzma/decompress.h"
    #ifdef ZMAT_USE_LZMA_SDK
        #include "easylzma/lzma/XzEnc.h"
        #include "easylzma/lzma/Xz.h"
        #include "easylzma/lzma/Alloc.h"
        #ifndef _WIN32
            #include <pthread.h>
        #endif
    #endif
#endif

#ifndef NO_LZ4
    #include "lz4/lz4.h"
    #include "lz4/lz4hc.h"
#endif

#ifndef NO_BLOSC2
    #include "blosc2.h"
#endif

#ifndef NO_ZSTD
    #include "zstd.h"
    unsigned long long ZSTD_decompressBound(const void* src, size_t srcSize);
#endif

/**
 * @brief Maximum single allocation size (1 GB) to prevent runaway growth
 */
#define ZMAT_MAX_ALLOC  ((size_t)1 << 30)

/**
 * @brief Maximum number of realloc rounds during decompression
 */
#define ZMAT_MAX_DECOMPRESS_ROUNDS 20

/**
 * @brief Minimum decompression output buffer size
 */
#define ZMAT_MIN_OUTBUF 1024

/**
 * @brief Threads used for zlib/gzip when the caller does not ask for a count.
 *
 * Override at build time with -DZMAT_DEFAULT_NTHREAD=N. N=1 makes the threaded
 * path a no-op in practice; a negative nthread from the caller forces the
 * historical single-stream output, see zmat_run_indexed.
 */
#ifndef ZMAT_DEFAULT_NTHREAD
    #define ZMAT_DEFAULT_NTHREAD 8
#endif

#ifdef NO_ZLIB
int miniz_gzip_uncompress(void* in_data, size_t in_len,
                          void** out_data, size_t* out_len);
#endif

#ifndef NO_LZMA
/**
 * @brief Easylzma interface to perform compression
 *
 * @param[in] format: output format (0 for lzip format, 1 for lzma-alone format)
 * @param[in] inData: input stream buffer pointer
 * @param[in] inLen: input stream buffer length
 * @param[in] outData: output stream buffer pointer
 * @param[in] outLen: output stream buffer length
 * @param[in] level: positive number: use default compression level (5);
 *             negative interger: set compression level (-1, less, to -9, more compression)
 * @return return the fine grained lzma error code.
 */

int simpleCompress(elzma_file_format format,
                   const unsigned char* inData,
                   size_t inLen,
                   unsigned char** outData,
                   size_t* outLen,
                   int level,
                   int nthread);

/**
 * @brief Easylzma interface to perform decompression
 *
 * @param[in] format: output format (0 for lzip format, 1 for lzma-alone format)
 * @param[in] inData: input stream buffer pointer
 * @param[in] inLen: input stream buffer length
 * @param[in] outData: output stream buffer pointer
 * @param[in] outLen: output stream buffer length
 * @return return the fine grained lzma error code.
 */

int simpleDecompress(elzma_file_format format,
                     const unsigned char* inData,
                     size_t inLen,
                     unsigned char** outData,
                     size_t* outLen,
                     size_t* consumed);

#ifdef ZMAT_USE_LZMA_SDK
int xzCompress(const unsigned char* inData, size_t inLen,
               unsigned char** outData, size_t* outLen,
               int level, int nthread);
int xzDecompress(const unsigned char* inData, size_t inLen,
                 unsigned char** outData, size_t* outLen);
#ifndef _WIN32
int simpleCompressLzipMT(const unsigned char* inData, size_t inLen,
                         unsigned char** outData, size_t* outLen,
                         int level, int nthread);
#endif
#endif
#endif

/**
 * @brief Coarse grained error messages (encoder-specific detailed error codes are in the status parameter)
 *
 */

const char* zmat_errcode[] = {
    "No error", /*0*/
    "input can not be empty", /*-1*/
    "failed to initialize zlib", /*-2*/
    "zlib error, see info.status for error flag, often a result of mismatch in compression method", /*-3*/
    "easylzma error, see info.status for error flag, often a result of mismatch in compression method",/*-4*/
    "can not allocate output buffer",/*-5*/
    "lz4 error, see info.status for error flag, often a result of mismatch in compression method",/*-6*/
    "unsupported blosc2 codec",/*-7*/
    "blosc2 error, see info.status for error flag, often a result of mismatch in compression method",/*-8*/
    "zstd error, see info.status for error flag, often a result of mismatch in compression method",/*-9*/
    "miniz error, see info.status for error flag, often a result of mismatch in compression method",/*-10*/
    "unsupported method" /*-999*/
};

/**
 * @brief Convert error code to a string error message
 *
 * @param[in] id: zmat error code
 */

const char* zmat_error(int id) {
    if (id >= 0 && id < (int)(sizeof(zmat_errcode) / sizeof(zmat_errcode[0]))) {
        return zmat_errcode[id];
    } else {
        return "zmatlib: unknown error";
    }
}

/**
 * @brief Safely compute initial decompression buffer size (inputsize * multiplier),
 *        with overflow protection and floor/ceiling clamping.
 *
 * @param[in] inputsize: the compressed input size
 * @param[in] multiplier: initial growth multiplier (e.g. 4)
 * @return a safe allocation size
 */

static size_t zmat_initial_outbuf(size_t inputsize, size_t multiplier) {
    size_t outalloc = inputsize * multiplier;

    /* check for overflow */
    if (multiplier != 0 && outalloc / multiplier != inputsize) {
        outalloc = ZMAT_MAX_ALLOC;
    }

    if (outalloc < ZMAT_MIN_OUTBUF) {
        outalloc = ZMAT_MIN_OUTBUF;
    }

    if (outalloc > ZMAT_MAX_ALLOC) {
        outalloc = ZMAT_MAX_ALLOC;
    }

    return outalloc;
}

/**
 * @brief Safely grow a buffer by doubling, with overflow and cap checks.
 *
 * @param[in,out] buf: pointer to the buffer pointer (updated on success)
 * @param[in,out] alloc: pointer to current allocation size (updated on success)
 * @return 0 on success, -5 on failure (*buf is freed and set to NULL)
 */

static int zmat_grow_buf(unsigned char** buf, size_t* alloc) {
    size_t newalloc = (*alloc) * 2;

    /* overflow or exceeds cap */
    if (newalloc <= *alloc) {
        newalloc = (*alloc < ZMAT_MAX_ALLOC) ? ZMAT_MAX_ALLOC : 0;
    }

    if (newalloc > ZMAT_MAX_ALLOC) {
        newalloc = ZMAT_MAX_ALLOC;
    }

    /* if we can't actually grow, fail */
    if (newalloc <= *alloc) {
        free(*buf);
        *buf = NULL;
        return -5;
    }

    unsigned char* tmp = (unsigned char*)realloc(*buf, newalloc);

    if (tmp == NULL) {
        free(*buf);
        *buf = NULL;
        return -5;
    }

    *buf = tmp;
    *alloc = newalloc;
    return 0;
}

/**
 * @brief Shrink buffer to actual used size to free excess memory.
 *
 * @param[in,out] buf: pointer to the buffer pointer
 * @param[in] used: actual bytes used
 */

static void zmat_shrink_buf(unsigned char** buf, size_t used) {
    if (*buf != NULL && used > 0) {
        unsigned char* tmp = (unsigned char*)realloc(*buf, used);

        if (tmp != NULL) {
            *buf = tmp;
        }

        /* if shrink fails, the original pointer is still valid */
    }
}

/**
 * @brief Main interface to perform compression/decompression
 *
 * @param[in] inputsize: input stream buffer length
 * @param[in] inputstr: input stream buffer pointer
 * @param[in, out] outputsize: output stream buffer length
 * @param[in, out] outputbuf: output stream buffer pointer
 * @param[out] ret: encoder/decoder specific detailed error code (if error occurs)
 * @param[in] iscompress: 0: decompression, 1: use default compression level;
 *             negative interger: set compression level (-1, less, to -9, more compression)
 * @return return the coarse grained zmat error code; detailed error code is in ret.
 *
 * On error, *outputbuf is guaranteed to be NULL and *outputsize is 0.
 */

int zmat_run(const size_t inputsize, unsigned char* inputstr, size_t* outputsize, unsigned char** outputbuf, const int zipid, int* ret, const int iscompress) {
    z_stream zs;
    int clevel;
    union cflag {
        int iscompress;
        struct settings {
            char clevel;
            char nthread;
            char shuffle;
            char typesize;
        } param;
    } flags;

    *outputbuf = NULL;
    *outputsize = 0;

    flags.iscompress = iscompress;

    zs.zalloc = Z_NULL;
    zs.zfree = Z_NULL;
    zs.opaque = Z_NULL;

    if (inputsize == 0) {
        return -1;
    }

    clevel = flags.param.clevel;
    unsigned int nthread = (flags.param.nthread <= 0) ? 1 : (unsigned int)flags.param.nthread;
    (void)nthread;

    if (clevel) {
        /**
          * perform compression or encoding
          */
        if (zipid == zmBase64) {
            /**
              * base64 encoding
              */
            *outputbuf = base64_encode((const unsigned char*)inputstr, inputsize, outputsize, clevel);

            if (*outputbuf == NULL) {
                *outputsize = 0;
                return -5;
            }
        } else if (zipid == zmZlib || zipid == zmGzip) {
            /**
              * zlib (.zip) or gzip (.gz) compression
              */
            if (zipid == zmZlib) {
                if (deflateInit(&zs,  (clevel > 0) ? Z_DEFAULT_COMPRESSION : (-clevel)) != Z_OK) {
                    return -2;
                }
            } else {
#ifdef NO_ZLIB
                /* Initialize streaming buffer context (memset clears all fields) */
                memset(&zs, '\0', sizeof(zs));
                zs.next_in  = inputstr;
                zs.avail_in = inputsize;

                if (deflateInit2(&zs, (clevel > 0) ? Z_DEFAULT_COMPRESSION : (-clevel), Z_DEFLATED, -Z_DEFAULT_WINDOW_BITS, 9, Z_DEFAULT_STRATEGY) != Z_OK) {
                    return -2;
                }

#else

                if (deflateInit2(&zs, (clevel > 0) ? Z_DEFAULT_COMPRESSION : (-clevel), Z_DEFLATED, 15 | 16, MAX_MEM_LEVEL, Z_DEFAULT_STRATEGY) != Z_OK) {
                    return -2;
                }

#endif
            }

#ifdef NO_ZLIB

            if (zipid == zmGzip) {

                /*
                 * miniz based gzip compression code was adapted based on the following
                 * https://github.com/atheriel/fluent-bit/blob/8f0002b36601006240d50ea3c86769629d99b1e8/src/flb_gzip.c
                 */
                int flush = Z_NO_FLUSH;
                void* out_buf;
                size_t out_size;
                unsigned char* pb;
                const unsigned char gzip_magic_header [] = {0x1F, 0x8B, 8, 0, 0, 0, 0, 0, 0, 0xFF};

                /* use deflateBound for safe sizing, plus header + footer */
                out_size = deflateBound(&zs, inputsize) + GZIP_HEADER_SIZE + 8;

                out_buf = (unsigned char*)malloc(out_size);

                if (out_buf == NULL) {
                    deflateEnd(&zs);
                    return -5;
                }

                memcpy(out_buf, gzip_magic_header, GZIP_HEADER_SIZE);
                pb = (unsigned char*) out_buf + GZIP_HEADER_SIZE;

                while (1) {
                    zs.next_out  = pb + zs.total_out;
                    zs.avail_out = out_size - GZIP_HEADER_SIZE - 8 - zs.total_out;

                    if (zs.avail_in == 0) {
                        flush = Z_FINISH;
                    }

                    *ret = deflate(&zs, flush);

                    if (*ret == Z_STREAM_END) {
                        break;
                    } else if (*ret != Z_OK) {
                        deflateEnd(&zs);
                        free(out_buf);
                        return -3;
                    }
                }

                if (deflateEnd(&zs) != Z_OK) {
                    free(out_buf);
                    return -3;
                }

                *outputsize = zs.total_out;

                /* Construct the gzip checksum (CRC32 footer) */
                int footer_start = GZIP_HEADER_SIZE + *outputsize;
                pb = (unsigned char*) out_buf + footer_start;

                mz_ulong crc = mz_crc32(MZ_CRC32_INIT, inputstr, inputsize);
                *pb++ = crc & 0xFF;
                *pb++ = (crc >> 8) & 0xFF;
                *pb++ = (crc >> 16) & 0xFF;
                *pb++ = (crc >> 24) & 0xFF;
                *pb++ = inputsize & 0xFF;
                *pb++ = (inputsize >> 8) & 0xFF;
                *pb++ = (inputsize >> 16) & 0xFF;
                *pb++ = (inputsize >> 24) & 0xFF;

                /* update the final output buffer size */
                *outputsize += GZIP_HEADER_SIZE + 8;
                *outputbuf = (unsigned char*)out_buf;

                /* shrink to actual size */
                zmat_shrink_buf(outputbuf, *outputsize);
            } else {
#endif
                size_t bound = deflateBound(&zs, inputsize);
                *outputbuf = (unsigned char*)malloc(bound);

                if (*outputbuf == NULL) {
                    deflateEnd(&zs);
                    return -5;
                }

                zs.avail_in = inputsize;
                zs.next_in = (Bytef*)inputstr;
                zs.avail_out = bound;
                zs.next_out =  (Bytef*)(*outputbuf);

                *ret = deflate(&zs, Z_FINISH);
                *outputsize = zs.total_out;

                if (*ret != Z_STREAM_END && *ret != Z_OK) {
                    deflateEnd(&zs);
                    free(*outputbuf);
                    *outputbuf = NULL;
                    *outputsize = 0;
                    return -3;
                }

                deflateEnd(&zs);

                /* shrink to actual size */
                zmat_shrink_buf(outputbuf, *outputsize);
#ifdef NO_ZLIB
            }

#endif
#ifndef NO_LZMA
        } else if (zipid == zmLzma || zipid == zmLzip) {
            /**
              * lzma (.lzma) or lzip (.lzip) compression
              * for lzip with nthread>1: compress chunks in parallel (Option 3)
              */
#if defined(ZMAT_USE_LZMA_SDK) && !defined(_WIN32)
            if (zipid == zmLzip && nthread > 1) {
                *ret = simpleCompressLzipMT((unsigned char*)inputstr, inputsize,
                                            outputbuf, outputsize, clevel, nthread);
            } else
#endif
            {
                *ret = simpleCompress((elzma_file_format)(zipid - 3), (unsigned char*)inputstr,
                                      inputsize, outputbuf, outputsize, clevel, nthread);
            }

            if (*ret != ELZMA_E_OK) {
                if (*outputbuf) {
                    free(*outputbuf);
                    *outputbuf = NULL;
                }

                *outputsize = 0;
                return -4;
            }

#endif
#if defined(ZMAT_USE_LZMA_SDK) && !defined(NO_LZMA)
        } else if (zipid == zmXz) {
            /**
              * XZ (.xz) compression using LZMA2 with native multi-thread block encoding
              */
            *ret = xzCompress((unsigned char*)inputstr, inputsize, outputbuf, outputsize,
                              clevel, nthread);

            if (*ret != SZ_OK) {
                if (*outputbuf) {
                    free(*outputbuf);
                    *outputbuf = NULL;
                }

                *outputsize = 0;
                return -4;
            }

#endif
#ifndef NO_LZ4
        } else if (zipid == zmLz4 || zipid == zmLz4hc) {
            /**
              * lz4 or lz4hc compression
              */
            *outputsize = LZ4_compressBound(inputsize);

            if (*outputsize == 0) {
                return -6;
            }

            if (!(*outputbuf = (unsigned char*)malloc(*outputsize))) {
                *outputsize = 0;
                return -5;
            }

            if (zipid == zmLz4) {
                *outputsize = LZ4_compress_default((const char*)inputstr, (char*)(*outputbuf), inputsize, *outputsize);
            } else {
                *outputsize = LZ4_compress_HC((const char*)inputstr, (char*)(*outputbuf), inputsize, *outputsize, (clevel > 0) ? 8 : (-clevel));
            }

            *ret = *outputsize;

            if (*outputsize == 0) {
                free(*outputbuf);
                *outputbuf = NULL;
                return -6;
            }

            /* shrink to actual size */
            zmat_shrink_buf(outputbuf, *outputsize);

#endif
#ifndef NO_ZSTD
        } else if (zipid == zmZstd) {
            /**
              * zstd compression — uses MT worker threads when nthread > 1
              * (ZSTD_MULTITHREAD is compiled into the bundled zstd)
              */
            *outputsize = ZSTD_compressBound(inputsize);

            if (!(*outputbuf = (unsigned char*)malloc(*outputsize))) {
                *outputsize = 0;
                return -5;
            }

            {
                ZSTD_CCtx* zctx = ZSTD_createCCtx();

                if (!zctx) {
                    free(*outputbuf);
                    *outputbuf = NULL;
                    *outputsize = 0;
                    return -5;
                }

                ZSTD_CCtx_setParameter(zctx, ZSTD_c_compressionLevel,
                                       (clevel > 0) ? ZSTD_CLEVEL_DEFAULT : (-clevel));
                /* nbWorkers=0 → single-thread (no overhead); >=1 → MT worker threads */
                ZSTD_CCtx_setParameter(zctx, ZSTD_c_nbWorkers, (int)nthread > 1 ? (int)nthread : 0);

                *ret = (int)ZSTD_compress2(zctx, (char*)(*outputbuf), *outputsize,
                                           (const char*)inputstr, inputsize);
                ZSTD_freeCCtx(zctx);
            }

            if (ZSTD_isError((size_t)*ret)) {
                free(*outputbuf);
                *outputbuf = NULL;
                *outputsize = 0;
                return -9;
            }

            *outputsize = *ret;

            /* shrink to actual size */
            zmat_shrink_buf(outputbuf, *outputsize);

#endif
#ifndef NO_BLOSC2
        } else if (zipid >= zmBlosc2Blosclz && zipid <= zmBlosc2Zstd) {
            /**
              * blosc2 meta-compressor (support various filters and compression codecs)
              */
            unsigned int shuffle = 1, typesize = 4;
            const char* codecs[] = {"blosclz", "lz4", "lz4hc", "zlib", "zstd"};
            shuffle = (flags.param.shuffle == 0 || flags.param.shuffle == -1) ? 1 : flags.param.shuffle;
            typesize = (flags.param.typesize == 0 || flags.param.typesize == -1) ? 4 : flags.param.typesize;

            if (blosc1_set_compressor(codecs[zipid - zmBlosc2Blosclz]) == -1) {
                return -7;
            }

            blosc2_set_nthreads(nthread);

            *outputsize = inputsize + BLOSC2_MAX_OVERHEAD;

            if (!(*outputbuf = (unsigned char*)malloc(*outputsize))) {
                *outputsize = 0;
                return -5;
            }

            *ret = blosc1_compress((clevel > 0) ? 5 : (-clevel), shuffle, typesize, inputsize, (const void*)inputstr, (void*)(*outputbuf), *outputsize);

            if (*ret < 0) {
                free(*outputbuf);
                *outputbuf = NULL;
                *outputsize = 0;
                return -8;
            }

            *outputsize = *ret;

            /* shrink to actual size */
            zmat_shrink_buf(outputbuf, *outputsize);

#endif
        } else {
            return -999;
        }
    } else {
        /**
          * perform decompression or decoding
          */
        if (zipid == zmBase64) {
            /**
              * base64 decoding
              */
            *outputbuf = base64_decode((const unsigned char*)inputstr, inputsize, outputsize);

            if (*outputbuf == NULL) {
                *outputsize = 0;
                return -5;
            }
        } else if (zipid == zmZlib || zipid == zmGzip) {
            /**
              * zlib (.zip) or gzip (.gz) decompression
              */
            if (zipid == zmZlib) {
                if (inflateInit(&zs) != Z_OK) {
                    return -2;
                }
            } else {
#ifndef NO_ZLIB

                if (inflateInit2(&zs, 15 | 32) != Z_OK) {
                    return -2;
                }

#endif
            }

#ifdef NO_ZLIB

            if (zipid == zmZlib) {
#endif
                size_t outalloc = zmat_initial_outbuf(inputsize, 4);
                *outputbuf = (unsigned char*)malloc(outalloc);

                if (*outputbuf == NULL) {
                    inflateEnd(&zs);
                    return -5;
                }

                zs.avail_in = inputsize;
                zs.next_in = inputstr;
                zs.avail_out = outalloc;
                zs.next_out =  (Bytef*)(*outputbuf);

                int rounds = 0;

                while (1) {
                    *ret = inflate(&zs, Z_SYNC_FLUSH);

                    if (*ret == Z_STREAM_END) {
                        break;
                    }

                    if (*ret != Z_OK && *ret != Z_BUF_ERROR) {
                        inflateEnd(&zs);
                        free(*outputbuf);
                        *outputbuf = NULL;
                        return -3;
                    }

                    /* no progress with room left to write: the input ran out before
                       the stream ended (truncated data), and inflate would say so
                       forever; Z_OK with room left is progress (miniz returns it
                       after a block), so inflate is called again */
                    if (*ret == Z_BUF_ERROR && zs.avail_out != 0) {
                        inflateEnd(&zs);
                        free(*outputbuf);
                        *outputbuf = NULL;
                        return -3;
                    }

                    /* output buffer full — need to grow */
                    if (zs.avail_out == 0) {
                        rounds++;

                        if (rounds > ZMAT_MAX_DECOMPRESS_ROUNDS) {
                            inflateEnd(&zs);
                            free(*outputbuf);
                            *outputbuf = NULL;
                            return -5;
                        }

                        if (zmat_grow_buf(outputbuf, &outalloc) != 0) {
                            inflateEnd(&zs);
                            /* outputbuf already freed and set to NULL by zmat_grow_buf */
                            return -5;
                        }

                        zs.next_out = (Bytef*)(*outputbuf + zs.total_out);
                        zs.avail_out = outalloc - zs.total_out;
                    }
                }

                *outputsize = zs.total_out;

                if (*ret != Z_STREAM_END && *ret != Z_OK) {
                    inflateEnd(&zs);
                    free(*outputbuf);
                    *outputbuf = NULL;
                    *outputsize = 0;
                    return -3;
                }

                inflateEnd(&zs);

                /* shrink to actual size */
                zmat_shrink_buf(outputbuf, *outputsize);

#ifdef NO_ZLIB
            } else {

                *ret = miniz_gzip_uncompress(inputstr, inputsize, (void**)outputbuf, outputsize);

                if (*ret != 0) {
                    if (*outputbuf) {
                        free(*outputbuf);
                        *outputbuf = NULL;
                    }

                    *outputsize = 0;
                    return -10;
                }
            }

#endif

#ifndef NO_LZMA
        } else if (zipid == zmLzma || zipid == zmLzip) {
            /**
              * lzma (.lzma) or lzip (.lzip) decompression
              * lzip supports multi-member streams: loop over members
              */
            if (zipid == zmLzip) {
#if defined(ZMAT_USE_LZMA_SDK) && !defined(_WIN32)
                /* Attempt v1 backward-scan to locate multi-member boundaries.
                 * simpleCompressLzipMT() produces lzip v1 members: the version
                 * byte is 1 and an 8-byte member_size (little-endian uint64)
                 * is appended after the standard 12-byte footer.  Walking
                 * backward from the end of the stream using member_size gives
                 * exact per-member byte ranges.  Passing exact sizes to
                 * simpleDecompress avoids the consumed-overshoot bug where the
                 * decompressor reads ahead into subsequent members.
                 * The v0 decompressor ignores both the version byte and the
                 * trailing 8 bytes (it reads only the 12-byte footer).
                 */
                size_t n_members = 0;
                size_t* member_starts = NULL;
                size_t* member_sizes  = NULL;

                {
                    size_t end  = inputsize;
                    size_t cap  = 0;
                    int scan_ok = 1;

                    /* minimum v1 member: 6 header + some LZMA + 12 footer + 8 member_size */
                    while (end >= 40 && scan_ok) {
                        /* read 8-byte member_size as little-endian uint64 */
                        const unsigned char* ms_ptr = inputstr + end - 8;
                        unsigned long long ms64 = 0;
                        int k;

                        for (k = 0; k < 8; k++) {
                            ms64 |= ((unsigned long long)ms_ptr[k]) << (8 * k);
                        }

                        if (ms64 < 40 || ms64 > (unsigned long long)end) {
                            scan_ok = 0;
                            break;
                        }

                        size_t ms     = (size_t)ms64;
                        size_t mstart = end - ms;

                        /* verify lzip magic "LZIP" and version == 1 */
                        if (mstart + 5 > inputsize ||
                                inputstr[mstart]     != 'L' ||
                                inputstr[mstart + 1] != 'Z' ||
                                inputstr[mstart + 2] != 'I' ||
                                inputstr[mstart + 3] != 'P' ||
                                inputstr[mstart + 4] != 1) {
                            scan_ok = 0;
                            break;
                        }

                        /* grow member arrays if needed (appending in reverse order) */
                        if (n_members >= cap) {
                            size_t newcap = (cap == 0) ? 8 : cap * 2;
                            size_t* ts = (size_t*)realloc(member_starts, newcap * sizeof(size_t));
                            size_t* tz = (size_t*)realloc(member_sizes,  newcap * sizeof(size_t));

                            if (!ts || !tz) {
                                free(ts ? ts : member_starts);
                                free(tz ? tz : member_sizes);
                                member_starts = NULL;
                                member_sizes  = NULL;
                                n_members     = 0;
                                scan_ok       = 0;
                                break;
                            }

                            member_starts = ts;
                            member_sizes  = tz;
                            cap           = newcap;
                        }

                        member_starts[n_members] = mstart;  /* stored in reverse order */
                        member_sizes [n_members] = ms;
                        n_members++;
                        end = mstart;
                    }

                    /* scan succeeds when we consumed ALL input and found >= 2 members */
                    if (!scan_ok || end != 0 || n_members < 2) {
                        free(member_starts);
                        free(member_sizes);
                        member_starts = NULL;
                        member_sizes  = NULL;
                        n_members     = 0;
                    } else {
                        /* reverse arrays to restore forward order */
                        size_t lo = 0, hi = n_members - 1;

                        while (lo < hi) {
                            size_t ts = member_starts[lo];
                            member_starts[lo] = member_starts[hi];
                            member_starts[hi] = ts;
                            size_t tz = member_sizes[lo];
                            member_sizes[lo]  = member_sizes[hi];
                            member_sizes[hi]  = tz;
                            lo++;
                            hi--;
                        }
                    }
                }

                if (n_members >= 2) {
                    /* multi-member v1 path: decompress each member with exact byte range */
                    size_t total = 0;
                    unsigned char* accum = NULL;
                    size_t mi;
                    *ret = ELZMA_E_OK;

                    for (mi = 0; mi < n_members && *ret == ELZMA_E_OK; mi++) {
                        unsigned char* chunk = NULL;
                        size_t chunk_len = 0;
                        *ret = simpleDecompress(ELZMA_lzip,
                                                inputstr + member_starts[mi],
                                                member_sizes[mi],
                                                &chunk, &chunk_len, NULL);

                        if (*ret == ELZMA_E_OK) {
                            if (chunk_len > ZMAT_MAX_ALLOC - total) {
                                free(chunk);
                                free(accum);
                                free(member_starts);
                                free(member_sizes);
                                *outputsize = 0;
                                return -5;
                            }

                            unsigned char* tmp = (unsigned char*)realloc(accum, total + chunk_len);

                            if (!tmp) {
                                free(chunk);
                                free(accum);
                                free(member_starts);
                                free(member_sizes);
                                *outputsize = 0;
                                return -5;
                            }

                            accum = tmp;
                            memcpy(accum + total, chunk, chunk_len);
                            free(chunk);
                            total += chunk_len;
                        } else {
                            free(chunk);
                        }
                    }

                    free(member_starts);
                    free(member_sizes);
                    *outputbuf  = accum;
                    *outputsize = total;
                } else {
                    /* single-member or non-v1 stream: decompress directly */
                    *ret = simpleDecompress(ELZMA_lzip, (unsigned char*)inputstr,
                                            inputsize, outputbuf, outputsize, NULL);
                }

#else
                /* without ZMAT_USE_LZMA_SDK, always single-member */
                *ret = simpleDecompress(ELZMA_lzip, (unsigned char*)inputstr,
                                        inputsize, outputbuf, outputsize, NULL);
#endif
            } else {
                *ret = simpleDecompress(ELZMA_lzma, (unsigned char*)inputstr,
                                        inputsize, outputbuf, outputsize, NULL);
            }

            if (*ret != ELZMA_E_OK) {
                if (*outputbuf) {
                    free(*outputbuf);
                    *outputbuf = NULL;
                }

                *outputsize = 0;
                return -4;
            }

#endif
#if defined(ZMAT_USE_LZMA_SDK) && !defined(NO_LZMA)
        } else if (zipid == zmXz) {
            /**
              * XZ (.xz) decompression
              */
            *ret = xzDecompress((unsigned char*)inputstr, inputsize, outputbuf, outputsize);

            if (*ret != SZ_OK) {
                if (*outputbuf) {
                    free(*outputbuf);
                    *outputbuf = NULL;
                }

                *outputsize = 0;
                return -4;
            }

#endif
#ifndef NO_LZ4
        } else if (zipid == zmLz4 || zipid == zmLz4hc) {
            /**
              * lz4 or lz4hc decompression
              */
            size_t outalloc = zmat_initial_outbuf(inputsize, 4);
            int rounds = 0;

            if (!(*outputbuf = (unsigned char*)malloc(outalloc))) {
                return -5;
            }

            while ((*ret = LZ4_decompress_safe((const char*)inputstr, (char*)(*outputbuf), inputsize, outalloc)) < 0) {
                rounds++;

                if (rounds > ZMAT_MAX_DECOMPRESS_ROUNDS) {
                    free(*outputbuf);
                    *outputbuf = NULL;
                    *outputsize = 0;
                    return -6;
                }

                if (zmat_grow_buf(outputbuf, &outalloc) != 0) {
                    /* outputbuf already freed and set to NULL by zmat_grow_buf */
                    *outputsize = 0;
                    return -5;
                }
            }

            *outputsize = *ret;

            /* shrink to actual size */
            zmat_shrink_buf(outputbuf, *outputsize);

#endif
#ifndef NO_ZSTD
        } else if (zipid == zmZstd) {
            /**
              * zstd decompression
              */
            {
                unsigned long long zstd_bound = ZSTD_decompressBound(inputstr, inputsize);

                if (zstd_bound == ZSTD_CONTENTSIZE_ERROR || zstd_bound > ZMAT_MAX_ALLOC) {
                    *ret = -9;
                    *outputsize = 0;
                    return -9;
                }

                *outputsize = (size_t)zstd_bound;
            }

            if (!(*outputbuf = (unsigned char*)malloc(*outputsize))) {
                *ret = -5;
                *outputsize = 0;
                return -5;
            }

            *ret = ZSTD_decompress((void*)(*outputbuf), *outputsize, (const void*)inputstr, inputsize);

            if (ZSTD_isError(*ret)) {
                free(*outputbuf);
                *outputbuf = NULL;
                *outputsize = 0;
                return -9;
            }

            *outputsize = *ret;

            /* shrink to actual size */
            zmat_shrink_buf(outputbuf, *outputsize);

#endif
#ifndef NO_BLOSC2
        } else if (zipid >= zmBlosc2Blosclz && zipid <= zmBlosc2Zstd) {
            /**
              * blosc2 meta-compressor (support various filters and compression codecs)
              */
            size_t outalloc = zmat_initial_outbuf(inputsize, 4);
            int rounds = 0;

            if (!(*outputbuf = (unsigned char*)malloc(outalloc))) {
                return -5;
            }

            while ((*ret = blosc1_decompress((const char*)inputstr, (char*)(*outputbuf), outalloc)) <= 0) {
                rounds++;

                if (rounds > ZMAT_MAX_DECOMPRESS_ROUNDS) {
                    free(*outputbuf);
                    *outputbuf = NULL;
                    *outputsize = 0;
                    return -8;
                }

                if (zmat_grow_buf(outputbuf, &outalloc) != 0) {
                    /* outputbuf already freed and set to NULL by zmat_grow_buf */
                    *outputsize = 0;
                    return -5;
                }
            }

            *outputsize = *ret;

            /* shrink to actual size */
            zmat_shrink_buf(outputbuf, *outputsize);

#endif
        } else {
            return -999;
        }
    }

    return 0;
}

/*==============================================================================
 * Block-parallel zlib/gzip
 *
 * DEFLATE is a bit-oriented stream with no framing: a decoder cannot locate
 * block N without inflating everything before it, and back-references reach
 * arbitrarily far into the 32 KB window. Two things make a parallel codec
 * possible anyway:
 *
 *   - Z_FULL_FLUSH ends a block AND resets the window, so each block becomes
 *     independently inflatable while the concatenation stays an ordinary
 *     zlib/gzip stream that any decompressor can read start to finish.
 *   - the block offsets, free to record at compression time, are handed back
 *     to the caller so a later decode can go straight to block N.
 *
 * The block size is fixed rather than derived from the thread count, so the
 * output is a pure function of (data, level, blocksize) and stays byte-
 * identical however many threads produced it -- necessary when the compressed
 * bytes are content-addressed. Threads pull block indices from a shared
 * counter, so a block that compresses slowly cannot starve the others.
 *============================================================================*/

#ifndef NO_PTHREAD

/**
 * @brief Bytes of input per independently-deflated block.
 *
 * Matches jdata.zlibmt's default so the two implementations produce identical
 * streams. Larger blocks compress slightly better and parallelise less.
 */
#define ZMAT_BLOCKSIZE ((size_t)4 << 20)

#ifdef NO_ZLIB
    #define ZMAT_ADLER32(a, b, c) mz_adler32((a), (b), (c))
    #define ZMAT_CRC32(a, b, c)   mz_crc32((a), (b), (c))
    #define ZMAT_ADLER_INIT       MZ_ADLER32_INIT
    #define ZMAT_CRC_INIT         MZ_CRC32_INIT
#else
    #define ZMAT_ADLER32(a, b, c) adler32((a), (b), (c))
    #define ZMAT_CRC32(a, b, c)   crc32((a), (b), (c))
    #define ZMAT_ADLER_INIT       1
    #define ZMAT_CRC_INIT         0
#endif

/** @brief One unit of work: a slice of the input and where its output landed */
typedef struct {
    unsigned char* in;         /**< start of this block's input */
    size_t inlen;              /**< bytes of input in this block */
    unsigned char* out;        /**< malloc'ed compressed segment, or slice of the output */
    size_t outcap;             /**< capacity of out */
    size_t outlen;             /**< bytes actually produced */
    int status;                /**< zlib return code for this block */
} zmat_block;

/** @brief Shared state for the worker pool */
typedef struct zmat_pool_s {
    zmat_block* blocks;
    size_t nblock;
    size_t next;               /**< next block to claim; guarded by lock */
    zmat_mutex_t lock;
    int level;
    int failed;                /**< set by any worker that errors */
    void* (*fn)(void*);        /**< the worker, so the Win32 thunk can dispatch */
} zmat_pool;

#if defined(_MSC_VER)
/** @brief Win32 entry point: adapts the pthread-shaped worker signature */
static DWORD WINAPI zmat_thread_thunk(LPVOID arg) {
    zmat_pool* pool = (zmat_pool*)arg;
    pool->fn(pool);
    return 0;
}
#endif

/** @brief Claim the next unprocessed block, or return 0 when none are left */
static int zmat_pool_next(zmat_pool* pool, size_t* idx) {
    int havework = 0;
    zmat_mutex_lock(&pool->lock);

    if (pool->next < pool->nblock && !pool->failed) {
        *idx = pool->next++;
        havework = 1;
    }

    zmat_mutex_unlock(&pool->lock);
    return havework;
}

/** @brief Mark the whole job failed so the other workers stop early */
static void zmat_pool_fail(zmat_pool* pool) {
    zmat_mutex_lock(&pool->lock);
    pool->failed = 1;
    zmat_mutex_unlock(&pool->lock);
}

/**
 * @brief Deflate one block into a self-contained FULL_FLUSH-terminated segment
 */
static void* zmat_deflate_worker(void* arg) {
    zmat_pool* pool = (zmat_pool*)arg;
    size_t idx;

    while (zmat_pool_next(pool, &idx)) {
        zmat_block* blk = &pool->blocks[idx];
        z_stream zs;
        memset(&zs, 0, sizeof(zs));

        /* raw deflate: this block carries no zlib/gzip wrapper of its own */
        /* memLevel 8 is zlib's default; MAX_MEM_LEVEL would compress slightly
         * better but would no longer match a serial deflate or jdata.zlibmt */
        if (deflateInit2(&zs, pool->level, Z_DEFLATED, -MAX_WBITS,
                         8, Z_DEFAULT_STRATEGY) != Z_OK) {
            blk->status = -2;
            zmat_pool_fail(pool);
            return NULL;
        }

        zs.next_in = (Bytef*)blk->in;
        zs.avail_in = (uInt)blk->inlen;
        zs.next_out = (Bytef*)blk->out;
        zs.avail_out = (uInt)blk->outcap;

        /* FULL_FLUSH, not FINISH: FINISH would set BFINAL and stop every
         * decoder at the end of this block */
        blk->status = deflate(&zs, Z_FULL_FLUSH);

        if (blk->status != Z_OK || zs.avail_in != 0) {
            deflateEnd(&zs);
            zmat_pool_fail(pool);
            return NULL;
        }

        blk->outlen = pool->blocks[idx].outcap - zs.avail_out;
        deflateEnd(&zs);
    }

    return NULL;
}

/**
 * @brief Inflate one block from its recorded offset into its output slice
 */
static void* zmat_inflate_worker(void* arg) {
    zmat_pool* pool = (zmat_pool*)arg;
    size_t idx;

    while (zmat_pool_next(pool, &idx)) {
        zmat_block* blk = &pool->blocks[idx];
        z_stream zs;
        memset(&zs, 0, sizeof(zs));

        if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) {
            blk->status = -2;
            zmat_pool_fail(pool);
            return NULL;
        }

        zs.next_in = (Bytef*)blk->in;
        zs.avail_in = (uInt)blk->inlen;
        zs.next_out = (Bytef*)blk->out;
        zs.avail_out = (uInt)blk->outcap;

        blk->status = inflate(&zs, Z_SYNC_FLUSH);
        blk->outlen = blk->outcap - zs.avail_out;
        inflateEnd(&zs);

        /* the index must describe the stream exactly; a short block means it
         * does not, and guessing would hand back wrong bytes */
        if (blk->outlen != blk->outcap) {
            zmat_pool_fail(pool);
            return NULL;
        }
    }

    return NULL;
}

/** @brief Run a worker pool over the blocks, joining all threads before return */
static int zmat_pool_run(zmat_pool* pool, void* (*fn)(void*), int nthread) {
    zmat_thread_t* tid;
    int i, spawned = 0;
    size_t want = (size_t)nthread;

    if (want > pool->nblock) {
        want = pool->nblock;
    }

    if (want < 1) {
        want = 1;
    }

    if (zmat_mutex_init(&pool->lock) != 0) {
        return -5;
    }

    pool->fn = fn;
    tid = (zmat_thread_t*)calloc(want, sizeof(zmat_thread_t));

    if (!tid) {
        zmat_mutex_destroy(&pool->lock);
        return -5;
    }

    for (i = 0; i < (int)want; i++) {
#if defined(_MSC_VER)
        tid[i] = CreateThread(NULL, 0, zmat_thread_thunk, pool, 0, NULL);

        if (tid[i] == NULL) {
            break;
        }

#else

        if (pthread_create(&tid[i], NULL, fn, pool) != 0) {
            break;
        }

#endif
        spawned++;
    }

    if (spawned == 0) {
        /* no threads available: do the work on this thread so the call still
         * succeeds, just serially */
        fn(pool);
    }

    for (i = 0; i < spawned; i++) {
        zmat_thread_join(tid[i]);
    }

    free(tid);
    zmat_mutex_destroy(&pool->lock);
    return pool->failed ? -3 : 0;
}

/**
 * @brief Compress a buffer to a standard zlib or gzip stream using threads
 *
 * @param[in] inputsize: input length
 * @param[in] inputstr: input buffer
 * @param[out] outputsize: length of the compressed stream
 * @param[out] outputbuf: malloc'ed compressed stream, caller frees
 * @param[in] is_gzip: 0 for a zlib wrapper, 1 for a gzip wrapper
 * @param[in] nthread: worker threads, capped by the block count
 * @param[in] level: zlib compression level 0-9
 * @param[out] offsets: malloc'ed flat index, 2*(nblock+1) entries laid out as
 *             compressed0, uncompressed0, compressed1, ... with a final
 *             sentinel row closing the last block; NULL if not wanted
 * @param[out] noffsets: number of size_t entries written to offsets
 * @param[out] ret: zlib status of the last block
 * @return 0 on success, negative zmat error code otherwise
 */
static int zmat_deflate_blocks(const size_t inputsize, unsigned char* inputstr,
                               size_t* outputsize, unsigned char** outputbuf,
                               int is_gzip, int nthread, int level,
                               size_t** offsets, size_t* noffsets, int* ret) {
    static const unsigned char gzhdr[10] = {0x1F, 0x8B, 8, 0, 0, 0, 0, 0, 0, 0xFF};
    zmat_pool pool;
    size_t nblock, i, pos, uncomp, hdrlen, total;
    unsigned char* out = NULL;
    size_t* idx = NULL;
    z_stream tail;
    unsigned char tailbuf[64];
    size_t taillen = 0;
    int rc;

    memset(&pool, 0, sizeof(pool));
    nblock = (inputsize + ZMAT_BLOCKSIZE - 1) / ZMAT_BLOCKSIZE;

    if (nblock == 0) {
        nblock = 1;
    }

    pool.blocks = (zmat_block*)calloc(nblock, sizeof(zmat_block));

    if (!pool.blocks) {
        return -5;
    }

    /* Every block gets its own bound-sized output buffer. Blocks are equal
     * sized apart from the remainder, which is what pigz does and what the
     * earlier attempt in this repo got wrong -- it capped the block count at
     * the thread count while keeping a fixed block size, so the final block
     * received almost the whole input and no speedup was possible. */
    for (i = 0; i < nblock; i++) {
        size_t start = i * ZMAT_BLOCKSIZE;
        size_t len = inputsize - start;

        if (len > ZMAT_BLOCKSIZE) {
            len = ZMAT_BLOCKSIZE;
        }

        pool.blocks[i].in = inputstr + start;
        pool.blocks[i].inlen = len;
        /* compressBound is a safe per-block bound; +64 covers the flush marker */
        pool.blocks[i].outcap = compressBound((uLong)len) + 64;
        pool.blocks[i].out = (unsigned char*)malloc(pool.blocks[i].outcap);

        if (!pool.blocks[i].out) {
            for (pos = 0; pos < i; pos++) {
                free(pool.blocks[pos].out);
            }

            free(pool.blocks);
            return -5;
        }
    }

    pool.nblock = nblock;
    pool.level = (level > 0) ? Z_DEFAULT_COMPRESSION : (-level);
    rc = zmat_pool_run(&pool, zmat_deflate_worker, nthread);
    *ret = pool.blocks[nblock - 1].status;

    if (rc != 0) {
        for (i = 0; i < nblock; i++) {
            free(pool.blocks[i].out);
        }

        free(pool.blocks);
        return rc;
    }

    /* the terminating empty final block, produced the same way a serial
     * deflate would end the stream */
    memset(&tail, 0, sizeof(tail));

    if (deflateInit2(&tail, pool.level, Z_DEFLATED, -MAX_WBITS,
                     8, Z_DEFAULT_STRATEGY) == Z_OK) {
        tail.next_out = (Bytef*)tailbuf;
        tail.avail_out = (uInt)sizeof(tailbuf);
        deflate(&tail, Z_FINISH);
        taillen = sizeof(tailbuf) - tail.avail_out;
        deflateEnd(&tail);
    }

    hdrlen = is_gzip ? sizeof(gzhdr) : 2;
    total = hdrlen + taillen + (is_gzip ? 8 : 4);

    for (i = 0; i < nblock; i++) {
        total += pool.blocks[i].outlen;
    }

    out = (unsigned char*)malloc(total);
    idx = (size_t*)malloc(2 * (nblock + 1) * sizeof(size_t));

    if (!out || !idx) {
        free(out);
        free(idx);

        for (i = 0; i < nblock; i++) {
            free(pool.blocks[i].out);
        }

        free(pool.blocks);
        return -5;
    }

    if (is_gzip) {
        memcpy(out, gzhdr, sizeof(gzhdr));
    } else {
        /* CMF/FLG for a 32 KB window at this level, with the check bits set so
         * (CMF<<8|FLG) is a multiple of 31 */
        unsigned int cmf = 0x78, flg;
        /* resolve Z_DEFAULT_COMPRESSION before deriving FLEVEL, or level-6 data
         * ends up advertising level 0 in its header */
        int actual = (pool.level == Z_DEFAULT_COMPRESSION) ? 6 : pool.level;
        unsigned int flevel = (actual <= 1) ? 0 : (actual <= 5 ? 1 : (actual == 6 ? 2 : 3));
        flg = flevel << 6;
        flg += 31 - ((cmf << 8 | flg) % 31);
        out[0] = (unsigned char)cmf;
        out[1] = (unsigned char)flg;
    }

    pos = hdrlen;
    uncomp = 0;

    for (i = 0; i < nblock; i++) {
        idx[2 * i] = pos;
        idx[2 * i + 1] = uncomp;
        memcpy(out + pos, pool.blocks[i].out, pool.blocks[i].outlen);
        pos += pool.blocks[i].outlen;
        uncomp += pool.blocks[i].inlen;
        free(pool.blocks[i].out);
    }

    idx[2 * nblock] = pos;         /* sentinel closes the last block */
    idx[2 * nblock + 1] = uncomp;
    free(pool.blocks);

    memcpy(out + pos, tailbuf, taillen);
    pos += taillen;

    if (is_gzip) {
        unsigned long crc = ZMAT_CRC32(ZMAT_CRC_INIT, inputstr, (unsigned int)inputsize);
        out[pos++] = (unsigned char)(crc & 0xFF);
        out[pos++] = (unsigned char)((crc >> 8) & 0xFF);
        out[pos++] = (unsigned char)((crc >> 16) & 0xFF);
        out[pos++] = (unsigned char)((crc >> 24) & 0xFF);
        out[pos++] = (unsigned char)(inputsize & 0xFF);
        out[pos++] = (unsigned char)((inputsize >> 8) & 0xFF);
        out[pos++] = (unsigned char)((inputsize >> 16) & 0xFF);
        out[pos++] = (unsigned char)((inputsize >> 24) & 0xFF);
    } else {
        unsigned long ad = ZMAT_ADLER32(ZMAT_ADLER_INIT, inputstr, (unsigned int)inputsize);
        out[pos++] = (unsigned char)((ad >> 24) & 0xFF);
        out[pos++] = (unsigned char)((ad >> 16) & 0xFF);
        out[pos++] = (unsigned char)((ad >> 8) & 0xFF);
        out[pos++] = (unsigned char)(ad & 0xFF);
    }

    *outputbuf = out;
    *outputsize = pos;

    if (offsets && noffsets) {
        *offsets = idx;
        *noffsets = 2 * (nblock + 1);
    } else {
        free(idx);
    }

    return 0;
}

/**
 * @brief Decompress a block-indexed stream using threads
 *
 * @param[in] offsets: the index produced by zmat_deflate_blocks
 * @param[in] noffsets: number of entries in offsets
 * @return 0 on success; -3 if the index does not describe the stream, in which
 *         case the caller should fall back to a serial inflate
 */
static int zmat_inflate_blocks(const size_t inputsize, unsigned char* inputstr,
                               size_t* outputsize, unsigned char** outputbuf,
                               int nthread, const size_t* offsets, size_t noffsets,
                               int* ret) {
    zmat_pool pool;
    size_t nblock, i, outlen;
    unsigned char* out;
    int rc;

    if (!offsets || noffsets < 4 || (noffsets & 1)) {
        return -3;
    }

    nblock = noffsets / 2 - 1;
    outlen = offsets[2 * nblock + 1];

    /* sanity: the index must stay inside the buffers it describes, and the
     * first block must start the output -- otherwise its leading bytes would
     * be returned uninitialized */
    if (offsets[1] != 0 || offsets[2 * nblock] > inputsize || outlen == 0 || outlen > ZMAT_MAX_ALLOC) {
        return -3;
    }

    for (i = 0; i < nblock; i++) {
        if (offsets[2 * i] >= offsets[2 * i + 2] || offsets[2 * i + 1] >= offsets[2 * i + 3]) {
            return -3;
        }
    }

    memset(&pool, 0, sizeof(pool));
    pool.blocks = (zmat_block*)calloc(nblock, sizeof(zmat_block));

    if (!pool.blocks) {
        return -5;
    }

    out = (unsigned char*)malloc(outlen);

    if (!out) {
        free(pool.blocks);
        return -5;
    }

    /* every block writes into a disjoint slice of one allocation, so the
     * workers never touch the same bytes and no locking is needed */
    for (i = 0; i < nblock; i++) {
        pool.blocks[i].in = inputstr + offsets[2 * i];
        pool.blocks[i].inlen = offsets[2 * i + 2] - offsets[2 * i];
        pool.blocks[i].out = out + offsets[2 * i + 1];
        pool.blocks[i].outcap = offsets[2 * i + 3] - offsets[2 * i + 1];
    }

    pool.nblock = nblock;
    rc = zmat_pool_run(&pool, zmat_inflate_worker, nthread);
    *ret = pool.blocks[nblock - 1].status;
    free(pool.blocks);

    if (rc != 0) {
        free(out);
        return -3;
    }

    *outputbuf = out;
    *outputsize = outlen;
    return 0;
}

#endif /* NO_PTHREAD */

/**
 * @brief Compression/decompression with an optional block index
 *
 * Behaves exactly like zmat_run except that zlib and gzip gain a threaded
 * path. On compression, when nthread > 1, the input is deflated in fixed-size
 * blocks concurrently and, if offsets is non-NULL, the block index is returned
 * alongside. On decompression, when an index is supplied and nthread > 1, the
 * blocks are inflated concurrently; if the index does not describe the stream
 * the call falls back to the ordinary serial path rather than failing, since
 * the payload is a valid stream either way.
 *
 * Every other codec, and zlib/gzip with a single thread, is handed straight to
 * zmat_run, so the bytes are unchanged.
 *
 * @param[out] offsets: on compression, malloc'ed index (caller frees) or NULL;
 *             on decompression, an index previously produced here, or NULL
 * @param[in,out] noffsets: entry count for offsets
 */
int zmat_run_indexed(const size_t inputsize, unsigned char* inputstr, size_t* outputsize,
                     unsigned char** outputbuf, const int zipid, int* ret, const int iscompress,
                     size_t** offsets, size_t* noffsets) {
    union cflag {
        int iscompress;
        struct settings {
            char clevel;
            char nthread;
            char shuffle;
            char typesize;
        } param;
    } flags;
    int nthread, reqthread;

    flags.iscompress = iscompress;
    /* nthread == 0 means the caller never asked about threading; 1 or more is
     * an explicit request. The distinction matters: the blocked construction
     * changes the compressed bytes, so it must be opt-in, and it must then be
     * used for every explicit thread count including 1. Gating on nthread > 1
     * instead would make nthread=1 and nthread=2 disagree byte for byte, which
     * breaks content addressing of the compressed payload; gating on anything
     * always-true would change the output of existing callers. */
    reqthread = (int)flags.param.nthread;

    if (reqthread == 0) {
        /* caller said nothing: use the build-time default */
        reqthread = ZMAT_DEFAULT_NTHREAD;
    }

    /* A negative count is an explicit request for the historical single-stream
     * output. It is the only way back to those exact bytes now that the
     * threaded path is the default, and it is needed to reproduce data written
     * by earlier releases. */
    nthread = (reqthread <= 0) ? 1 : reqthread;

#ifndef NO_PTHREAD

    if ((zipid == zmZlib || zipid == zmGzip) && reqthread >= 1 &&
            inputsize > ZMAT_BLOCKSIZE) {
        if (flags.param.clevel) {
            int rc;
            *outputbuf = NULL;
            *outputsize = 0;
            rc = zmat_deflate_blocks(inputsize, inputstr, outputsize, outputbuf,
                                     (zipid == zmGzip), nthread, flags.param.clevel,
                                     offsets, noffsets, ret);

            if (rc == 0) {
                return 0;
            }

            /* threaded deflate failed; the serial path still works */
        } else if (offsets && noffsets && *offsets && *noffsets >= 4) {
            /* an index is only a speed hint; it is verified in
             * zmat_inflate_blocks and discarded if it does not fit */
            int rc;
            unsigned char* buf = NULL;
            size_t len = 0;
            rc = zmat_inflate_blocks(inputsize, inputstr, &len, &buf, nthread,
                                     *offsets, *noffsets, ret);

            if (rc == 0) {
                *outputbuf = buf;
                *outputsize = len;
                return 0;
            }

            /* index unusable -- fall through to the serial inflate below */
        }
    }

#else
    (void)nthread;
    (void)reqthread;
#endif

    if (offsets && noffsets && flags.param.clevel) {
        *offsets = NULL;
        *noffsets = 0;
    }

    return zmat_run(inputsize, inputstr, outputsize, outputbuf, zipid, ret, iscompress);
}

/**
 * @brief Simplified interface to perform compression (use default compression level)
 *
 * @param[in] inputsize: input stream buffer length
 * @param[in] inputstr: input stream buffer pointer
 * @param[in] outputsize: output stream buffer length
 * @param[in] outputbuf: output stream buffer pointer
 * @param[in] ret: encoder/decoder specific detailed error code (if error occurs)
 * @return return the coarse grained zmat error code; detailed error code is in ret.
 */

int zmat_encode(const size_t inputsize, unsigned char* inputstr, size_t* outputsize, unsigned char** outputbuf, const int zipid, int* ret) {
    /* Routed through the indexed entry point with a null index so that zlib and
     * gzip pick up ZMAT_DEFAULT_NTHREAD. This is the path the Fortran binding
     * and plain C callers take, and it is otherwise identical to zmat_run. */
    size_t* offsets = NULL;
    size_t noffsets = 0;
    return zmat_run_indexed(inputsize, inputstr, outputsize, outputbuf, zipid, ret, 1,
                            &offsets, &noffsets);
}

/**
 * @brief Simplified interface to perform decompression
 *
 * @param[in] inputsize: input stream buffer length
 * @param[in] inputstr: input stream buffer pointer
 * @param[in] outputsize: output stream buffer length
 * @param[in] outputbuf: output stream buffer pointer
 * @param[in] ret: encoder/decoder specific detailed error code (if error occurs)
 * @return return the coarse grained zmat error code; detailed error code is in ret.
 */

int zmat_decode(const size_t inputsize, unsigned char* inputstr, size_t* outputsize, unsigned char** outputbuf, const int zipid, int* ret) {
    size_t* offsets = NULL;
    size_t noffsets = 0;
    return zmat_run_indexed(inputsize, inputstr, outputsize, outputbuf, zipid, ret, 0,
                            &offsets, &noffsets);
}

/**
 * @brief Look up a string in a string list and return the index
 *
 * @param[in] origkey: string to be looked up
 * @param[out] table: the dictionary where the string is searched
 * @return if found, return the index of the string in the dictionary, otherwise -1.
 */

int zmat_keylookup(char* origkey, const char* table[]) {
    int i = 0;
    char* key = (char*)malloc(strlen(origkey) + 1);

    if (key == NULL) {
        return -1;
    }

    memcpy(key, origkey, strlen(origkey) + 1);

    while (key[i]) {
        key[i] = tolower(key[i]);
        i++;
    }

    i = 0;

    while (table[i] && table[i][0] != '\0') {
        if (strcmp(key, table[i]) == 0) {
            free(key);
            return i;
        }

        i++;
    }

    free(key);
    return -1;
}

/**
 * @brief Free the output buffer to facilitate use in fortran
 *
 * @param[in,out] outputbuf: the outputbuf buffer's initial address to be freed
 */

void zmat_free(unsigned char** outputbuf) {
    if (*outputbuf) {
        free(*outputbuf);
    }

    *outputbuf = NULL;
}

/*
 * @brief Base64 encoding/decoding (RFC1341)
 * @author Copyright (c) 2005-2011, Jouni Malinen <j@w1.fi>
 *
 * This software may be distributed under the terms of the BSD license.
 * See README for more details.
 */

static const unsigned char base64_table[65] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/**
 * @brief base64_encode - Base64 encode
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

unsigned char* base64_encode(const unsigned char* src, size_t len,
                             size_t* out_len, int mode) {
    unsigned char* out, *pos;
    const unsigned char* end, *in;
    size_t olen;
    size_t line_len;

    olen = len * 4 / 3 + 4; /* 3-byte blocks to 4-byte */

    if (olen < len) {
        return NULL;    /* integer overflow */
    }

    olen += olen / 72; /* line feeds */
    olen++; /* nul termination */

    if (olen < len) {
        return NULL;    /* integer overflow */
    }

    out = (unsigned char*)malloc(olen);

    if (out == NULL) {
        return NULL;
    }

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

        if (mode > 1 && line_len >= 72) {
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

    if (mode > 2 && line_len) {
        *pos++ = '\n';
    }

    *pos = '\0';

    if (out_len) {
        *out_len = pos - out;
    }

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

unsigned char* base64_decode(const unsigned char* src, size_t len,
                             size_t* out_len) {
    unsigned char dtable[256], *out, *pos, block[4], tmp;
    size_t i, count, olen;
    int pad = 0;

    memset(dtable, 0x80, 256);

    for (i = 0; i < sizeof(base64_table) - 1; i++) {
        dtable[base64_table[i]] = (unsigned char) i;
    }

    dtable[(unsigned char)'='] = 0;

    count = 0;

    for (i = 0; i < len; i++) {
        if (dtable[src[i]] != 0x80) {
            count++;
        }
    }

    if (count == 0 || count % 4) {
        return NULL;
    }

    olen = count / 4 * 3;
    pos = out = (unsigned char*)malloc(olen);

    if (out == NULL) {
        return NULL;
    }

    count = 0;

    for (i = 0; i < len; i++) {
        tmp = dtable[src[i]];

        if (tmp == 0x80) {
            continue;
        }

        if (src[i] == '=') {
            pad++;
        }

        block[count] = tmp;
        count++;

        if (count == 4) {
            *pos++ = (block[0] << 2) | (block[1] >> 4);
            *pos++ = (block[1] << 4) | (block[2] >> 2);
            *pos++ = (block[2] << 6) | block[3];
            count = 0;

            if (pad) {
                if (pad == 1) {
                    pos--;
                } else if (pad == 2) {
                    pos -= 2;
                } else {
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

/**
 * @brief Easylzma compression interface
 */

struct dataStream {
    const unsigned char* inData;
    size_t inLen;
    size_t consumed;    /* tracks how many input bytes were read (for multi-member lzip) */

    unsigned char* outData;
    size_t outLen;
};

/**
 * @brief Easylzma input callback function
 */

static int
inputCallback(void* ctx, void* buf, size_t* size) {
    size_t rd = 0;
    struct dataStream* ds = (struct dataStream*) ctx;
    assert(ds != NULL);

    rd = (ds->inLen < *size) ? ds->inLen : *size;

    if (rd > 0) {
        memcpy(buf, (void*) ds->inData, rd);
        ds->inData += rd;
        ds->inLen -= rd;
        ds->consumed += rd;
    }

    *size = rd;

    return 0;
}

/**
 * @brief Easylzma output callback function
 */

static size_t
outputCallback(void* ctx, const void* buf, size_t size) {
    struct dataStream* ds = (struct dataStream*) ctx;
    assert(ds != NULL);

    if (size > 0) {
        unsigned char* tmp = (unsigned char*)realloc(ds->outData, ds->outLen + size);

        if (tmp == NULL) {
            /* realloc failed — preserve existing data pointer for caller to free */
            return 0;
        }

        ds->outData = tmp;
        memcpy((void*) (ds->outData + ds->outLen), buf, size);
        ds->outLen += size;
    }

    return size;
}

/**
 * @brief Easylzma interface to perform compression
 *
 * @param[in] format: output format (0 for lzip format, 1 for lzma-alone format)
 * @param[in] inData: input stream buffer pointer
 * @param[in] inLen: input stream buffer length
 * @param[in] outData: output stream buffer pointer
 * @param[in] outLen: output stream buffer length
 * @param[in] level: positive number: use default compression level (5);
 *             negative interger: set compression level (-1, less, to -9, more compression)
 * @return return the fine grained lzma error code.
 */

int
simpleCompress(elzma_file_format format, const unsigned char* inData,
               size_t inLen, unsigned char** outData,
               size_t* outLen, int level, int nthread) {
    int rc;
    elzma_compress_handle hand;

    /* allocate compression handle */
    hand = elzma_compress_alloc();

    if (hand == NULL) {
        return ELZMA_E_COMPRESS_ERROR;
    }

    /* set thread count (clamped to 1–2 by SDK; effective only with COMPRESS_MF_MT) */
    elzma_compress_set_numthreads(hand, nthread);

    rc = elzma_compress_config(hand, ELZMA_LC_DEFAULT,
                               ELZMA_LP_DEFAULT, ELZMA_PB_DEFAULT,
                               ((level > 0) ? 5 : -level), (1 << 20) /* 1mb */,
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
        ds.consumed = 0;
        ds.outData = NULL;
        ds.outLen = 0;

        rc = elzma_compress_run(hand, inputCallback, (void*) &ds,
                                outputCallback, (void*) &ds,
                                NULL, NULL);

        if (rc != ELZMA_E_OK) {
            if (ds.outData != NULL) {
                free(ds.outData);
            }

            elzma_compress_free(&hand);
            return rc;
        }

        *outData = ds.outData;
        *outLen = ds.outLen;
    }

    elzma_compress_free(&hand);

    return rc;
}


/**
 * @brief Easylzma interface to perform decompression
 *
 * @param[in] format: output format (0 for lzip format, 1 for lzma-alone format)
 * @param[in] inData: input stream buffer pointer
 * @param[in] inLen: input stream buffer length
 * @param[in] outData: output stream buffer pointer
 * @param[in] outLen: output stream buffer length
 * @return return the fine grained lzma error code.
 */

int
simpleDecompress(elzma_file_format format, const unsigned char* inData,
                 size_t inLen, unsigned char** outData,
                 size_t* outLen, size_t* consumed) {
    int rc;
    elzma_decompress_handle hand;

    hand = elzma_decompress_alloc();

    if (hand == NULL) {
        return ELZMA_E_DECOMPRESS_ERROR;
    }

    /* now run the decompression */
    {
        struct dataStream ds;
        ds.inData = inData;
        ds.inLen = inLen;
        ds.consumed = 0;
        ds.outData = NULL;
        ds.outLen = 0;

        rc = elzma_decompress_run(hand, inputCallback, (void*) &ds,
                                  outputCallback, (void*) &ds, format);

        if (consumed != NULL) {
            *consumed = ds.consumed;
        }

        if (rc != ELZMA_E_OK) {
            if (ds.outData != NULL) {
                free(ds.outData);
            }

            elzma_decompress_free(&hand);
            return rc;
        }

        elzma_decompress_free(&hand);

        *outData = ds.outData;
        *outLen = ds.outLen;
    }

    return rc;
}

#ifdef ZMAT_USE_LZMA_SDK

/* -----------------------------------------------------------------------
 * XZ stream wrappers — ISeqOutStream / ISeqInStream must have the
 * vtable function pointer as their FIRST field (C-style interface).
 * ----------------------------------------------------------------------- */

typedef struct {
    ISeqOutStream   vt;  /* first field — SDK casts the pointer to ISeqOutStreamPtr */
    struct dataStream* ds;
} ZmatXzOutStream;

typedef struct {
    ISeqInStream    vt;  /* first field */
    struct dataStream* ds;
} ZmatXzInStream;

static size_t zmat_xz_write(ISeqOutStreamPtr p, const void* buf, size_t size) {
    ZmatXzOutStream* s = (ZmatXzOutStream*)(void*)p;
    return outputCallback(s->ds, buf, size);
}

static SRes zmat_xz_read(ISeqInStreamPtr p, void* buf, size_t* size) {
    ZmatXzInStream* s = (ZmatXzInStream*)(void*)p;
    return (SRes)inputCallback(s->ds, buf, size);
}

/**
 * @brief XZ compression using LZMA2 with native multi-thread block encoding
 */
int
xzCompress(const unsigned char* inData, size_t inLen,
           unsigned char** outData, size_t* outLen,
           int level, int nthread) {
    CXzProps props;
    CXzEncHandle enc;
    SRes rc;
    struct dataStream ds;
    ZmatXzOutStream outStream;
    ZmatXzInStream  inStream;

    XzProps_Init(&props);
    props.lzma2Props.lzmaProps.level      = (level > 0) ? 5 : (-level);
    props.lzma2Props.lzmaProps.numThreads = 1;              /* no match-finder MT: all parallelism at block level */
    props.lzma2Props.numBlockThreads_Max  = (int)nthread;   /* block-level MT */
    /* explicit block size: split input evenly across threads, 1 MB minimum.
     * avoids the default 128 MB auto block size (dictSize*4 at level 5)
     * which leaves small inputs as a single solid block with zero parallelism. */
    {
        size_t blk = (inLen + (size_t)nthread - 1) / (size_t)nthread;

        if (blk < (1u << 20)) {
            blk = (1u << 20);
        }

        props.lzma2Props.blockSize = (UInt64)blk;
    }
    props.checkId = XZ_CHECK_CRC32;

    ds.inData   = inData;
    ds.inLen    = inLen;
    ds.consumed = 0;
    ds.outData  = NULL;
    ds.outLen   = 0;

    outStream.vt.Write = zmat_xz_write;
    outStream.ds       = &ds;
    inStream.vt.Read   = zmat_xz_read;
    inStream.ds        = &ds;

    enc = XzEnc_Create(&g_Alloc, &g_BigAlloc);

    if (!enc) {
        return SZ_ERROR_MEM;
    }

    rc = XzEnc_SetProps(enc, &props);

    if (rc == SZ_OK) {
        XzEnc_SetDataSize(enc, (UInt64)inLen);
        rc = XzEnc_Encode(enc, &outStream.vt, &inStream.vt, NULL);
    }

    XzEnc_Destroy(enc);

    if (rc != SZ_OK) {
        free(ds.outData);
        return rc;
    }

    *outData = ds.outData;
    *outLen  = ds.outLen;
    return SZ_OK;
}

/**
 * @brief XZ decompression using XzUnpacker streaming decoder
 */
int
xzDecompress(const unsigned char* inData, size_t inLen,
             unsigned char** outData, size_t* outLen) {
    CXzUnpacker xz;
    ECoderStatus status = CODER_STATUS_NOT_SPECIFIED;
    SRes rc = SZ_OK;
    unsigned char* accum = NULL;
    size_t total = 0;
    const Byte* src = (const Byte*)inData;
    SizeT srcLeft = (SizeT)inLen;

#define XZ_DECODE_BUF 65536
    Byte outBuf[XZ_DECODE_BUF];

    XzUnpacker_Construct(&xz, &g_Alloc);
    XzUnpacker_Init(&xz);

    while (srcLeft > 0 || status == CODER_STATUS_NOT_FINISHED) {
        SizeT destLen = XZ_DECODE_BUF;
        SizeT srcUsed = srcLeft;

        rc = XzUnpacker_Code(&xz, outBuf, &destLen, src, &srcUsed,
                             (srcLeft == 0) ? 1 : 0,
                             CODER_FINISH_ANY, &status);

        if (rc != SZ_OK) {
            break;
        }

        if (destLen > 0) {
            if (destLen > ZMAT_MAX_ALLOC - total) {
                rc = SZ_ERROR_MEM;
                break;
            }

            unsigned char* tmp = (unsigned char*)realloc(accum, total + destLen);

            if (!tmp) {
                rc = SZ_ERROR_MEM;
                break;
            }

            accum = tmp;
            memcpy(accum + total, outBuf, destLen);
            total += destLen;
        }

        src     += srcUsed;
        srcLeft -= srcUsed;

        if (status == CODER_STATUS_FINISHED_WITH_MARK) {
            break;
        }

        if (srcUsed == 0 && destLen == 0) {
            break;    /* no progress — avoid infinite loop */
        }
    }

    if (rc == SZ_OK && !XzUnpacker_IsStreamWasFinished(&xz)) {
        rc = SZ_ERROR_DATA;
    }

    XzUnpacker_Free(&xz);

    if (rc != SZ_OK) {
        free(accum);
        return rc;
    }

    *outData = accum;
    *outLen  = total;
    return SZ_OK;
}

/* -----------------------------------------------------------------------
 * Option 3: parallel lzip — compress chunks independently, concatenate
 * ----------------------------------------------------------------------- */

#ifndef _WIN32

typedef struct {
    const unsigned char* in;
    size_t               inLen;
    unsigned char*       out;
    size_t               outLen;
    int                  level;
    int                  rc;
} LzipChunk;

static void* lzip_compress_chunk(void* arg) {
    LzipChunk* c = (LzipChunk*)arg;
    c->rc = simpleCompress(ELZMA_lzip, c->in, c->inLen,
                           &c->out, &c->outLen, c->level, 1);

    /* Upgrade v0 → lzip v1: patch version byte (byte[4]) and append an
     * 8-byte member_size field after the standard 12-byte footer.
     * The v0 decompressor ignores the version byte and reads only 12 footer
     * bytes, so v1 members are backward-compatible with the existing code.
     * Backward-scanning the member_size fields lets the decompressor locate
     * each member boundary precisely, fixing the consumed-overshoot bug. */
    if (c->rc == ELZMA_E_OK && c->out != NULL && c->outLen > 18) {
        size_t v1_size = c->outLen + 8;
        unsigned char* tmp = (unsigned char*)realloc(c->out, v1_size);

        if (tmp) {
            int k;
            c->out    = tmp;
            c->out[4] = 1;    /* version 1 */

            /* write member_size as little-endian uint64 */
            for (k = 0; k < 8; k++) {
                c->out[c->outLen + k] = (unsigned char)(v1_size >> (8 * k));
            }

            c->outLen = v1_size;
        }

        /* if realloc fails, leave as v0 — single-member fallback still works */
    }

    return NULL;
}

int
simpleCompressLzipMT(const unsigned char* inData, size_t inLen,
                     unsigned char** outData, size_t* outLen,
                     int level, int nthread) {
    if (nthread <= 1 || inLen == 0) {
        return simpleCompress(ELZMA_lzip, inData, inLen,
                              outData, outLen, level, 1);
    }

    size_t chunk = (inLen + (size_t)nthread - 1) / (size_t)nthread;

    /* Require at least 1 KB per chunk to make parallelism worthwhile.
     * For small inputs, fall back to single-thread (produces standard v0). */
    if (chunk < 1024) {
        return simpleCompress(ELZMA_lzip, inData, inLen,
                              outData, outLen, level, 1);
    }

    LzipChunk*  chunks  = (LzipChunk*)calloc((size_t)nthread, sizeof(LzipChunk));
    pthread_t*  threads = (pthread_t*)calloc((size_t)nthread, sizeof(pthread_t));

    if (!chunks || !threads) {
        free(chunks);
        free(threads);
        return ELZMA_E_COMPRESS_ERROR;
    }

    int i;

    for (i = 0; i < nthread; i++) {
        chunks[i].in    = inData + (size_t)i * chunk;
        chunks[i].inLen = ((size_t)i == (size_t)nthread - 1)
                          ? (inLen - (size_t)i * chunk) : chunk;
        chunks[i].level = level;

        if (pthread_create(&threads[i], NULL, lzip_compress_chunk, &chunks[i]) != 0) {
            /* join already-started threads and clean up */
            int j;

            for (j = 0; j < i; j++) {
                pthread_join(threads[j], NULL);
                free(chunks[j].out);
            }

            free(chunks);
            free(threads);
            return ELZMA_E_COMPRESS_ERROR;
        }
    }

    size_t total = 0;
    int rc = ELZMA_E_OK;

    for (i = 0; i < nthread; i++) {
        pthread_join(threads[i], NULL);

        if (chunks[i].rc != ELZMA_E_OK) {
            rc = chunks[i].rc;
        }

        total += chunks[i].outLen;
    }

    if (rc != ELZMA_E_OK) {
        for (i = 0; i < nthread; i++) {
            free(chunks[i].out);
        }

        free(chunks);
        free(threads);
        return rc;
    }

    unsigned char* buf = (unsigned char*)malloc(total);

    if (!buf) {
        for (i = 0; i < nthread; i++) {
            free(chunks[i].out);
        }

        free(chunks);
        free(threads);
        return ELZMA_E_COMPRESS_ERROR;
    }

    size_t pos = 0;

    for (i = 0; i < nthread; i++) {
        memcpy(buf + pos, chunks[i].out, chunks[i].outLen);
        free(chunks[i].out);
        pos += chunks[i].outLen;
    }

    free(chunks);
    free(threads);
    *outData = buf;
    *outLen  = total;
    return ELZMA_E_OK;
}

#endif  /* !_WIN32 */
#endif  /* ZMAT_USE_LZMA_SDK */

#endif

#ifdef NO_ZLIB
/*
 * miniz based gzip compression code was adapted based on the following
 * https://github.com/atheriel/fluent-bit/blob/8f0002b36601006240d50ea3c86769629d99b1e8/src/flb_gzip.c
 */

/*  Fluent Bit
 *  ==========
 *  Copyright (C) 2019-2020 The Fluent Bit Authors
 *  Copyright (C) 2015-2018 Treasure Data Inc.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

typedef enum {
    FTEXT    = 1,
    FHCRC    = 2,
    FEXTRA   = 4,
    FNAME    = 8,
    FCOMMENT = 16
} miniz_tinf_gzip_flag;

static unsigned int read_le16(const unsigned char* p) {
    return ((unsigned int) p[0]) | ((unsigned int) p[1] << 8);
}

static unsigned int read_le32(const unsigned char* p) {
    return ((unsigned int) p[0])
           | ((unsigned int) p[1] << 8)
           | ((unsigned int) p[2] << 16)
           | ((unsigned int) p[3] << 24);
}

/* Uncompress (inflate) GZip data */
int miniz_gzip_uncompress(void* in_data, size_t in_len,
                          void** out_data, size_t* out_len) {
    int status;
    unsigned char* p;
    void* out_buf;
    size_t out_size = 0;
    void* zip_data;
    size_t zip_len;
    unsigned char flg;
    unsigned int xlen, hcrc;
    unsigned int dlen, crc;
    mz_ulong crc_out;
    mz_stream stream;
    const unsigned char* start;

    *out_data = NULL;
    *out_len = 0;

    /* Minimal length: header + crc32 */
    if (in_len < 18) {
        return -1;
    }

    /* Magic bytes */
    p = (unsigned char*)in_data;

    if (p[0] != 0x1F || p[1] != 0x8B) {
        return -2;
    }

    if (p[2] != 8) {
        return -3;
    }

    /* Flag byte */
    flg = p[3];

    /* Reserved bits */
    if (flg & 0xE0) {
        return -4;
    }

    /* Skip base header of 10 bytes */
    start = p + GZIP_HEADER_SIZE;

    /* Skip extra data if present */
    if (flg & FEXTRA) {
        if (start + 2 > p + in_len) {
            return -5;
        }

        xlen = read_le16(start);

        if (xlen > in_len - (start - p) - 2) {
            return -5;
        }

        start += xlen + 2;
    }

    /* Skip file name if present */
    if (flg & FNAME) {
        do {
            if (start - p >= (ptrdiff_t)in_len) {
                return -6;
            }
        } while (*start++);
    }

    /* Skip file comment if present */
    if (flg & FCOMMENT) {
        do {
            if (start - p >= (ptrdiff_t)in_len) {
                return -6;
            }
        } while (*start++);
    }

    /* Check header crc if present */
    if (flg & FHCRC) {
        if (start - p > (ptrdiff_t)(in_len - 2)) {
            return -7;
        }

        hcrc = read_le16(start);
        crc = mz_crc32(MZ_CRC32_INIT, p, start - p) & 0x0000FFFF;

        if (hcrc != crc) {
            return -8;
        }

        start += 2;
    }

    /* Get decompressed length */
    dlen = read_le32(&p[in_len - 4]);

    /* Get CRC32 checksum of original data */
    crc = read_le32(&p[in_len - 8]);

    /* Decompress data */
    if ((p + in_len) - start < 8) {
        return -9;
    }

    /* Allocate outgoing buffer */
    out_buf = malloc(dlen);

    if (!out_buf) {
        return -10;
    }

    out_size = dlen;

    /* Map zip content */
    zip_data = (unsigned char*) start;
    zip_len = (p + in_len) - start - 8;

    memset(&stream, 0, sizeof(stream));
    stream.next_in = (unsigned char*)zip_data;
    stream.avail_in = zip_len;
    stream.next_out = (unsigned char*)out_buf;
    stream.avail_out = out_size;

    status = mz_inflateInit2(&stream, -Z_DEFAULT_WINDOW_BITS);

    if (status != MZ_OK) {
        free(out_buf);
        return -11;
    }

    status = mz_inflate(&stream, MZ_FINISH);

    if (status != MZ_STREAM_END) {
        mz_inflateEnd(&stream);
        free(out_buf);
        return -12;
    }

    if (stream.total_out != dlen) {
        mz_inflateEnd(&stream);
        free(out_buf);
        return -13;
    }

    /* terminate the stream, it's not longer required */
    mz_inflateEnd(&stream);

    /* Validate message CRC vs inflated data CRC */
    crc_out = mz_crc32(MZ_CRC32_INIT, (unsigned char*)out_buf, dlen);

    if (crc_out != crc) {
        free(out_buf);
        return -14;
    }

    /* set the uncompressed data */
    *out_len = dlen;
    *out_data = out_buf;

    return 0;
}
#endif