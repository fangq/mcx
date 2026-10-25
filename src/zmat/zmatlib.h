/***************************************************************************//**
**  \mainpage ZMat - A portable C-library and MATLAB/Octave toolbox for inline data compression
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2019-2023
**
**  ZMat provides an easy-to-use interface for stream compression and decompression.
**
**  It can be compiled as a MATLAB/Octave mex function (zipmat.mex/zmat.m) and compresses
**  arrays and strings in MATLAB/Octave. It can also be compiled as a lightweight
**  C-library (libzmat.a/libzmat.so) that can be called in C/C++/FORTRAN etc to
**  provide stream-level compression and decompression.
**
**  Currently, zmat/libzmat supports a list of different compression algorthms, including
**     - zlib and gzip : the most widely used algorithm algorithms for .zip and .gz files
**     - lzma and lzip : high compression ratio LZMA based algorithms for .lzma and .lzip files
**     - lz4 and lz4hc : real-time compression based on LZ4 and LZ4HC algorithms
**     - zstd : ZStandard compression algorithm
**     - blosc2{blosclz,lz4,lz4hc,zlib,zstd}: blosc2 multi-threading meta-compressor/decompressors
**     - base64        : base64 encoding and decoding
**
**  ZMat is part of the NeuroJSON project (https://neurojson.org)
**  More information can be found at https://github.com/NeuroJSON/zmat
**
**  Depencency: ZLib library: https://www.zlib.net/
**  \copyright (c) 1995-2017 Jean-loup Gailly and Mark Adler
**
**  Depencency: LZ4 library: https://lz4.github.io/lz4/
**  \copyright (c) 2011-2019, Yann Collet,
**
**  Depencency: Original LZMA library
**  \copyright Igor Pavlov
**
**  Depencency: Eazylzma: https://github.com/lloyd/easylzma
**  \copyright Lloyd Hilaiel (lloyd)
**
**  Depencency: base64_encode()/base64_decode()
**  \copyright (c) 2005-2011, Jouni Malinen <j@w1.fi>
**
**  Depencency: C-blosc2
**  \copyright (c) 2019-present The Blosc Development Team <blosc@blosc.org>
**  \copyright (c) 2009-2018 Francesc Alted <francesc@blosc.org>
**
**  Depencency: ZStandard
**  \copyright (c) Meta Platforms, Inc. and affiliates.
**
**  Depencency: miniz
**  \copyright (c) 2013-2014 RAD Game Tools and Valve Software
**  \copyright (c) 2010-2014 Rich Geldreich and Tenacious Software LLC
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    zmatlib.h

@brief   zmat library header file
*******************************************************************************/

#ifndef ZMAT_LIB_H
#define ZMAT_LIB_H

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Compression/encoding methods
 *
 * 0: zlib
 * 1: gzip
 * 2: base64
 * 3: lzip
 * 4: lzma
 * 5: lz4
 * 6: lz4hc
 * 7: zstd
 * 8: blosc2blosclz
 * 9: blosc2lz4
 * 10: blosc2lz4hc
 * 11: blosc2zlib
 * 12: blosc2zstd
 * -1: unknown
 */

typedef enum TZipMethod {zmZlib, zmGzip, zmBase64, zmLzip, zmLzma, zmLz4, zmLz4hc, zmZstd, zmBlosc2Blosclz, zmBlosc2Lz4, zmBlosc2Lz4hc, zmBlosc2Zlib, zmBlosc2Zstd, zmXz, zmUnknown = -1} TZipMethod;

/**
 * @brief advanced ZMat parameters needed for blosc2 metacompressor
 */

typedef union TZMatFlags {
    int iscompress;      /**< combined flag used to pass on to zmat_run */
    struct settings {    /**< unpacked flags */
        char clevel;     /**< compression level, 0: decompression, 1: use default level; negative: set compression level (-1 to -19) */
        char nthread;    /**< number of compression/decompression threads */
        char shuffle;    /**< byte shuffle length */
        char typesize;   /**< for ND-array, the byte-size for each array element */
    } param;
} TZMatFlags;

/**
 * @brief Main interface to perform compression/decompression
 *
 * @param[in] inputsize: input stream buffer length
 * @param[in] inputstr: input stream buffer pointer
 * @param[in] outputsize: output stream buffer length
 * @param[in] outputbuf: output stream buffer pointer
 * @param[in] ret: encoder/decoder specific detailed error code (if error occurs)
 * @param[in] iscompress: 0: decompression, 1: use default compression level;
 *            negative interger: set compression level (-1, less, to -9, more compression).
 * @return return the coarse grained zmat error code; detailed error code is in ret.
 */

int zmat_run(const size_t inputsize, unsigned char* inputstr, size_t* outputsize, unsigned char** outputbuf, const int zipid, int* ret, const int iscompress);

/**
 * @brief Compression/decompression with an optional block index
 *
 * Identical to zmat_run except that zlib and gzip gain a threaded path.
 *
 * There are two constructions, selected by nthread, and they do not produce the
 * same bytes:
 *
 *   nthread <= -1  the historical single deflate stream, unchanged from
 *                  zmat_run. Use this to reproduce data written by earlier
 *                  releases.
 *   nthread == 0   whatever ZMAT_DEFAULT_NTHREAD selects (blocked, by default).
 *   nthread >= 1   the blocked construction described below.
 *
 * In the blocked construction the input is deflated in fixed-size blocks
 * concurrently, each terminated with Z_FULL_FLUSH so it is independently
 * inflatable, and the blocks are concatenated into one ordinary zlib or gzip
 * stream that any decompressor can read start to finish. Within this
 * construction the output is a pure function of (data, level, blocksize): it is
 * byte-identical for nthread of 1, 8 or 32, so a payload can be content
 * addressed without pinning the thread count of whoever wrote it.
 *
 * It is not, and cannot be, identical to the serial construction. Resetting the
 * LZ77 window at every block boundary is what makes a block independently
 * inflatable, and it costs compression ratio: negligible on incompressible data
 * (+0.000% measured on 32 MB of random uint16) but as much as +21% on data whose
 * redundancy spans the block size, such as a long monotonic counter. Callers who
 * need the smallest possible output and do not need threading or random access
 * should pass a negative nthread.
 *
 * When offsets is non-NULL it receives a malloc'ed flat index of 2*(nblock+1)
 * entries, laid out as compressed0, uncompressed0, compressed1, ... with a final
 * sentinel row closing the last block; the caller frees it.
 *
 * On decompression with nthread > 1 and an index supplied in offsets, the
 * blocks are inflated concurrently into disjoint slices of one allocation. An
 * index that does not describe the stream is rejected and the call falls back
 * to the serial path, so a stale index costs time but never correctness.
 *
 * Every other codec, and zlib/gzip with a single thread, is forwarded
 * unchanged to zmat_run.
 *
 * @param[in] inputsize: input stream buffer length
 * @param[in] inputstr: input stream buffer pointer
 * @param[out] outputsize: output stream buffer length
 * @param[out] outputbuf: output stream buffer pointer
 * @param[in] zipid: compression method id
 * @param[out] ret: encoder/decoder specific detailed error code
 * @param[in] iscompress: packed clevel/nthread/shuffle/typesize flags
 * @param[in,out] offsets: block index, see above; NULL to ignore
 * @param[in,out] noffsets: number of size_t entries in offsets
 * @return the coarse grained zmat error code
 */

int zmat_run_indexed(const size_t inputsize, unsigned char* inputstr, size_t* outputsize,
                     unsigned char** outputbuf, const int zipid, int* ret, const int iscompress,
                     size_t** offsets, size_t* noffsets);

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

int zmat_encode(const size_t inputsize, unsigned char* inputstr, size_t* outputsize, unsigned char** outputbuf, const int zipid, int* ret);

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

int zmat_decode(const size_t inputsize, unsigned char* inputstr, size_t* outputsize, unsigned char** outputbuf, const int zipid, int* ret);

/**
 * @brief Free the output buffer to facilitate use in fortran
 *
 * @param[in,out] outputbuf: the outputbuf buffer's initial address to be freed
 */

void zmat_free(unsigned char** outputbuf);

/**
 * @brief Look up a string in a string list and return the index
 *
 * @param[in] origkey: string to be looked up
 * @param[out] table: the dictionary where the string is searched
 * @return if found, return the index of the string in the dictionary, otherwise -1.
 */

int  zmat_keylookup(char* origkey, const char* table[]);

/**
 * @brief Convert error code to a string error message
 *
 * @param[in] id: zmat error code
 */

const char* zmat_error(int id);

/**
 * @brief base64_encode - Base64 encode
 * @src: Data to be encoded
 * @len: Length of the data to be encoded
 * @out_len: Pointer to output length variable, or %NULL if not used
 * @mode: 0 or 1, newline every 72 char and at end; 2: no new line at end, 3: no newline
 * Returns: Allocated buffer of out_len bytes of encoded data,
 * or %NULL on failure
 *
 * Caller is responsible for freeing the returned buffer. Returned buffer is
 * nul terminated to make it easier to use as a C string. The nul terminator is
 * not included in out_len.
 */

unsigned char* base64_encode(const unsigned char* src, size_t len,
                             size_t* out_len, int mode);

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
                             size_t* out_len);

#ifdef __cplusplus
}
#endif

#endif
