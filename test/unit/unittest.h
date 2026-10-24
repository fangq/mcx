/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2026
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    unittest.h

@brief   Check macros and counters shared by the MCX unit test files
*******************************************************************************/

#ifndef _MCX_UNITTEST_H
#define _MCX_UNITTEST_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/wait.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int ntest, nfail;  /**< total and failed checks, defined in the file with main() */

#ifdef __cplusplus
}
#endif

#define EXPECT_NEAR(a, b, tol, msg) do {                                         \
        double va_ = (double)(a), vb_ = (double)(b);                             \
        ntest++;                                                                 \
        if (!(fabs(va_ - vb_) <= (tol))) {                                       \
            nfail++;                                                             \
            printf("FAIL %s:%d %s: %.9g != %.9g (tol %g)\n", __FILE__, __LINE__, \
                   msg, va_, vb_, (double)(tol));                                \
        }                                                                        \
    } while (0)

#define EXPECT_EQ(a, b, msg) EXPECT_NEAR(a, b, 0, msg)

#define EXPECT_TRUE(cond, msg) do {                                              \
        ntest++;                                                                 \
        if (!(cond)) {                                                           \
            nfail++;                                                             \
            printf("FAIL %s:%d %s\n", __FILE__, __LINE__, msg);                  \
        }                                                                        \
    } while (0)

/**
 * check that a statement reports an error and exits, such as MCX_ERROR does: the
 * statement runs in a forked child with its output discarded; the check passes
 * if the child exits with a non-zero code. The child always leaves through exit()
 * so that the coverage counters of an instrumented build are saved.
 */
#define EXPECT_EXIT(stmt, msg) do {                                              \
        int status_ = 0;                                                         \
        pid_t pid_;                                                              \
        ntest++;                                                                 \
        fflush(stdout);                                                          \
        pid_ = fork();                                                           \
        if (pid_ == 0) {                                                         \
            if (freopen("/dev/null", "w", stdout) == NULL ||                     \
                    freopen("/dev/null", "w", stderr) == NULL) {                 \
                exit(0);                                                         \
            }                                                                    \
            stmt;                                                                \
            exit(0);                                                             \
        }                                                                        \
        waitpid(pid_, &status_, 0);                                              \
        if (!(WIFEXITED(status_) && WEXITSTATUS(status_) != 0)) {                \
            nfail++;                                                             \
            printf("FAIL %s:%d %s: no error was reported\n", __FILE__, __LINE__, msg); \
        }                                                                        \
    } while (0)

/** check that a statement finishes the program normally, such as printing the help, with exit(0) */
#define EXPECT_EXIT_OK(stmt, msg) do {                                           \
        int status_ = 0;                                                         \
        pid_t pid_;                                                              \
        ntest++;                                                                 \
        fflush(stdout);                                                          \
        pid_ = fork();                                                           \
        if (pid_ == 0) {                                                         \
            if (freopen("/dev/null", "w", stdout) == NULL ||                     \
                    freopen("/dev/null", "w", stderr) == NULL) {                 \
                exit(3);                                                         \
            }                                                                    \
            stmt;                                                                \
            exit(0);                                                             \
        }                                                                        \
        waitpid(pid_, &status_, 0);                                              \
        if (!(WIFEXITED(status_) && WEXITSTATUS(status_) == 0)) {                \
            nfail++;                                                             \
            printf("FAIL %s:%d %s: the program did not finish normally\n", __FILE__, __LINE__, msg); \
        }                                                                        \
    } while (0)

/** relative tolerance check, |a-b| <= rtol*|b| */
#define EXPECT_REL(a, b, rtol, msg) EXPECT_NEAR(a, b, (rtol) * fabs((double)(b)), msg)

#endif
