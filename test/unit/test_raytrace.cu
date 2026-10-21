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
\file    test_raytrace.cu

@brief   Host-side unit tests of the MCX ray-tracing helpers

The pure helpers in mcx_core.cu (hitgrid, transmit, reflectcoeff, rotatevector)
and the xorshift128+ RNG are marked MCX_HD (__host__ __device__), so this file
includes mcx_core.cu and calls them directly on the CPU, comparing against
analytic results. No GPU is needed to run the resulting binary; build it with
"make unittest" under src/.
*******************************************************************************/

#include "../../src/mcx_core.cu"

#include <cstdio>
#include <cstdint>

static int ntest = 0, nfail = 0;

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

/** the textbook unpolarized Fresnel reflectance, computed in double precision */
static double fresnel(double n1, double n2, double cosi) {
    double sint2 = (n1 / n2) * (n1 / n2) * (1.0 - cosi * cosi);

    if (sint2 >= 1.0) {
        return 1.0;
    }

    double cost = sqrt(1.0 - sint2);
    double rs = (n1 * cosi - n2 * cost) / (n1 * cosi + n2 * cost);
    double rp = (n1 * cost - n2 * cosi) / (n1 * cost + n2 * cosi);
    return 0.5 * (rs * rs + rp * rp);
}

static void set_ray(float3* v, float* rv, float x, float y, float z) {
    float len = sqrtf(x * x + y * y + z * z);
    v->x = x / len;
    v->y = y / len;
    v->z = z / len;
    rv[0] = 1.f / v->x;
    rv[1] = 1.f / v->y;
    rv[2] = 1.f / v->z;
}

static void test_hitgrid(void) {
    float3 p0, v;
    float rv[3];
    short id[4];

    /* axis-aligned ray from the voxel center hits the +x face after 0.5 */
    p0 = make_float3(0.5f, 0.5f, 0.5f);
    id[0] = id[1] = id[2] = 0;
    set_ray(&v, rv, 1.f, 0.f, 0.f);
    EXPECT_NEAR(hitgrid(&p0, &v, rv, id), 0.5f, 1e-6, "hitgrid +x distance");
    EXPECT_EQ(id[3], 0, "hitgrid +x face");

    /* negative direction, ray hits the -y face of voxel (3,7,2) */
    p0 = make_float3(3.4f, 7.25f, 2.9f);
    id[0] = 3;
    id[1] = 7;
    id[2] = 2;
    set_ray(&v, rv, 0.f, -1.f, 0.f);
    EXPECT_NEAR(hitgrid(&p0, &v, rv, id), 0.25f, 1e-6, "hitgrid -y distance");
    EXPECT_EQ(id[3], 1, "hitgrid -y face");

    /* oblique ray, the z face is the nearest: (1-0.9)/vz */
    p0 = make_float3(0.2f, 0.5f, 0.9f);
    id[0] = id[1] = id[2] = 0;
    set_ray(&v, rv, 1.f, 1.f, 1.f);
    EXPECT_NEAR(hitgrid(&p0, &v, rv, id), 0.1f * sqrtf(3.f), 1e-6, "hitgrid diagonal distance");
    EXPECT_EQ(id[3], 2, "hitgrid diagonal face");

    /* the end point must lie on the face it reports */
    p0 = make_float3(5.3f, 1.7f, 9.05f);
    id[0] = 5;
    id[1] = 1;
    id[2] = 9;
    set_ray(&v, rv, -0.3f, 0.8f, -0.52f);
    float d = hitgrid(&p0, &v, rv, id);
    float3 p1 = make_float3(p0.x + d * v.x, p0.y + d * v.y, p0.z + d * v.z);
    float face = (id[3] == 0) ? p1.x : (id[3] == 1 ? p1.y : p1.z);
    EXPECT_NEAR(face, roundf(face), 1e-5, "hitgrid end point on a voxel face");
    EXPECT_NEAR(p1.x, fminf(fmaxf(p1.x, 5.f), 6.f), 1e-5, "hitgrid end point x inside voxel");
    EXPECT_NEAR(p1.y, fminf(fmaxf(p1.y, 1.f), 2.f), 1e-5, "hitgrid end point y inside voxel");
    EXPECT_NEAR(p1.z, fminf(fmaxf(p1.z, 9.f), 10.f), 1e-5, "hitgrid end point z inside voxel");
}

static void test_reflectcoeff(void) {
    MCXdir v;
    const float n[][2] = {{1.f, 1.5f}, {1.5f, 1.f}, {1.37f, 1.f}, {1.f, 1.37f}, {1.33f, 1.33f}};

    for (int i = 0; i < (int)(sizeof(n) / sizeof(n[0])); i++) {
        for (int deg = 0; deg < 90; deg += 5) {
            double ci = cos(deg * M_PI / 180.0), si = sin(deg * M_PI / 180.0);
            v.x = (float)si;
            v.y = 0.f;
            v.z = (float)ci;
            v.nscat = 0.f;
            EXPECT_NEAR(reflectcoeff(&v, n[i][0], n[i][1], 2), fresnel(n[i][0], n[i][1], ci), 2e-6, "Fresnel reflectance");
        }
    }

    v.x = 0.f;
    v.y = 0.f;
    v.z = 1.f;
    EXPECT_NEAR(reflectcoeff(&v, 1.f, 1.5f, 2), 0.04f, 1e-7, "normal incidence ((n1-n2)/(n1+n2))^2");
}

static void test_transmit(void) {
    MCXdir v;
    const float n1 = 1.f, n2 = 1.4f;

    for (int deg = 5; deg < 90; deg += 10) {
        double si = sin(deg * M_PI / 180.0);
        v.x = (float)(si * 0.6);
        v.y = (float)(si * 0.8);
        v.z = (float)cos(deg * M_PI / 180.0);
        v.nscat = 0.f;
        transmit(&v, n1, n2, 2);
        double st = sqrt(v.x * v.x + v.y * v.y);
        EXPECT_NEAR(v.x * v.x + v.y * v.y + v.z * v.z, 1.0, 1e-6, "transmitted direction is unit length");
        EXPECT_NEAR(st, si * n1 / n2, 1e-6, "Snell's law n1*sin(i)=n2*sin(t)");
        EXPECT_NEAR(v.x / st, 0.6, 1e-5, "transmission keeps the plane of incidence");
        EXPECT_NEAR(v.z > 0.f, 1, 0, "transmission keeps the propagation direction");
    }
}

static void test_rotatevector(void) {
    const float dirs[][3] = {{0.f, 0.f, 1.f}, {0.f, 0.f, -1.f}, {0.3f, -0.4f, 0.866f}, {1.f, 0.f, 0.f}, {-0.5f, 0.5f, -0.70710678f}};

    for (int i = 0; i < (int)(sizeof(dirs) / sizeof(dirs[0])); i++) {
        for (int k = 0; k < 8; k++) {
            float theta = 0.1f + 0.37f * k, phi = 0.9f * k;
            MCXdir v, v0;
            float len = sqrtf(dirs[i][0] * dirs[i][0] + dirs[i][1] * dirs[i][1] + dirs[i][2] * dirs[i][2]);
            v.x = dirs[i][0] / len;
            v.y = dirs[i][1] / len;
            v.z = dirs[i][2] / len;
            v.nscat = 3.f;
            v0 = v;
            rotatevector(&v, sinf(theta), cosf(theta), sinf(phi), cosf(phi));
            EXPECT_NEAR(v.x * v.x + v.y * v.y + v.z * v.z, 1.0, 2e-6, "rotated direction is unit length");
            EXPECT_NEAR(v.x * v0.x + v.y * v0.y + v.z * v0.z, cosf(theta), 2e-6, "rotation preserves the scattering angle");
            EXPECT_EQ(v.nscat, 3.f, "rotation keeps the scattering count");
        }
    }
}

/** reference xorshift128+ (Vigna 2014, shifts 23/18/5) on plain 64-bit integers */
static uint64_t ref_s[2];

static float ref_xorshift128p_nextf(void) {
    uint64_t s1 = ref_s[0];
    const uint64_t s0 = ref_s[1];
    ref_s[0] = s0;
    s1 ^= s1 << 23;
    ref_s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    uint32_t lo = (uint32_t)((ref_s[1] + s0) & 0xFFFFFFFFu);
    union {
        uint32_t u;
        float f;
    } r;
    r.u = 0x3F800000U | (lo >> 9);
    return r.f - 1.0f;
}

static void test_rng(void) {
    uint seed[4] = {0x12345678u, 0x9ABCDEF0u, 0x0F1E2D3Cu, 0x4B5A6978u};
    RandType t[RAND_BUF_LEN];

    xorshift128p_seed(seed, t);
    ref_s[0] = ((uint64_t)seed[0] << 32) | seed[1];
    ref_s[1] = ((uint64_t)seed[2] << 32) | seed[3];

    double sum = 0.0;
    const int n = 100000;

    for (int i = 0; i < n; i++) {
        float r = rand_uniform01(t);

        if (i < 1000) {
            EXPECT_EQ(r, ref_xorshift128p_nextf(), "xorshift128+ sequence matches the reference");
        }

        if (r < 0.f || r >= 1.f) {
            EXPECT_NEAR(r, 0.5, 0.5, "rand_uniform01 in [0,1)");
        }

        sum += r;
    }

    EXPECT_NEAR(sum / n, 0.5, 5e-3, "rand_uniform01 mean");

    /* scattering length is exponential with unit mean */
    sum = 0.0;

    for (int i = 0; i < n; i++) {
        sum += rand_next_scatlen(t);
    }

    EXPECT_NEAR(sum / n, 1.0, 2e-2, "rand_next_scatlen mean");
}

int main(void) {
    test_hitgrid();
    test_reflectcoeff();
    test_transmit();
    test_rotatevector();
    test_rng();

    printf("%d checks, %d failures\n", ntest, nfail);
    return nfail > 0;
}
