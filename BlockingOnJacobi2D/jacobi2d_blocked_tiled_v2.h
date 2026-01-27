#ifndef Jacobi2DBlockedV2_STYLE_H
#define Jacobi2DBlockedV2_STYLE_H

#include <omp.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>

/*
 MPI-style block decomposition for Jacobi 2D
 - Blocks: NB_I x NB_J  (NB_I = N / BM, NB_J = N / BN)
 - Each block stores a BM x BN inner tile contiguously inside A_blk / B_blk:
     layout: [bi][bj][ii][jj] flattened -> ((bi*NB_J + bj) * BM + ii) * BN + jj
 - Halos:
     top/bottom: [NB_I][NB_J][BN]
     left/right: [NB_I][NB_J][BM]
 - Global (undecomposed) indices: i = 0..N+1, j = 0..N+1
   interior indices:    i=1..N, j=1..N
*/

template<int BM, int BN>
class Jacobi2DBlockedV2 {
public:
    int N;                // interior size
    int NB_I, NB_J;       // number of blocks in i (rows), j (cols)
    int num_blocks;

    // Interiors stored as contiguous block arrays
    // A_blk and B_blk: [NB_I][NB_J][BM][BN] flattened
    double *A_blk; // current
    double *B_blk; // next

    // Halo arrays (compact)
    // top/bottom: [NB_I][NB_J][BN]
    // left/right: [NB_I][NB_J][BM]
    double *A_top, *A_bottom, *A_left, *A_right;
    double *B_top, *B_bottom, *B_left, *B_right;

    Jacobi2DBlockedV2(int size) : N(size) {
        if (N % BM != 0 || N % BN != 0) {
            fprintf(stderr, "N must be multiple of BM and BN\n");
            std::exit(1);
        }
        NB_I = N / BM;
        NB_J = N / BN;
        num_blocks = NB_I * NB_J;

        // Interiors: exactly N*N entries (checks)
        A_blk = new double[num_blocks * BM * BN];
        B_blk = new double[num_blocks * BM * BN];

        // Halos sizes
        A_top    = new double[num_blocks * BN];
        A_bottom = new double[num_blocks * BN];
        A_left   = new double[num_blocks * BM];
        A_right  = new double[num_blocks * BM];

        B_top    = new double[num_blocks * BN];
        B_bottom = new double[num_blocks * BN];
        B_left   = new double[num_blocks * BM];
        B_right  = new double[num_blocks * BM];

        // Zero-init (optional, safe)
        std::memset(A_blk, 0, sizeof(double) * num_blocks * BM * BN);
        std::memset(B_blk, 0, sizeof(double) * num_blocks * BM * BN);
        std::memset(A_top, 0, sizeof(double) * num_blocks * BN);
        std::memset(A_bottom,0,sizeof(double) * num_blocks * BN);
        std::memset(A_left,0,sizeof(double) * num_blocks * BM);
        std::memset(A_right,0,sizeof(double) * num_blocks * BM);
        std::memset(B_top, 0, sizeof(double) * num_blocks * BN);
        std::memset(B_bottom,0,sizeof(double) * num_blocks * BN);
        std::memset(B_left,0,sizeof(double) * num_blocks * BM);
        std::memset(B_right,0,sizeof(double) * num_blocks * BM);
    }

    ~Jacobi2DBlockedV2() {
        delete[] A_blk; delete[] B_blk;
        delete[] A_top; delete[] A_bottom; delete[] A_left; delete[] A_right;
        delete[] B_top; delete[] B_bottom; delete[] B_left; delete[] B_right;
    }

    /* ----------------- Index helpers ----------------- */

    inline int block_id(int bi, int bj) const {
        return bi * NB_J + bj;
    }

    // interior flatten index for BLK arrays
    inline int blk_index(int bi, int bj, int ii, int jj) const {
        return ((bi * NB_J + bj) * BM + ii) * BN + jj;
    }

    // top/bottom indexing (per-block BN entries)
    inline int top_index(int bi, int bj, int lj) const {
        return (block_id(bi, bj) * BN) + lj;
    }

    // left/right indexing (per-block BM entries)
    inline int left_index(int bi, int bj, int li) const {
        return (block_id(bi, bj) * BM) + li;
    }

    // access macros
    inline constexpr double &BLK(double *arr, int bi, int bj, int ii, int jj) const {
        return arr[blk_index(bi,bj,ii,jj)];
    }
    inline constexpr double &HT(double *top_arr, int bi, int bj, int lj) const {
        return top_arr[top_index(bi,bj,lj)];
    }
    inline constexpr double &HL(double *left_arr, int bi, int bj, int li) const {
        return left_arr[left_index(bi,bj,li)];
    }

    /* ----------------- Initialization -----------------
       Must reproduce undecomposed:
       for i=0..N+1, j=0..N+1:
           A[i*(N+2)+j] = (double)(i*(j+2))/N
           B[i*(N+2)+j] = (double)(i*(j+3))/N
    -------------------------------------------------- */
    void initialize() {
        // Interiors and halos filled from global coordinates
        #pragma omp parallel for collapse(2)
        for (int bi = 0; bi < NB_I; bi++) {
            for (int bj = 0; bj < NB_J; bj++) {
                // interior BM x BN
                #pragma omp unroll
                for (int ii = 0; ii < BM; ii++) {
                    #pragma omp simd
                    for (int jj = 0; jj < BN; jj++) {
                        int gi = bi * BM + ii + 1;   // global i in [1..N]
                        int gj = bj * BN + jj + 1;   // global j in [1..N]
                        BLK(A_blk, bi, bj, ii, jj) = (double)(gi * (gj + 2)) / N;
                        BLK(B_blk, bi, bj, ii, jj) = (double)(gi * (gj + 3)) / N;
                    }
                }

                // TOP halo: global row gi = bi*BM + 0  (could be 0..N)
                #pragma omp simd
                for (int lj = 0; lj < BN; lj++) {
                    int gi = bi * BM + 0;
                    int gj = bj * BN + lj + 1;
                    HT(A_top, bi, bj, lj)    = (double)(gi * (gj + 2)) / N;
                    HT(B_top, bi, bj, lj)    = (double)(gi * (gj + 3)) / N;
                }

                // BOTTOM halo: gi = bi*BM + BM + 1
                #pragma omp simd
                for (int lj = 0; lj < BN; lj++) {
                    int gi = bi * BM + BM + 1;
                    int gj = bj * BN + lj + 1;
                    HT(A_bottom, bi, bj, lj) = (double)(gi * (gj + 2)) / N;
                    HT(B_bottom, bi, bj, lj) = (double)(gi * (gj + 3)) / N;
                }

                // LEFT halo: gj = bj*BN + 0
                #pragma omp simd
                for (int li = 0; li < BM; li++) {
                    int gi = bi * BM + li + 1;
                    int gj = bj * BN + 0;
                    HL(A_left, bi, bj, li)  = (double)(gi * (gj + 2)) / N;
                    HL(B_left, bi, bj, li)  = (double)(gi * (gj + 3)) / N;
                }

                // RIGHT halo: gj = bj*BN + BN + 1
                #pragma omp simd
                for (int li = 0; li < BM; li++) {
                    int gi = bi * BM + li + 1;
                    int gj = bj * BN + BN + 1;
                    HL(A_right, bi, bj, li) = (double)(gi * (gj + 2)) / N;
                    HL(B_right, bi, bj, li) = (double)(gi * (gj + 3)) / N;
                }
            }
        }
    }

    /* ----------------- Compute kernel for one block -----------------
       Reads from 'curr' interior + curr halos, writes into 'next' interior.
       We keep two sets of halos (curr and next) so we can do pure Jacobi:
         - compute next = stencil(curr, curr_halos)
         - then fill next_halos from next interior (exchange)
    --------------------------------------------------------------- */
    inline void compute_block(double *curr_blk, double *next_blk,
                              double *curr_top, double *curr_bottom,
                              double *curr_left, double *curr_right,
                              int bi, int bj) const {
        #pragma omp unroll
        for (int ii = 0; ii < BM; ++ii) {
            #pragma omp simd
            for (int jj = 0; jj < BN; ++jj) {
                // center
                double center = BLK(curr_blk, bi, bj, ii, jj);

                // left
                double left = (jj > 0)
                    ? BLK(curr_blk, bi, bj, ii, jj - 1)
                    : HL(curr_left, bi, bj, ii);

                // right
                double right = (jj < BN - 1)
                    ? BLK(curr_blk, bi, bj, ii, jj + 1)
                    : HL(curr_right, bi, bj, ii);

                // top
                double top = (ii > 0)
                    ? BLK(curr_blk, bi, bj, ii - 1, jj)
                    : HT(curr_top, bi, bj, jj);

                // bottom
                double bottom = (ii < BM - 1)
                    ? BLK(curr_blk, bi, bj, ii + 1, jj)
                    : HT(curr_bottom, bi, bj, jj);

                BLK(next_blk, bi, bj, ii, jj) = 0.2 * (center + left + right + top + bottom);
            }
        }
    }

    /* ----------------- Exchange halos for 'next' array -----------------
       For every block, copy its boundary values from next_blk into the neighbor
       block's halo slots in next_top/next_bottom/next_left/next_right.

       If a block sits on a physical boundary (bi==0, bi==NB_I-1, bj==0, bj==NB_J-1)
       we set the corresponding halo values to the undecomposed formula (i=0 or i=N+1 etc).
    --------------------------------------------------------------- */
    inline void exchange_halos_from_next(double *next_blk,
                                         double *next_top, double *next_bottom,
                                         double *next_left, double *next_right) const {

        #pragma omp parallel for collapse(2)
        for (int bi = 0; bi < NB_I; ++bi) {
            for (int bj = 0; bj < NB_J; ++bj) {
                // compute global base indices
                int gi_base = bi * BM;      // gi for ii==0 is gi_base + 0
                int gj_base = bj * BN;      // gj for jj==0 is gj_base + 0

                // right -> neighbor's left
                if (bj < NB_J - 1) {
                    int dst_bi = bi, dst_bj = bj + 1;
                    #pragma omp simd
                    for (int li = 0; li < BM; ++li) {
                        // source column is jj = BN-1
                        double val = BLK(next_blk, bi, bj, li, BN - 1);
                        HL(next_left, dst_bi, dst_bj, li) = val;
                    }
                } else {
                    // physical right boundary: set next_right for this block (gj = gj_base + BN + 1)
                    #pragma omp simd
                    for (int li = 0; li < BM; ++li) {
                        int gi = gi_base + li + 1;    // interior gi
                        int gj = gj_base + BN + 1;    // global right boundary column
                        HL(next_right, bi, bj, li) = (double)(gi * (gj + 2)) / N;
                    }
                }

                // left -> neighbor's right
                if (bj > 0) {
                    int dst_bi = bi, dst_bj = bj - 1;
                    #pragma omp simd
                    for (int li = 0; li < BM; ++li) {
                        double val = BLK(next_blk, bi, bj, li, 0); // left-most column
                        HL(next_right, dst_bi, dst_bj, li) = val;
                    }
                } else {
                    // physical left boundary (gj = gj_base + 0)
                    #pragma omp simd
                    for (int li = 0; li < BM; ++li) {
                        int gi = gi_base + li + 1;
                        int gj = gj_base + 0;
                        HL(next_left, bi, bj, li) = (double)(gi * (gj + 2)) / N;
                    }
                }

                // bottom -> neighbor's top
                if (bi < NB_I - 1) {
                    int dst_bi = bi + 1, dst_bj = bj;
                    #pragma omp simd
                    for (int lj = 0; lj < BN; ++lj) {
                        double val = BLK(next_blk, bi, bj, BM - 1, lj); // bottom row
                        HT(next_top, dst_bi, dst_bj, lj) = val;
                    }
                } else {
                    // physical bottom boundary (gi = gi_base + BM + 1)
                    #pragma omp simd
                    for (int lj = 0; lj < BN; ++lj) {
                        int gi = gi_base + BM + 1;
                        int gj = gj_base + lj + 1;
                        HT(next_bottom, bi, bj, lj) = (double)(gi * (gj + 2)) / N;
                    }
                }

                // top -> neighbor's bottom
                if (bi > 0) {
                    int dst_bi = bi - 1, dst_bj = bj;
                    #pragma omp simd
                    for (int lj = 0; lj < BN; ++lj) {
                        double val = BLK(next_blk, bi, bj, 0, lj); // top row
                        HT(next_bottom, dst_bi, dst_bj, lj) = val;
                    }
                } else {
                    // physical top boundary (gi = gi_base + 0)
                    #pragma omp simd
                    for (int lj = 0; lj < BN; ++lj) {
                        int gi = gi_base + 0;
                        int gj = gj_base + lj + 1;
                        HT(next_top, bi, bj, lj) = (double)(gi * (gj + 2)) / N;
                    }
                }
            }
        }
    }

    /* ----------------- One iteration: compute then exchange ----------------- */
    void run_iteration_once(double *curr_blk, double *curr_top, double *curr_bottom,
                            double *curr_left, double *curr_right,
                            double *next_blk, double *next_top, double *next_bottom,
                            double *next_left, double *next_right) {

        // 1) compute next interiors from curr interiors and curr halos
        #pragma omp parallel for collapse(2)
        for (int bi = 0; bi < NB_I; ++bi) {
            for (int bj = 0; bj < NB_J; ++bj) {
                compute_block(curr_blk, next_blk,
                              curr_top, curr_bottom, curr_left, curr_right,
                              bi, bj);
            }
        }

        // 2) exchange halos by copying from next interiors into next halos
        //    (ensures full Jacobi semantics for next iteration)
        exchange_halos_from_next(next_blk, next_top, next_bottom, next_left, next_right);
    }

    /* ----------------- Convenience wrapper: run T steps ----------------- */
    double run(int tsteps) {
        // Start with A as current and A's halos; B and its halos are next
        double *curr_blk = A_blk;
        double *next_blk = B_blk;
        double *curr_top = A_top, *curr_bottom = A_bottom, *curr_left = A_left, *curr_right = A_right;
        double *next_top = B_top, *next_bottom = B_bottom, *next_left = B_left, *next_right = B_right;

        double t0 = omp_get_wtime();

        for (int t = 0; t < tsteps; ++t) {
            // compute next from curr, then fill next halos
            run_iteration_once(curr_blk, curr_top, curr_bottom, curr_left, curr_right,
                               next_blk, next_top, next_bottom, next_left, next_right);

            // swap curr<->next pointers and halos for next iteration
            std::swap(curr_blk, next_blk);
            std::swap(curr_top, next_top);
            std::swap(curr_bottom, next_bottom);
            std::swap(curr_left, next_left);
            std::swap(curr_right, next_right);
        }

        double t1 = omp_get_wtime();
        return t1 - t0;
    }

    /* ----------------- Checksum (sum of interior in A_blk) ----------------- */
    double checksum() const {
        double s = 0.0;
        for (int bi = 0; bi < NB_I; ++bi)
            for (int bj = 0; bj < NB_J; ++bj)
                for (int ii = 0; ii < BM; ++ii)
                    for (int jj = 0; jj < BN; ++jj)
                        s += BLK(A_blk, bi, bj, ii, jj);
        return s;
    }
};

#endif // Jacobi2DBlockedV2_STYLE_H
