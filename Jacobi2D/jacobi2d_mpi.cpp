#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define IDX(j, i, stride) ((j) * (stride) + (i))

typedef struct {
    int rank;
    int nprocs;
    int px, py;
    int coord_x, coord_y;
    
    int nx_global, ny_global;
    
    int nx_local, ny_local;
    int start_x, start_y;
    
    int north, south, east, west;
    
    MPI_Comm cart_comm;

    MPI_Datatype col_type;
} DomainInfo;

void initialize_domain(DomainInfo *domain, int nx_global, int ny_global, int px, int py) {
    MPI_Comm_rank(MPI_COMM_WORLD, &domain->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &domain->nprocs);
    
    domain->px = px;
    domain->py = py;
    domain->nx_global = nx_global;
    domain->ny_global = ny_global;
    
    if (domain->nprocs != px * py) {
        if (domain->rank == 0) {
            fprintf(stderr, "Error: Number of processes (%d) != px * py (%d * %d = %d)\n",
                    domain->nprocs, px, py, px * py);
        }
        MPI_Finalize();
        exit(1);
    }
    
    int dims[2] = {py, px};
    int periods[2] = {0, 0};
    int reorder = 0;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &domain->cart_comm);
    
    int coords[2];
    MPI_Cart_coords(domain->cart_comm, domain->rank, 2, coords);
    domain->coord_y = coords[0];
    domain->coord_x = coords[1];
    
    MPI_Cart_shift(domain->cart_comm, 0, 1, &domain->north, &domain->south);
    MPI_Cart_shift(domain->cart_comm, 1, 1, &domain->west, &domain->east);
    
    int base_nx = nx_global / px;
    int remainder_x = nx_global % px;
    
    int base_ny = ny_global / py;
    int remainder_y = ny_global % py;
    
    if (domain->coord_x < remainder_x) {
        domain->nx_local = base_nx + 1;
        domain->start_x = domain->coord_x * (base_nx + 1);
    } else {
        domain->nx_local = base_nx;
        domain->start_x = remainder_x * (base_nx + 1) + (domain->coord_x - remainder_x) * base_nx;
    }
    
    if (domain->coord_y < remainder_y) {
        domain->ny_local = base_ny + 1;
        domain->start_y = domain->coord_y * (base_ny + 1);
    } else {
        domain->ny_local = base_ny;
        domain->start_y = remainder_y * (base_ny + 1) + (domain->coord_y - remainder_y) * base_ny;
    }

    int nx = domain->nx_local;
    int ny = domain->ny_local;

    int stride = nx + 2;
    // Create column datatype for halo exchange (ny elements, stride apart)
    MPI_Type_vector(ny, 1, stride, MPI_DOUBLE, &domain->col_type);
    MPI_Type_commit(&domain->col_type);
}

void finalize_domain(DomainInfo *domain) {
}

void exchange_halos(double *u, DomainInfo *domain) {
    int nx = domain->nx_local;
    int ny = domain->ny_local;
    int stride = nx + 2;
    
    // Neighbor order for 2D Cartesian with dims={py,px}:
    // [north (dim0,-), south (dim0,+), west (dim1,-), east (dim1,+)]
    
    // Send/recv counts: rows use nx elements, columns use 1 col_type
    int sendcounts[4] = {nx, nx, 1, 1};
    int recvcounts[4] = {nx, nx, 1, 1};
    
    // Displacements in bytes from start of buffer
    MPI_Aint senddispls[4], recvdispls[4];
    
    // Send displacements (interior rows/columns)
    senddispls[0] = IDX(1, 1, stride) * sizeof(double);      // north: send row 1
    senddispls[1] = IDX(ny, 1, stride) * sizeof(double);     // south: send row ny
    senddispls[2] = IDX(1, 1, stride) * sizeof(double);      // west: send col 1
    senddispls[3] = IDX(1, nx, stride) * sizeof(double);     // east: send col nx
    
    // Receive displacements (ghost rows/columns)
    recvdispls[0] = IDX(0, 1, stride) * sizeof(double);      // north: recv into row 0
    recvdispls[1] = IDX(ny+1, 1, stride) * sizeof(double);   // south: recv into row ny+1
    recvdispls[2] = IDX(1, 0, stride) * sizeof(double);      // west: recv into col 0
    recvdispls[3] = IDX(1, nx+1, stride) * sizeof(double);   // east: recv into col nx+1
    
    // Types: rows are contiguous (MPI_DOUBLE), columns use col_type
    MPI_Datatype sendtypes[4] = {MPI_DOUBLE, MPI_DOUBLE, domain->col_type, domain->col_type};
    MPI_Datatype recvtypes[4] = {MPI_DOUBLE, MPI_DOUBLE, domain->col_type, domain->col_type};
    
    // Single collective call handles all 4 neighbor exchanges
    // MPI_PROC_NULL neighbors are automatically skipped
    MPI_Neighbor_alltoallw(u, sendcounts, senddispls, sendtypes,
                           u, recvcounts, recvdispls, recvtypes,
                           domain->cart_comm);
}


double checksum(double *u, DomainInfo *domain) {
    int nx = domain->nx_local;
    int ny = domain->ny_local;
    int stride = nx + 2;
    double local_sum = 0.0;
    
    for (int j = 1; j < ny + 1; j++) {
        for (int i = 1; i < nx + 1; i++) {
            local_sum += u[IDX(j, i, stride)];
        }
    }

    return local_sum;
}

void jacobi_step(double *u, double *u_new, DomainInfo *domain) {
    int nx = domain->nx_local;
    int ny = domain->ny_local;
    int stride = nx + 2;
    
    // 5-point stencil with 0.2 coefficient (matches baseline)
    for (int j = 1; j <= ny; j++) {
        for (int i = 1; i <= nx; i++) {
            u_new[IDX(j, i, stride)] = 0.2 * (
                u[IDX(j, i, stride)] +       // center
                u[IDX(j, i-1, stride)] +     // left
                u[IDX(j, i+1, stride)] +     // right
                u[IDX(j-1, i, stride)] +     // up
                u[IDX(j+1, i, stride)]       // down
            );
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = 2048;
    int nx_global = N;
    int ny_global = N;
    int px = 2;
    int py = 4;
    int tsteps = 100;
    
    if (argc >= 3) {
        N = atoi(argv[1]);
        nx_global = N;
        ny_global = N;
    }
    if (argc >= 5) {
        px = atoi(argv[2]);
        py = atoi(argv[3]);
    }
    if (argc >= 6) {
        tsteps = atoi(argv[4]);
    }
    
    DomainInfo domain;
    initialize_domain(&domain, nx_global, ny_global, px, py);

    MPI_Barrier(MPI_COMM_WORLD);
    
    int nx = domain.nx_local;
    int ny = domain.ny_local;
    int stride = nx + 2;
    
    // Allocate 1D arrays with ghost cells
    double *__restrict__ u = (double*)std::aligned_alloc(64, ((ny + 2) * stride * sizeof(double)));
    double *__restrict__ u_new = (double*)std::aligned_alloc(64, ((ny + 2) * stride * sizeof(double)));
    
    // Initialize to match baseline: A[i][j] = (double)(i * (j + 2)) / N
    // In global coordinates: global_i is row, global_j is column
    // Baseline uses: A[i * (N + 2) + j] = (double)(i * (j + 2)) / N
    // where i and j go from 0 to N+1 (including boundaries)
    for (int j = 0; j <= ny + 1; j++) {
        for (int i = 0; i <= nx + 1; i++) {
            // Local (j, i) maps to global (global_row, global_col)
            // j is local row, i is local column
            // global_row = start_y + j - 1 for interior (j=1..ny)
            // But we need to handle ghost cells too
            
            int global_row, global_col;
            
            // Handle ghost cells and interior
            if (j == 0) {
                global_row = domain.start_y - 1;  // Ghost row above
            } else if (j == ny + 1) {
                global_row = domain.start_y + ny; // Ghost row below
            } else {
                global_row = domain.start_y + (j - 1);
            }
            
            if (i == 0) {
                global_col = domain.start_x - 1;  // Ghost col left
            } else if (i == nx + 1) {
                global_col = domain.start_x + nx; // Ghost col right
            } else {
                global_col = domain.start_x + (i - 1);
            }
            
            // Convert to baseline's indexing (0 to N+1)
            // In baseline, row index i goes 0..N+1, col index j goes 0..N+1
            // Our global_row corresponds to baseline's i
            // Our global_col corresponds to baseline's j
            int baseline_i = global_row + 1;  // +1 because baseline has boundary at 0
            int baseline_j = global_col + 1;  // +1 because baseline has boundary at 0
            
            // Clamp to valid range for ghost cells at domain boundaries
            if (baseline_i >= 0 && baseline_i <= N + 1 && 
                baseline_j >= 0 && baseline_j <= N + 1) {
                u[IDX(j, i, stride)] = (double)(baseline_i * (baseline_j + 2)) / N;
                u_new[IDX(j, i, stride)] = (double)(baseline_i * (baseline_j + 3)) / N;
            }
        }
    }

    if (rank == 0) {
        printf("Jacobi2D MPI\n");
        printf("N=%d, tsteps=%d, processor grid=%dx%d\n", N, tsteps, px, py);
    }

    double start_time = MPI_Wtime();
    
    for (int iter = 0; iter < tsteps; iter++) {
        exchange_halos(u, &domain);
        jacobi_step(u, u_new, &domain);

        // Swap arrays
        double *__restrict__ temp = u;
        u = u_new;
        u_new = temp;
    }
    
    double end_time = MPI_Wtime();

    double local_checksum = checksum(u, &domain);
    double global_checksum = 0.0;

    MPI_Reduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Time: %.6f seconds\n", end_time - start_time);
        printf("Checksum: %.10e\n", global_checksum);
        printf("GFLOPS: %.3f\n", (double)N * N * tsteps * 6 / (end_time - start_time) / 1e9);
    }
    
    std::free(u);
    std::free(u_new);
    
    finalize_domain(&domain);
    MPI_Comm_free(&domain.cart_comm);
    MPI_Finalize();
    
    return 0;
}
