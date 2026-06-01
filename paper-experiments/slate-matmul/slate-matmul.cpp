#include <mpi.h>
#include <slate/slate.hh>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "MPI_THREAD_MULTIPLE not provided by the MPI implementation\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int64_t n  = 0;
    int     p  = 0, q = 0;
    int64_t nb = 384;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      ((arg == "-n"  || arg == "--n")  && i + 1 < argc) n  = std::stoll(argv[++i]);
        else if ((arg == "-p"  || arg == "--p")  && i + 1 < argc) p  = std::stoi (argv[++i]);
        else if ((arg == "-q"  || arg == "--q")  && i + 1 < argc) q  = std::stoi (argv[++i]);
        else if ((arg == "-nb" || arg == "--nb") && i + 1 < argc) nb = std::stoll(argv[++i]);
        else {
            if (myrank == 0)
                fprintf(stderr, "Error: unknown or malformed argument '%s'\n", arg.c_str());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    auto fail = [&](const char* msg) {
        if (myrank == 0) fprintf(stderr, "Error: %s\n", msg);
        MPI_Abort(MPI_COMM_WORLD, 1);
    };

    if (n  <= 0) fail("missing or invalid -n (must be > 0)");
    if (p  <= 0) fail("missing or invalid -p (must be > 0)");
    if (q  <= 0) fail("missing or invalid -q (must be > 0)");
    if (nb <= 0) fail("missing or invalid -nb (must be > 0)");
    if (p * q != nprocs) fail("p*q must equal MPI nprocs");

    if (myrank == 0) {
        printf("testing %lldx%lld with %lldx%lld on %dx%dx1 grid\n",
               (long long)n, (long long)n, (long long)n, (long long)n, p, q);
        printf("SLATE tile size nb = %lld\n", (long long)nb);
    }

    slate::Matrix<double> A(n, n, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> B(n, n, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> C(n, n, nb, p, q, MPI_COMM_WORLD);

    A.insertLocalTiles();
    B.insertLocalTiles();
    C.insertLocalTiles();

    // Fill A and B with a constant so DGEMM does real work; C is initialised to 0
    // (gemm writes it anyway when beta = 0, but explicit init is cheap and safer).
    slate::set(1.0, A);
    slate::set(1.0, B);
    slate::set(0.0, C);

    slate::Options opts = {
        {slate::Option::Target,    slate::Target::HostTask},
        {slate::Option::Lookahead, 1}
    };

    // Warm-up: pays for first-touch allocation, OpenMP team startup,
    // any deferred internal SLATE setup. Discarded.
    slate::gemm(1.0, A, B, 0.0, C, opts);
    MPI_Barrier(MPI_COMM_WORLD);

    // Timed call
    double t0 = MPI_Wtime();
    slate::gemm(1.0, A, B, 0.0, C, opts);
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // Take the max time across ranks (the slowest rank is the wallclock cost)
    double dt_local = t1 - t0;
    double dt_max;
    MPI_Allreduce(&dt_local, &dt_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (myrank == 0) {
        double gflops = 2.0 * (double)n * (double)n * (double)n / dt_max / 1e9;
        printf("Time for SLATE gemm: %lf sec\n", dt_max);
        printf("Performance: %lf GF/s\n", gflops);
    }

    MPI_Finalize();
    return 0;
}
