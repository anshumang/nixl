/*
 * Microbenchmark: atomic fetch_add vs thread_local increment
 *
 * Measures throughput of a shared atomic counter vs per-thread counters
 * under varying thread counts, simulating the round-robin rail selection
 * path in prepareAndSubmitTransfer().
 *
 * Build: g++ -O2 -std=c++17 -pthread -o atomic_vs_tls atomic_vs_tls.cpp
 * Run:   ./atomic_vs_tls
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>
#include <barrier>

static constexpr uint64_t OPS_PER_THREAD = 1'000'000;

static std::atomic<size_t> g_atomic_counter{0};

static void bench_atomic(std::barrier<> &start_barrier, uint64_t ops) {
    start_barrier.arrive_and_wait();
    for (uint64_t i = 0; i < ops; ++i) {
        g_atomic_counter.fetch_add(1, std::memory_order_relaxed);
    }
}

static void bench_tls(std::barrier<> &start_barrier, uint64_t ops) {
    static thread_local size_t tls_counter{0};
    start_barrier.arrive_and_wait();
    for (uint64_t i = 0; i < ops; ++i) {
        tls_counter++;
    }
}

static double run_bench(int num_threads, bool use_atomic) {
    g_atomic_counter.store(0);
    std::barrier start_barrier(num_threads + 1);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        if (use_atomic)
            threads.emplace_back(bench_atomic, std::ref(start_barrier), OPS_PER_THREAD);
        else
            threads.emplace_back(bench_tls, std::ref(start_barrier), OPS_PER_THREAD);
    }

    start_barrier.arrive_and_wait();
    auto t0 = std::chrono::steady_clock::now();

    for (auto &t : threads) t.join();

    auto t1 = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    double total_ops = static_cast<double>(num_threads) * OPS_PER_THREAD;
    return total_ops / elapsed_s;
}

int main() {
    const int thread_counts[] = {1, 2, 4, 8, 16, 32, 48, 64, 96, 192};
    const int num_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);
    constexpr int WARMUP = 1;
    constexpr int TRIALS = 3;

    std::printf("%-8s  %15s  %15s  %10s\n",
                "Threads", "Atomic (Mops/s)", "TLS (Mops/s)", "Ratio");
    std::printf("%-8s  %15s  %15s  %10s\n",
                "-------", "---------------", "---------------", "----------");

    for (int c = 0; c < num_counts; ++c) {
        int nt = thread_counts[c];

        // warmup
        for (int w = 0; w < WARMUP; ++w) {
            run_bench(nt, true);
            run_bench(nt, false);
        }

        double best_atomic = 0, best_tls = 0;
        for (int t = 0; t < TRIALS; ++t) {
            double a = run_bench(nt, true);
            double l = run_bench(nt, false);
            if (a > best_atomic) best_atomic = a;
            if (l > best_tls) best_tls = l;
        }

        std::printf("%-8d  %15.2f  %15.2f  %9.1fx\n",
                    nt, best_atomic / 1e6, best_tls / 1e6, best_tls / best_atomic);
    }

    return 0;
}
