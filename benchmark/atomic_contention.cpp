/*
 * Microbenchmark: atomic fetch_add contention under varying thread counts
 *
 * Tight-loop fetch_add on a shared atomic counter — worst-case contention
 * for the round-robin rail selection in prepareAndSubmitTransfer().
 *
 * Build: g++ -O2 -std=c++20 -pthread -o atomic_contention atomic_contention.cpp
 * Run:   ./atomic_contention
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>
#include <barrier>

static constexpr uint64_t OPS_PER_THREAD = 1'000'000;

static std::atomic<size_t> g_counter{0};

static void worker(std::barrier<> &start_barrier, uint64_t ops) {
    start_barrier.arrive_and_wait();
    for (uint64_t i = 0; i < ops; ++i) {
        g_counter.fetch_add(1, std::memory_order_relaxed);
    }
}

static double run(int num_threads) {
    g_counter.store(0);
    std::barrier start_barrier(num_threads + 1);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int i = 0; i < num_threads; ++i)
        threads.emplace_back(worker, std::ref(start_barrier), OPS_PER_THREAD);

    start_barrier.arrive_and_wait();
    auto t0 = std::chrono::steady_clock::now();

    for (auto &t : threads) t.join();

    auto t1 = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    return static_cast<double>(num_threads) * OPS_PER_THREAD / elapsed_s;
}

int main() {
    const int thread_counts[] = {1, 2, 4, 8, 16, 32, 48, 64, 96, 192};
    constexpr int WARMUP = 1;
    constexpr int TRIALS = 3;

    std::printf("%-8s  %15s  %12s\n", "Threads", "Mops/s", "ns/op");
    std::printf("%-8s  %15s  %12s\n", "-------", "---------------", "------------");

    for (int nt : thread_counts) {
        for (int w = 0; w < WARMUP; ++w) run(nt);

        double best = 0;
        for (int t = 0; t < TRIALS; ++t) {
            double r = run(nt);
            if (r > best) best = r;
        }

        std::printf("%-8d  %15.2f  %12.1f\n", nt, best / 1e6, 1e9 / (best / nt));
    }

    return 0;
}
