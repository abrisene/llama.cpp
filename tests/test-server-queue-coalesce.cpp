#include "server-queue.h"

#include "ggml.h"

#include <atomic>
#include <chrono>
#include <thread>

static server_task completion_task(server_queue & queue) {
    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id = queue.get_new_id();
    return task;
}

static bool test_single_request_has_no_wait() {
    server_queue queue;
    queue.set_admission_coalesce(50000, 4);

    std::atomic<int> processed{0};
    // upstream's callback takes an is_yielding flag and returns whether the task
    // was accepted; these tests never yield, so always accept
    queue.on_new_task([&](server_task &&, bool) {
        processed.fetch_add(1, std::memory_order_relaxed);
        return true;
    });
    queue.on_has_batch_capacity([&]() {
        return processed.load(std::memory_order_relaxed) < 4;
    });
    queue.on_update_slots([&]() {
        queue.terminate();
    });
    queue.post(completion_task(queue));

    queue.start_loop();
    const auto stats = queue.get_coalesce_stats();
    return processed.load(std::memory_order_relaxed) == 1 &&
           stats.wait_cycles == 0;
}

static bool test_burst_collects_late_request() {
    server_queue queue;
    queue.set_admission_coalesce(50000, 3);

    std::atomic<int> processed{0};
    // upstream's callback takes an is_yielding flag and returns whether the task
    // was accepted; these tests never yield, so always accept
    queue.on_new_task([&](server_task &&, bool) {
        processed.fetch_add(1, std::memory_order_relaxed);
        return true;
    });
    queue.on_has_batch_capacity([&]() {
        return processed.load(std::memory_order_relaxed) < 3;
    });
    queue.on_update_slots([&]() {
        queue.terminate();
    });
    queue.post(completion_task(queue));
    queue.post(completion_task(queue));

    std::thread late_request([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        queue.post(completion_task(queue));
    });
    queue.start_loop();
    late_request.join();

    const auto stats = queue.get_coalesce_stats();
    return processed.load(std::memory_order_relaxed) == 3 &&
           stats.wait_cycles >= 1 &&
           stats.wakeups >= 1 &&
           stats.admitted_tasks == 3;
}

static bool test_full_batch_has_no_wait() {
    server_queue queue;
    queue.set_admission_coalesce(50000, 2);

    std::atomic<int> processed{0};
    // upstream's callback takes an is_yielding flag and returns whether the task
    // was accepted; these tests never yield, so always accept
    queue.on_new_task([&](server_task &&, bool) {
        processed.fetch_add(1, std::memory_order_relaxed);
        return true;
    });
    queue.on_has_batch_capacity([&]() {
        return processed.load(std::memory_order_relaxed) < 2;
    });
    queue.on_update_slots([&]() {
        queue.terminate();
    });
    queue.post(completion_task(queue));
    queue.post(completion_task(queue));
    queue.start_loop();

    const auto stats = queue.get_coalesce_stats();
    return processed.load(std::memory_order_relaxed) == 2 && stats.wait_cycles == 0;
}

static bool test_timeout_shrinks_window() {
    server_queue queue;
    queue.set_admission_coalesce(20000, 3);

    std::atomic<int> processed{0};
    // upstream's callback takes an is_yielding flag and returns whether the task
    // was accepted; these tests never yield, so always accept
    queue.on_new_task([&](server_task &&, bool) {
        processed.fetch_add(1, std::memory_order_relaxed);
        return true;
    });
    queue.on_has_batch_capacity([&]() {
        return true;
    });
    queue.on_update_slots([&]() {
        queue.terminate();
    });
    queue.post(completion_task(queue));
    queue.post(completion_task(queue));
    queue.start_loop();

    const auto stats = queue.get_coalesce_stats();
    return processed.load(std::memory_order_relaxed) == 2 &&
           stats.timeouts == 1 &&
           stats.current_wait_us == 15000;
}

int main() {
    // required on Windows: ggml_time_ms/us divide by a timer frequency that stays
    // zero until this runs, so the queue's ggml_time_ms() call would trap.
    // normally llama_backend_init() does this, but these tests never load a model.
    ggml_time_init();

    return test_single_request_has_no_wait() &&
           test_burst_collects_late_request() &&
           test_full_batch_has_no_wait() &&
           test_timeout_shrinks_window() ? 0 : 1;
}
