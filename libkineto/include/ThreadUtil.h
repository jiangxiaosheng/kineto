/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace libkineto {

int32_t systemThreadId(bool cache = true);
int32_t threadId();
bool setThreadName(const std::string& name);
std::string getThreadName();

int32_t processId(bool cache = true);
std::string processName(int32_t pid);

// Return a list of pids and process names for the current process
// and its parents.
std::vector<std::pair<int32_t, std::string>> pidCommandPairsOfAncestors();

// Resets all cached Thread local state, this must be done on
// forks to prevent stale values from being retained.
void resetTLS();

// Don't pollute the public namespace.
namespace impl {

// The ThreadPool class manages a pool of worker threads to execute tasks.
class ThreadPool {
 public:
  // Constructor to create a thread pool with a specified number of threads.
  explicit ThreadPool(size_t num_threads);

  // Enqueue a task to be executed by a worker thread.
  template <class F, class... Args>
  void enqueue(F&& f, Args&&... args);

  // Destructor to stop all worker threads and join them.
  ~ThreadPool();

 private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

// Enqueue a new task into the queue.
template <class F, class... Args>
void ThreadPool::enqueue(F&& f, Args&&... args) {
  auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);

  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    if (stop) {
      return;
    }
    tasks.emplace(task);
  }
  condition.notify_one();
}

} // namespace impl
} // namespace libkineto
