/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include "libkineto.h"

namespace libkineto {
// forward declaration
class CpuTraceSnapshotInterface;

class ClientInterface {
 public:
  virtual ~ClientInterface() {}
  virtual void init() = 0;
  virtual void prepare(bool, bool, bool, bool, bool) = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
  // Flush the traces and return a snapshot of the trace
  // without terminating the profiler.
  virtual std::unique_ptr<CpuTraceSnapshotInterface> flush() = 0;
  // Different from stop(), shutdown() just cleans up the client state
  // and resources, and stops it, but does not drain the traces or process them.
  virtual void shutdown() = 0;
};

} // namespace libkineto
