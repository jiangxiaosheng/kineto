/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <fstream>
#include <map>
#include <ostream>
#include <ratio>
#include <string>
#include <thread>
#include <unordered_map>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ActivityBuffers.h"
#include "GenericTraceActivity.h"
#include "output_base.h"
#include "time_since_epoch.h"

namespace KINETO_NAMESPACE {
// Previous declaration of TraceSpan is struct. Must match the same here.
struct TraceSpan;
} // namespace KINETO_NAMESPACE

namespace KINETO_NAMESPACE {

class Config;

class ChromeTraceLogger : public libkineto::ActivityLogger {
 public:
  explicit ChromeTraceLogger(const std::string& traceFileName);

  // Note: the caller of these functions should handle concurrency
  // i.e., we these functions are not thread-safe
  void handleDeviceInfo(const DeviceInfo& info, uint64_t time) override;

  void handleOverheadInfo(const OverheadInfo& info, int64_t time) override;

  void handleResourceInfo(const ResourceInfo& info, int64_t time) override;

  void handleTraceSpan(const TraceSpan& span) override;

  void handleActivity(const ITraceActivity& activity) override;
  void handleGenericActivity(const GenericTraceActivity& activity) override;

  void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata,
      const std::string& device_properties) override;

  void finalizeTrace(
      const Config& config,
      std::unique_ptr<ActivityBuffers> buffers,
      int64_t endTime,
      std::unordered_map<std::string, std::vector<std::string>>& metadata)
      override;

  std::string traceFileName() const {
    return fileName_;
  }

 protected:
  void finalizeTrace(
      int64_t endTime,
      std::unordered_map<std::string, std::vector<std::string>>& metadata);

 private:
  // Create a flow event (arrow)
  void handleLink(
      char type,
      const ITraceActivity& e,
      int64_t id,
      const std::string& name);

  void addIterationMarker(const TraceSpan& span);

  void openTraceFile();

  void handleGenericInstantEvent(const ITraceActivity& op);

  void handleGenericLink(const ITraceActivity& activity);

  void metadataToJSON(
      const std::unordered_map<std::string, std::string>& metadata);

  void sanitizeStrForJSON(std::string& value);

  void addOnDemandDistMetadata();

  std::string fileName_;
  std::string tempFileName_;
  std::ofstream traceOf_;
  DistributedInfo distInfo_ = DistributedInfo();
  // Map of all observed process groups to their configs in trace. Key is
  // pg_name, value is pgConfig that will be used to populate pg_config in
  // distributedInfo of trace
  std::unordered_map<std::string, pgConfig> pgMap = {};
};

} // namespace KINETO_NAMESPACE
