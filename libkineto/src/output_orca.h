#pragma once

#include "ActivityBuffers.h"
#include "GenericTraceActivity.h"
#include "kineto_tracer.h"
#include "output_base.h"

#include <arrow/api.h>
#include <arrow/scalar.h>
#include <arrow/type_fwd.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>

namespace KINETO_NAMESPACE {

class MonTraceLogger : public ActivityLogger {
 public:
  explicit MonTraceLogger(
      const KinetoTracerRef& kinetoTracer,
      int rank);

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

  void setTimestep(int timestep) override {
    timestep_ = timestep;
  }

 private:
  constexpr static const char* kKinetoProbeName = "kineto_events";
  KinetoTracerRef kinetoTracer_;
  int rank_;
  int timestep_;
};

} // namespace KINETO_NAMESPACE