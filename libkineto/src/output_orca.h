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

class OrcaTraceLogger : public ActivityLogger {
 public:
  explicit OrcaTraceLogger();

  void handleDeviceInfo(const DeviceInfo& info, uint64_t time) override;

  void handleOverheadInfo(const OverheadInfo& info, int64_t time) override;

  void handleResourceInfo(const ResourceInfo& info, int64_t time) override;

  void handleTraceSpan(const TraceSpan& span) override;

  void handleActivity(const ITraceActivity& activity) override;
  void handleGenericActivity(const GenericTraceActivity& activity) override;

  void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata,
      const std::string& device_properties) override;

  void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata,
      const std::unordered_map<std::string, std::string>& device_properties)
      override;

  void finalizeTrace(
      const Config& config,
      std::unique_ptr<ActivityBuffers> buffers,
      int64_t endTime,
      std::unordered_map<std::string, std::vector<std::string>>& metadata)
      override;

  void setTimestep(int timestep) override {
    timestep_ = timestep;
  }

  void setRank(int rank) override {
    rank_ = rank;
  }

  // must be called before use
  void bindTracers(
      const KinetoTorchOpTracerRef& kinetoTorchOpTracer,
      const KinetoMiscTracerRef& kinetoMiscTracer,
      const KinetoMetadataTracerRef& kinetoMetadataTracer) {
    kinetoTorchOpTracer_ = kinetoTorchOpTracer;
    kinetoMiscTracer_ = kinetoMiscTracer;
    kinetoMetadataTracer_ = kinetoMetadataTracer;
  }

 private:
  constexpr static const char* kKinetoTorchOpProbeName = "torch_op";
  constexpr static const char* kKinetoMiscProbeName = "misc";
  constexpr static const char* kKinetoMetadataProbeName = "metadata";

  KinetoTorchOpTracerRef kinetoTorchOpTracer_{nullptr};
  KinetoMiscTracerRef kinetoMiscTracer_{nullptr};
  KinetoMetadataTracerRef kinetoMetadataTracer_{nullptr};
  int rank_ = -1;
  int timestep_;
  DistributedInfo distInfo_ = DistributedInfo();
  std::unordered_map<std::string, pgConfig> pgMap = {};

  void handleLink(
      char type,
      const ITraceActivity& e,
      int64_t id,
      const std::string& name);

  void addIterationMarker(const TraceSpan& span);

  void handleGenericInstantEvent(const ITraceActivity& op);

  void addOnDemandDistMetadata();

  void handleGenericLink(const ITraceActivity& activity);

  void addKinetoTorchOpEvent(const KinetoTorchOpEvent& event) {
    auto probe_id = kinetoTorchOpTracer_->GetProbeID(kKinetoTorchOpProbeName);
    kinetoTorchOpTracer_->AddRow(probe_id, event);
  }

  void addKinetoMiscEvent(const KinetoMiscEvent& event) {
    auto probe_id = kinetoMiscTracer_->GetProbeID(kKinetoMiscProbeName);
    kinetoMiscTracer_->AddRow(probe_id, event);
  }

  void addKinetoMetadataEvent(const KinetoMetadataEvent& event) {
    auto probe_id = kinetoMetadataTracer_->GetProbeID(kKinetoMetadataProbeName);
    kinetoMetadataTracer_->AddRow(probe_id, event);
  }
};

} // namespace KINETO_NAMESPACE