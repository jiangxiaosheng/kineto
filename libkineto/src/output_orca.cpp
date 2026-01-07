#include "output_orca.h"
#include <arrow/array/array_base.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/type_fwd.h>
#include <parquet/arrow/writer.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include "Logger.h"
#include "time_since_epoch.h"

namespace KINETO_NAMESPACE {

// TODO: Most of the code here is copied from the json logger.
// However, it turns out many logging code is unnecessary for the
// arrow table and HTA.
// The current solution is simply make those logging methods empty,
// but it would be better not call them in the first place since
// there are some other logics involved, which are unnecessary.

static inline int32_t sanitizeTid(int32_t tid) {
  // Convert all negative tids to its positive value. Create a specific case
  // for INT_MIN so it is obvious how it is being handled.
  if (tid == INT_MIN) {
    return 0;
  }
  return std::abs(tid);
}

MonTraceLogger::MonTraceLogger(const KinetoTracerRef& kinetoTracer, int rank)
    : kinetoTracer_(kinetoTracer), rank_(rank), timestep_(0) {}

// 'dur' = 0, do nothing
void MonTraceLogger::handleDeviceInfo(const DeviceInfo& info, uint64_t time) {}

// 'dur' = 0, do nothing
void MonTraceLogger::handleOverheadInfo(
    const OverheadInfo& info,
    int64_t time) {}

// 'cat' = 'Trace' or 'dur' = 0, do nothing
void MonTraceLogger::handleTraceSpan(const TraceSpan& span) {}

// 'cat' = 'Trace' or 'dur' = 0, do nothing
void MonTraceLogger::handleResourceInfo(
    const ResourceInfo& info,
    int64_t time) {}

// 'cat' = 'Trace' or 'dur' = 0, do nothing
void MonTraceLogger::handleGenericActivity(
    const GenericTraceActivity& activity) {
  handleActivity(activity);
}

// In json this writes metadata as the header, but it might be unneeded for HTA.
void MonTraceLogger::handleTraceStart(
    const std::unordered_map<std::string, std::string>& metadata,
    const std::string& device_properties) {}

// 'dur' = 0, do nothing
void MonTraceLogger::handleActivity(const ITraceActivity& op) {
  int64_t ts = op.timestamp();
  int64_t duration = op.duration();

  // 'cat' = 'Trace' is dropped
  auto cat = toString(op.type());
  if (std::strcmp(cat, "Trace") == 0)
    return;

  if (duration < 0) {
    // This should never happen but can occasionally suffer from regression in
    // handling incomplete events. Having negative duration in Chrome trace can
    // yield in very poor experience so add an extra guard before we generate
    // trace events.
    duration = 0;
  }

  if (op.type() == ActivityType::GPU_USER_ANNOTATION) {
    // The GPU user annotations start at the same time as the
    // first associated GPU op. Since they appear later
    // in the trace file, this causes a visualization issue in Chrome.
    // Make it start one ns earlier and end 2 ns later.
    ts -= 1;
    duration += 2; // Still need it to end at the original point rounded up.
  }

  std::string op_name = op.name() == "kernel" ? "Kernel" : op.name();
  int device = op.deviceId();
  int resource = op.resourceId();
  auto tid = sanitizeTid(resource);
  int64_t end = ts + duration;
  ts = std::ceil(ts / 1000.0);
  end = std::floor(end / 1000.0);
  duration = end - ts;

  auto extra_fields = op.getExtraFields();
  KinetoEvent event {
    .base = EventBase {
      .timestep = timestep_,
      .rank = rank_,
    },
    .cat = cat,
    .name = op_name,
    .pid = 0,
    .tid = resource,
    .ts = ts,
    .dur = duration,
    .end = end,
    .stream = device,
    .correlation = extra_fields.correlation,
    .bytes = extra_fields.bytes,
    .mem_bw = extra_fields.memBw,
    .wait_on_stream = extra_fields.waitOnStream,
    .wait_on_cuda_event = extra_fields.waitOnCudaEvent,
  };

  auto probe_id = kinetoTracer_->GetProbeID(kKinetoProbeName);
  kinetoTracer_->AddRow(probe_id, event);
}

void MonTraceLogger::finalizeTrace(
    const Config& config,
    std::unique_ptr<ActivityBuffers> buffers,
    int64_t endTime,
    std::unordered_map<std::string, std::vector<std::string>>& metadata) {
}

} // namespace KINETO_NAMESPACE