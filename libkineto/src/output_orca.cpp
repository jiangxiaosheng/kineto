#include "output_orca.h"
#include <arrow/array/array_base.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/type_fwd.h>
#include <parquet/arrow/writer.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include "ChromeTime.h"
#include "Logger.h"
#include "fmt/format.h"
#include "kineto_tracer.h"
namespace KINETO_NAMESPACE {

static constexpr int kSchemaVersion = 1;
static constexpr char kFlowStart = 's';
static constexpr char kFlowEnd = 'f';

// CPU op name that is used to store collectives metadata
// TODO: share the same string across c10d, profiler and libkineto
static constexpr const char* kParamCommsCallName = "record_param_comms";
// Collective function metadata populated from CPU op to GPU kernel
static constexpr const char* kCollectiveName = "Collective name";
static constexpr const char* kDtype = "dtype";
static constexpr const char* kInMsgNelems = "In msg nelems";
static constexpr const char* kOutMsgNelems = "Out msg nelems";
static constexpr const char* kGroupSize = "Group size";
static constexpr const char* kInSplit = "In split size";
static constexpr const char* kOutSplit = "Out split size";
static constexpr const char* kProcessGroupName = "Process Group Name";
static constexpr const char* kProcessGroupDesc = "Process Group Description";
static constexpr const char* kGroupRanks = "Process Group Ranks";
static constexpr const char* kInTensorsStart = "Input Tensors start";
static constexpr const char* kOutTensorsStart = "Output Tensors start";
static constexpr const char* kRank = "Rank";
static constexpr const char* kP2pSrc = "Src Rank";
static constexpr const char* kP2pDst = "Dst Rank";

static void sanitizeForNonReadableChars(std::string& value) {
  for (auto& c : value) {
    if (!std::isprint(c)) {
      LOG(WARNING) << "Non JSON compliant character found in string: " << value
                   << " Replacing with 'unknown'";
      value = "unknown";
      break;
    }
  }
}

static inline int32_t sanitizeTid(int32_t tid) {
  // Convert all negative tids to its positive value. Create a specific case
  // for INT_MIN so it is obvious how it is being handled.
  if (tid == INT_MIN) {
    return 0;
  }
  return std::abs(tid);
}

void OrcaTraceLogger::handleTraceStart(
    const std::unordered_map<std::string, std::string>& metadata,
    const std::string& device_properties) {}

// Record the metadata of the device.
// In chrome json it's "deviceProperties" field including compute
// capability, warp size, etc.
// In orca logger we map it to metadata events.
void OrcaTraceLogger::handleTraceStart(
    const std::unordered_map<std::string, std::string>& metadata,
    const std::unordered_map<std::string, std::string>& device_properties) {
  KinetoMetadataEvent metadata_event{
      .base =
          EventBase{
              .timestep = timestep_,
              .rank = rank_,
          },
  };

  metadata_event.key = "schemaVersion";
  metadata_event.value = std::to_string(kSchemaVersion);
  addKinetoMetadataEvent(metadata_event);

  for (const auto& [k, v] : metadata) {
    if (k == "distributedInfo") {
      distInfo_.distInfo_present_ = true;
    }
    metadata_event.key = k;
    metadata_event.value = v;
    addKinetoMetadataEvent(metadata_event);
  }

  for (const auto& [k, v] : device_properties) {
    metadata_event.key = k;
    metadata_event.value = v;
    addKinetoMetadataEvent(metadata_event);
  }
}

OrcaTraceLogger::OrcaTraceLogger() : timestep_(0) {}

void OrcaTraceLogger::handleDeviceInfo(const DeviceInfo& info, uint64_t time) {
  int64_t time_rel = transToRelativeTime(time);
  float time_rel_float = static_cast<float>(time_rel) / 1000.0f;
  std::string args = fmt::format(R"("name"={})", info.name);
  KinetoMiscEvent device_event{
      .base =
          EventBase{
              .timestep = timestep_,
              .rank = rank_,
          },
      .ph = 'M',
      .cat = "",
      .name = "process_name",
      .pid = info.id,
      .tid = 0,
      .ts = time_rel_float,
      .args = args,
  };
  addKinetoMiscEvent(device_event);

  args = fmt::format(R"("labels"={})", info.label);
  device_event = KinetoMiscEvent{
      .base =
          EventBase{
              .timestep = timestep_,
              .rank = rank_,
          },
      .ph = 'M',
      .cat = "",
      .name = "process_labels",
      .pid = info.id,
      .tid = 0,
      .ts = time_rel_float,
      .args = args,
  };
  addKinetoMiscEvent(device_event);

  args = fmt::format(R"("sort_index"={})", info.sortIndex);
  device_event = KinetoMiscEvent{
      .base =
          EventBase{
              .timestep = timestep_,
              .rank = rank_,
          },
      .ph = 'M',
      .cat = "",
      .name = "process_sort_index",
      .pid = info.id,
      .tid = 0,
      .ts = time_rel_float,
      .args = args,
  };
  addKinetoMiscEvent(device_event);
}

void OrcaTraceLogger::handleResourceInfo(
    const ResourceInfo& info,
    int64_t time) {
  time = transToRelativeTime(time);
  float time_rel_float = static_cast<float>(time) / 1000.0f;
  std::string args = fmt::format(R"("name"={})", info.name);
  KinetoMiscEvent resource_event{
      .base =
          EventBase{
              .timestep = timestep_,
              .rank = rank_,
          },
      .ph = 'M',
      .cat = "",
      .name = "thread_name",
      .pid = info.deviceId,
      .tid = sanitizeTid(static_cast<int32_t>(info.id)),
      .ts = time_rel_float,
      .args = args,
  };
  addKinetoMiscEvent(resource_event);

  args = fmt::format(R"("sort_index"={})", info.sortIndex);
  resource_event = KinetoMiscEvent{
      .base =
          EventBase{
              .timestep = timestep_,
              .rank = rank_,
          },
      .ph = 'M',
      .cat = "",
      .name = "thread_sort_index",
      .pid = info.deviceId,
      .tid = sanitizeTid(static_cast<int32_t>(info.id)),
      .ts = time_rel_float,
      .args = args,
  };
  addKinetoMiscEvent(resource_event);
}

void OrcaTraceLogger::handleOverheadInfo(
    const OverheadInfo& info,
    int64_t time) {
  time = transToRelativeTime(time);
  std::string args = fmt::format(R"("name"={})", info.name);
  KinetoMiscEvent overhead_event{
      .base =
          EventBase{
              .timestep = timestep_,
              .rank = rank_,
          },
      .ph = 'M',
      .cat = "",
      .name = "process_name",
      .pid = -1,
      .tid = 0,
      .ts = static_cast<float>(time) / 1000.0f,
      .args = args,
  };
  addKinetoMiscEvent(overhead_event);

  args = fmt::format(R"("sort_index"={})", 0x100000All);
  overhead_event = KinetoMiscEvent{
      .base =
          EventBase{
              .timestep = timestep_,
              .rank = rank_,
          },
      .ph = 'M',
      .cat = "",
      .name = "process_sort_index",
      .pid = -1,
      .tid = 0,
      .ts = static_cast<float>(time) / 1000.0f,
      .args = args,
  };
  addKinetoMiscEvent(overhead_event);
}

void OrcaTraceLogger::handleTraceSpan(const TraceSpan& span) {
  uint64_t start = transToRelativeTime(span.startTime);
  uint64_t dur = (span.endTime == 0) ? 0 : span.endTime - span.startTime;

  // trace spans use strings as the pid and tid, so we move them to the args
  // field and leave the pid and tid as -1.
  std::string args = fmt::format(
      R"("pid"=Spans,"tid"={},"dur"={},"Op count"={})",
      span.name,
      static_cast<float>(dur) / 1000.0f,
      span.opCount);

  KinetoMiscEvent span_event{
      .base =
          EventBase{
              .timestep = timestep_,
              .rank = rank_,
          },
      .ph = 'X',
      .cat = "Trace",
      .name = fmt::format("{}{} ({})", span.prefix, span.name, span.iteration),
      .pid = -1,
      .tid = -1,
      .ts = static_cast<float>(start) / 1000.0f,
      .args = args,
  };
  addKinetoMiscEvent(span_event);

  args = fmt::format(R"("pid"=Spans,"tid"=0,"sort_index"={})", 0x20000000ll);
  span_event = KinetoMiscEvent{
      .base =
          EventBase{
              .timestep = timestep_,
              .rank = rank_,
          },
      .ph = 'M',
      .cat = "",
      .name = "process_sort_index",
      .pid = -1,
      .tid = -1,
      .ts = static_cast<float>(start) / 1000.0f,
      .args = args,
  };
  addKinetoMiscEvent(span_event);

  addIterationMarker(span);
}

void OrcaTraceLogger::addIterationMarker(const TraceSpan& span) {
  uint64_t start = transToRelativeTime(span.startTime);
  std::string args = fmt::format(
      R"("pid"=Traces,"tid"=Trace {},"s"=g,"ts"={})",
      span.name,
      static_cast<float>(start) / 1000.0f);
  KinetoMiscEvent iteration_event{
      .base =
          EventBase{
              .timestep = timestep_,
              .rank = rank_,
          },
      .ph = 'i',
      .cat = "",
      .name = fmt::format("Iteration Start: {}", span.name),
      .pid = -1,
      .tid = -1,
      .ts = static_cast<float>(start) / 1000.0f,
      .args = args,
  };
  addKinetoMiscEvent(iteration_event);
}

void OrcaTraceLogger::handleGenericInstantEvent(const ITraceActivity& op) {
  uint64_t ts = transToRelativeTime(op.timestamp());
  auto [promoted_fields, op_metadata] = op.getMetadata();
  std::string args = fmt::format(R"("s"=t,{})", op_metadata);
  KinetoMiscEvent instant_event{
      .base =
          EventBase{
              .timestep = timestep_,
              .rank = rank_,
          },
      .ph = 'i',
      .cat = toString(op.type()),
      .name = op.name(),
      .pid = op.deviceId(),
      .tid = sanitizeTid(static_cast<int32_t>(op.resourceId())),
      .ts = static_cast<float>(ts) / 1000.0f,
      .args = args,
  };
}

void OrcaTraceLogger::handleGenericActivity(
    const GenericTraceActivity& activity) {
  handleActivity(activity);
}

// 'dur' = 0, do nothing
void OrcaTraceLogger::handleActivity(const ITraceActivity& op) {
  if (op.type() == ActivityType::CPU_INSTANT_EVENT) {
    handleGenericInstantEvent(op);
    return;
  }

  int64_t ts = op.timestamp();
  int64_t duration = op.duration();

  if (duration < 0) {
    duration = 0;
  }

  if (op.type() == ActivityType::GPU_USER_ANNOTATION) {
    ts -= 1;
    duration += 2;
  }

  KinetoTorchOpEvent torch_op_event{
      .base =
          EventBase{
              .timestep = timestep_,
              .rank = rank_,
          },
  };

  int external_id = 0;
  if (op.linkedActivity()) {
    external_id = op.linkedActivity()->correlationId();
  } else {
    static const std::set<libkineto::ActivityType> excludedTypes = {
        libkineto::ActivityType::GPU_MEMCPY,
        libkineto::ActivityType::GPU_MEMSET,
        libkineto::ActivityType::CONCURRENT_KERNEL,
        libkineto::ActivityType::CUDA_RUNTIME,
        libkineto::ActivityType::CUDA_DRIVER,
        libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
        libkineto::ActivityType::PRIVATEUSE1_DRIVER};
    if (excludedTypes.find(op.type()) == excludedTypes.end()) {
      external_id = op.correlationId();
    }
  }

  // Check the default values for promoted fields in
  // hta/configs/default_event_args.py If external id is not available in the
  // trace it should be set to -1
  if (external_id != 0) {
    torch_op_event.external_id = external_id;
  } else {
    torch_op_event.external_id = -1;
  }

  std::stringstream arg_values;
  const auto& [promoted_fields, op_metadata] = op.getMetadata();
  torch_op_event.stream = promoted_fields.stream;
  torch_op_event.correlation = promoted_fields.correlation;
  torch_op_event.mem_bw_gbps = promoted_fields.mem_bw;

  if (!op_metadata.empty()) {
    if (arg_values.tellp() > 0) {
      arg_values << ",";
    }
    arg_values << op_metadata;
  }

  if (op.type() == ActivityType::CONCURRENT_KERNEL && op.linkedActivity() &&
      op.linkedActivity()->name() == kParamCommsCallName) {
    const auto* collectiveRecord = op.linkedActivity();
    const auto& collectiveName =
        collectiveRecord->getMetadataValue(kCollectiveName);
    const auto& inMsgSize = collectiveRecord->getMetadataValue(kInMsgNelems);
    const auto& outMsgSize = collectiveRecord->getMetadataValue(kOutMsgNelems);
    const auto& groupSize = collectiveRecord->getMetadataValue(kGroupSize);
    const auto& dtype = collectiveRecord->getMetadataValue(kDtype);
    if (!collectiveName.empty() && !inMsgSize.empty() && !outMsgSize.empty() &&
        !groupSize.empty() && !dtype.empty()) {
      if (arg_values.tellp() > 0) {
        arg_values << ",";
      }
      arg_values << fmt::format(
          R"("{}"={},"{}"={},"{}"={},"{}"={},"{}"={})",
          kCollectiveName,
          collectiveName,
          kInMsgNelems,
          inMsgSize,
          kOutMsgNelems,
          outMsgSize,
          kGroupSize,
          groupSize,
          kDtype,
          dtype);
    }
    const auto& input_tensor_starts =
        collectiveRecord->getMetadataValue(kInTensorsStart);
    const auto output_tensor_starts =
        collectiveRecord->getMetadataValue(kOutTensorsStart);
    if (!input_tensor_starts.empty()) {
      if (arg_values.tellp() > 0) {
        arg_values << ",";
      }
      arg_values << fmt::format(
          R"("{}"={})", kInTensorsStart, input_tensor_starts);
    }
    if (!output_tensor_starts.empty()) {
      if (arg_values.tellp() > 0) {
        arg_values << ",";
      }
      arg_values << fmt::format(
          R"("{}"={})", kOutTensorsStart, output_tensor_starts);
    }
    const auto& inSplitSize = collectiveRecord->getMetadataValue(kInSplit);
    const auto& outSplitSize = collectiveRecord->getMetadataValue(kOutSplit);
    if (!inSplitSize.empty() && !outSplitSize.empty()) {
      if (arg_values.tellp() > 0) {
        arg_values << ",";
      }
      arg_values << fmt::format(
          R"("{}"={},"{}"={})", kInSplit, inSplitSize, kOutSplit, outSplitSize);
    }
    const auto& processGroupName =
        collectiveRecord->getMetadataValue(kProcessGroupName);
    if (!processGroupName.empty()) {
      if (arg_values.tellp() > 0) {
        arg_values << ",";
      }
      arg_values << fmt::format(
          R"("{}"={})", kProcessGroupName, processGroupName);
    }
    const auto& processGroupDesc =
        collectiveRecord->getMetadataValue(kProcessGroupDesc);
    if (!processGroupName.empty()) {
      if (arg_values.tellp() > 0) {
        arg_values << ",";
      }
      arg_values << fmt::format(
          R"("{}"={})", kProcessGroupDesc, processGroupDesc);
    }
    const auto& groupRanks = collectiveRecord->getMetadataValue(kGroupRanks);
    if (!groupRanks.empty()) {
      if (arg_values.tellp() > 0) {
        arg_values << ",";
      }
      arg_values << fmt::format(R"("{}"={})", kGroupRanks, groupRanks);
    }
    const auto& dstRank = collectiveRecord->getMetadataValue(kP2pDst);
    const auto& srcRank = collectiveRecord->getMetadataValue(kP2pSrc);
    if (!dstRank.empty()) {
      arg_values << fmt::format(R"(,"{}"={})", kP2pDst, dstRank);
    }
    if (!srcRank.empty()) {
      arg_values << fmt::format(R"(,"{}"={})", kP2pSrc, srcRank);
    }

    if (distInfo_.backend == "" && processGroupDesc == "\"default_pg\"") {
      distInfo_.backend = "nccl";
      distInfo_.rank = collectiveRecord->getMetadataValue(kRank);
      distInfo_.world_size = groupSize;
      distInfo_.nccl_version = "unknown";
    }
    auto pg_config = pgConfig();
    pg_config.pg_name = processGroupName;
    pg_config.pg_desc = processGroupDesc;
    pg_config.backend_config = "cuda:nccl";
    pg_config.pg_size = groupSize;
    pg_config.ranks = groupRanks;
    pgMap.insert({processGroupName, pg_config});
  }

  std::string args = arg_values.str();

  int device = op.deviceId();
  int resource = op.resourceId();
  std::string op_name = op.name() == "kernel" ? "Kernel" : op.name();
  sanitizeForNonReadableChars(op_name);

  ts = transToRelativeTime(ts);
  torch_op_event.ph = 'X';
  torch_op_event.cat = toString(op.type());
  torch_op_event.name = op_name;
  torch_op_event.pid = device;
  torch_op_event.tid = sanitizeTid(resource);
  torch_op_event.ts = static_cast<float>(ts) / 1000.0f;
  torch_op_event.dur = static_cast<float>(duration) / 1000.0f;
  torch_op_event.args = args;

  addKinetoTorchOpEvent(torch_op_event);
  if (op.flowId() > 0) {
    handleGenericLink(op);
  }
}

void OrcaTraceLogger::handleGenericLink(const ITraceActivity& act) {
  static struct {
    int type;
    char name[16];
  } flow_names[] = {{kLinkFwdBwd, "fwdbwd"}, {kLinkAsyncCpuGpu, "ac2g"}};
  for (auto& flow : flow_names) {
    if (act.flowType() == flow.type) {
      if (act.flowStart()) {
        handleLink(kFlowStart, act, act.flowId(), flow.name);
      } else {
        handleLink(kFlowEnd, act, act.flowId(), flow.name);
      }
      return;
    }
  }
  LOG(WARNING) << "Unknown flow type: " << act.flowType();
}

void OrcaTraceLogger::handleLink(
    char type,
    const ITraceActivity& e,
    int64_t id,
    const std::string& name) {
  uint64_t ts = transToRelativeTime(e.timestamp());
  std::string args =
      fmt::format(R"("id"={}{})", id, type == kFlowEnd ? R"(,"bp"=e)" : "");
  KinetoMiscEvent link_event{
      .base =
          EventBase{
              .timestep = timestep_,
              .rank = rank_,
          },
      .ph = type,
      .cat = name,
      .name = name,
      .pid = e.deviceId(),
      .tid = sanitizeTid(e.resourceId()),
      .ts = static_cast<float>(ts) / 1000.0f,
      .args = args,
  };
  addKinetoMiscEvent(link_event);
}

void OrcaTraceLogger::addOnDemandDistMetadata() {
  if (distInfo_.backend == "") {
    return;
  }
  KinetoMetadataEvent dist_metadata_event{
      .base =
          EventBase{
              .timestep = timestep_,
              .rank = rank_,
          },
  };
  std::string distinfo = fmt::format(
      R"("backend"={},"rank"={},"world_size"={},"pg_count"={},"pg_config"=[)",
      distInfo_.backend,
      distInfo_.rank,
      distInfo_.world_size,
      std::to_string(pgMap.size()));
  for (const auto& element : pgMap) {
    distinfo += fmt::format(
        R"("pg_name"={},"pg_desc"={},"backend_config"={},"pg_size"={},"ranks"={})",
        element.second.pg_name,
        element.second.pg_desc,
        element.second.backend_config,
        element.second.pg_size,
        element.second.ranks);
  }
  distinfo += fmt::format(R"(],"nccl_version"={})", distInfo_.nccl_version);

  dist_metadata_event.key = "distributedInfo";
  dist_metadata_event.value = distinfo;
  addKinetoMetadataEvent(dist_metadata_event);

  distInfo_.distInfo_present_ = true;
}

void OrcaTraceLogger::finalizeTrace(
    const Config& config,
    std::unique_ptr<ActivityBuffers> buffers,
    int64_t endTime,
    std::unordered_map<std::string, std::vector<std::string>>& metadata) {
  endTime = transToRelativeTime(endTime);
  std::string args = fmt::format(R"("s"=g)");
  KinetoMiscEvent end_event{
      .base =
          EventBase{
              .timestep = timestep_,
              .rank = rank_,
          },
      .ph = 'i',
      .cat = "",
      .name = "Record Window End",
      .pid = -1,
      .tid = -1,
      .ts = static_cast<float>(endTime) / 1000.0f,
      .args = args,
  };
  addKinetoMiscEvent(end_event);

  if (!distInfo_.distInfo_present_) {
    addOnDemandDistMetadata();
  }

  KinetoMetadataEvent end_metadata_event{
      .base =
          EventBase{
              .timestep = timestep_,
              .rank = rank_,
          },
  };

#if !USE_GOOGLE_LOG
  std::unordered_map<std::string, std::string> prepared_metadata;
  for (const auto& kv : metadata) {
    if (!kv.second.empty()) {
      std::string value = "[";
      int mdv_count = kv.second.size();
      for (auto v : kv.second) {
        value.append(fmt::format(R"("{}")", v));
        if (mdv_count > 1) {
          value.append(",");
          mdv_count--;
        }
      }
      value.append("]");
      prepared_metadata[kv.first] = value;
    }
  }

  for (const auto& [k, v] : prepared_metadata) {
    if (k == "distributedInfo") {
      distInfo_.distInfo_present_ = true;
    }
    end_metadata_event.key = k;
    end_metadata_event.value = v;
    addKinetoMetadataEvent(end_metadata_event);
  }
#endif

  end_metadata_event.key = "displayTimeUnit";
  end_metadata_event.value = "ms";
  addKinetoMetadataEvent(end_metadata_event);

  end_metadata_event.key = "baseTimeNanoseconds";
  end_metadata_event.value =
      std::to_string(ChromeTraceBaseTime::singleton().get());
  addKinetoMetadataEvent(end_metadata_event);
}

} // namespace KINETO_NAMESPACE