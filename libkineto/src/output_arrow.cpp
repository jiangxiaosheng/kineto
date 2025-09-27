#include "output_arrow.h"
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

ArrowTraceLogger::ArrowTraceLogger(
    const std::string& arrowTableName,
    ArrowStats* arrowStats)
    : arrowTableName_(arrowTableName), arrowStats_(arrowStats) {
  std::vector<std::shared_ptr<arrow::Field>> fields = {
      arrow::field("rank", arrow::int32()),
      arrow::field("cat", arrow::utf8()),
      arrow::field("name", arrow::utf8()),
      arrow::field("pid", arrow::int64()),
      arrow::field("tid", arrow::int64()),
      arrow::field("ts", arrow::int64()),
      arrow::field("dur", arrow::int64()),
      arrow::field("end", arrow::int64()),
      arrow::field("stream", arrow::int64()),
      arrow::field("correlation", arrow::int64()),
      arrow::field("bytes", arrow::int64()),
      arrow::field("memory_bw_gbps", arrow::float64()),
      arrow::field("wait_on_stream", arrow::int64()),
      arrow::field("wait_on_cuda_event_record_corr_id", arrow::int64()),
  };
  schema_ = std::make_shared<arrow::Schema>(fields);
  // If run by torchrun, the rank is set in the environment variable.
  const char* rank = getenv("RANK");
  if (rank) {
    rank_ = std::stoi(rank);
  }
}

// 'dur' = 0, do nothing
void ArrowTraceLogger::handleDeviceInfo(const DeviceInfo& info, uint64_t time) {
}

// 'dur' = 0, do nothing
void ArrowTraceLogger::handleOverheadInfo(
    const OverheadInfo& info,
    int64_t time) {}

// 'cat' = 'Trace' or 'dur' = 0, do nothing
void ArrowTraceLogger::handleTraceSpan(const TraceSpan& span) {}

// 'cat' = 'Trace' or 'dur' = 0, do nothing
void ArrowTraceLogger::handleResourceInfo(
    const ResourceInfo& info,
    int64_t time) {}

// 'cat' = 'Trace' or 'dur' = 0, do nothing
void ArrowTraceLogger::handleGenericActivity(
    const GenericTraceActivity& activity) {
  handleActivity(activity);
}

// In json this writes metadata as the header, but it might be unneeded for HTA.
void ArrowTraceLogger::handleTraceStart(
    const std::unordered_map<std::string, std::string>& metadata,
    const std::string& device_properties) {}

// 'dur' = 0, do nothing
void ArrowTraceLogger::handleActivity(const ITraceActivity& op) {
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

  ActivityArrowFields fields{
      .cat = cat,
      .name = op_name,
      .pid = 0,
      .tid = resource,
      .ts = ts,
      .dur = duration,
      .end = end,
      .metadata = op.getArrowMetadata(),
  };

  auto status = appendActivity(fields);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to append activity: " << status.ToString();
  }
}

arrow::Status ArrowTraceLogger::appendActivity(
    const ActivityArrowFields& fields) {
  ARROW_RETURN_NOT_OK(rankBuilder_.Append(rank_));
  ARROW_RETURN_NOT_OK(catBuilder_.Append(fields.cat));
  ARROW_RETURN_NOT_OK(nameBuilder_.Append(fields.name));
  ARROW_RETURN_NOT_OK(pidBuilder_.Append(fields.pid));
  ARROW_RETURN_NOT_OK(tidBuilder_.Append(fields.tid));
  ARROW_RETURN_NOT_OK(tsBuilder_.Append(fields.ts));
  ARROW_RETURN_NOT_OK(durBuilder_.Append(fields.dur));
  ARROW_RETURN_NOT_OK(endBuilder_.Append(fields.end));
  ARROW_RETURN_NOT_OK(streamBuilder_.Append(fields.metadata.stream));
  ARROW_RETURN_NOT_OK(correlationBuilder_.Append(fields.metadata.correlation));
  ARROW_RETURN_NOT_OK(bytesBuilder_.Append(fields.metadata.bytes));
  ARROW_RETURN_NOT_OK(memBwBuilder_.Append(fields.metadata.memBw));
  ARROW_RETURN_NOT_OK(
      waitOnStreamBuilder_.Append(fields.metadata.waitOnStream));
  ARROW_RETURN_NOT_OK(
      waitOnCudaEventBuilder_.Append(fields.metadata.waitOnCudaEvent));

  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ArrowTraceLogger::
    buildArrowTable() {
  std::shared_ptr<arrow::Array> rank_array;
  ARROW_RETURN_NOT_OK(rankBuilder_.Finish(&rank_array));
  std::shared_ptr<arrow::Array> cat_array;
  ARROW_RETURN_NOT_OK(catBuilder_.Finish(&cat_array));
  std::shared_ptr<arrow::Array> name_array;
  ARROW_RETURN_NOT_OK(nameBuilder_.Finish(&name_array));
  std::shared_ptr<arrow::Array> pid_array;
  ARROW_RETURN_NOT_OK(pidBuilder_.Finish(&pid_array));
  std::shared_ptr<arrow::Array> tid_array;
  ARROW_RETURN_NOT_OK(tidBuilder_.Finish(&tid_array));
  std::shared_ptr<arrow::Array> ts_array;
  ARROW_RETURN_NOT_OK(tsBuilder_.Finish(&ts_array));
  std::shared_ptr<arrow::Array> dur_array;
  ARROW_RETURN_NOT_OK(durBuilder_.Finish(&dur_array));
  std::shared_ptr<arrow::Array> end_array;
  ARROW_RETURN_NOT_OK(endBuilder_.Finish(&end_array));
  std::shared_ptr<arrow::Array> stream_array;
  ARROW_RETURN_NOT_OK(streamBuilder_.Finish(&stream_array));
  std::shared_ptr<arrow::Array> correlation_array;
  ARROW_RETURN_NOT_OK(correlationBuilder_.Finish(&correlation_array));
  std::shared_ptr<arrow::Array> bytes_array;
  ARROW_RETURN_NOT_OK(bytesBuilder_.Finish(&bytes_array));
  std::shared_ptr<arrow::Array> memBw_array;
  ARROW_RETURN_NOT_OK(memBwBuilder_.Finish(&memBw_array));
  std::shared_ptr<arrow::Array> waitOnStream_array;
  ARROW_RETURN_NOT_OK(waitOnStreamBuilder_.Finish(&waitOnStream_array));
  std::shared_ptr<arrow::Array> waitOnCudaEvent_array;
  ARROW_RETURN_NOT_OK(waitOnCudaEventBuilder_.Finish(&waitOnCudaEvent_array));

  std::vector<std::shared_ptr<arrow::Array>> columns = {
      rank_array,
      cat_array,
      name_array,
      pid_array,
      tid_array,
      ts_array,
      dur_array,
      end_array,
      stream_array,
      correlation_array,
      bytes_array,
      memBw_array,
      waitOnStream_array,
      waitOnCudaEvent_array};
  auto num_rows = rank_array->length();
  bool ok = std::all_of(
      columns.begin(),
      columns.end(),
      [num_rows](const std::shared_ptr<arrow::Array>& array) {
        return array->length() == num_rows;
      });
  if (!ok) {
    return arrow::Status::Invalid(fmt::format(
        "All columns must have the same length, {} != {}",
        num_rows,
        columns[0]->length()));
  }
  auto rb = arrow::RecordBatch::Make(schema_, cat_array->length(), columns);
  return rb;
}

void ArrowTraceLogger::finalizeTrace(
    const Config& config,
    std::unique_ptr<ActivityBuffers> buffers,
    int64_t endTime,
    std::unordered_map<std::string, std::vector<std::string>>& metadata) {
  auto rb = buildArrowTable().ValueOrDie();
  int64_t ipcsz = 0;
  auto szstatus = arrow::ipc::GetRecordBatchSize(*rb, &ipcsz);
  if (!szstatus.ok()) {
    LOG(ERROR) << "Failed to get record batch size";
    return;
  }
  {
    std::lock_guard guard(arrowStats_->rw_mutex);
    arrowStats_->num_rows.push_back(rb->num_rows());
    arrowStats_->bytes.push_back(ipcsz);
    arrowStats_->logging_durations.push_back(
        timeSinceEpoch(std::chrono::system_clock::now()) - startTime_);
  }
  // FIXME: Write to parquet just for testing. Should send out this arrow table
  // via RPC.
  std::shared_ptr<arrow::io::FileOutputStream> outfile;
  outfile = arrow::io::FileOutputStream::Open(arrowTableName_).ValueOrDie();
  auto arrow_writer = parquet::arrow::FileWriter::Open(
                          *schema_, arrow::default_memory_pool(), outfile)
                          .ValueOrDie();
  auto status = arrow_writer->WriteRecordBatch(*rb);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to write record batch";
  }

  status = arrow_writer->Close();
  if (!status.ok()) {
    LOG(ERROR) << "Failed to close arrow writer";
  }
}

} // namespace KINETO_NAMESPACE