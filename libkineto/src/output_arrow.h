#pragma once

#include "ActivityBuffers.h"
#include "GenericTraceActivity.h"
#include "output_base.h"

#include <arrow/api.h>
#include <arrow/scalar.h>
#include <arrow/type_fwd.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>

namespace KINETO_NAMESPACE {

struct ArrowStats {
  std::vector<size_t> num_rows;
  std::vector<size_t> bytes;
  // Durations of logging in-memory trace records into arrow record batches,
  // including both preprocessing cpu and gpu traces and appending them
  // to the record batch plus generating the complete record batch.
  std::vector<uint64_t> logging_durations;


  // Used to measure the end-to-end rates between flushing starts and ends.
  // This is much lower than the rates the writing arrow record batches can
  // achieve because it contains the training durations too.
  // Therefore, the end-to-end rates should be treated as a reflection of
  // how fast the pytorch program is generating traces, rather than the
  // maximum rates arrow logger can achieve.
  int64_t start_time;
  int64_t end_time;

  std::shared_mutex rw_mutex;
};

class ArrowTraceLogger : public ActivityLogger {
 public:
  explicit ArrowTraceLogger(
      const std::string& arrowTableName,
      ArrowStats* arrowStats);

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

  void setStartTime(int64_t startTime) override {
    startTime_ = startTime;
  }

 private:
  struct ActivityArrowFields {
    std::string cat;
    std::string name;
    int64_t pid;
    int64_t tid;
    int64_t ts;
    int64_t dur;
    int64_t end;
    ActivityArrowMetadata metadata;
  };

  arrow::Status appendActivity(const ActivityArrowFields& activityAttributes);

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> buildArrowTable();

  std::string arrowTableName_;
  std::shared_ptr<arrow::Schema> schema_;
  ArrowStats* arrowStats_ = nullptr;
  int64_t startTime_ = 0;
  int rank_ = -1;

  // Column builders
  arrow::Int32Builder rankBuilder_;
  arrow::StringBuilder catBuilder_;
  arrow::StringBuilder nameBuilder_;
  arrow::Int64Builder pidBuilder_;
  arrow::Int64Builder tidBuilder_;
  arrow::Int64Builder tsBuilder_;
  arrow::Int64Builder durBuilder_;
  arrow::Int64Builder endBuilder_;
  arrow::Int64Builder streamBuilder_;
  arrow::Int64Builder correlationBuilder_; // TODO: may not need
  arrow::Int64Builder bytesBuilder_; // TODO: may not need
  arrow::DoubleBuilder memBwBuilder_; // TODO: may not need
  arrow::Int64Builder waitOnStreamBuilder_; // TODO: may not need
  arrow::Int64Builder waitOnCudaEventBuilder_; // TODO: may not need
};

} // namespace KINETO_NAMESPACE