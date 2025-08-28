#pragma once

#include "ActivityBuffers.h"
#include "GenericTraceActivity.h"
#include "output_base.h"

#include <arrow/api.h>
#include <arrow/type_fwd.h>
#include <memory>

namespace KINETO_NAMESPACE {

class ArrowTraceLogger : public ActivityLogger {
 public:
  explicit ArrowTraceLogger(const std::string& arrowTableName);

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

  // Column builders
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