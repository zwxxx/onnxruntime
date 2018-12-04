// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/rewrite_rule.h"
#include "core/graph/slice_elimination.h"
#include "core/graph/graph.h"
#include "core/graph/op.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {

Status EliminateSlice::Apply(GraphEditor& graph_editor, Node& node, bool& modified) {
  if (graph_editor.RemoveSingleInSingleOutNode(node.Index())) {
    modified = true;
  }

  return Status::OK();
}

bool EliminateSlice::SatisfyCondition(const GraphEditor& graph_editor, const Node& node) {
  // At the moment, we eliminate a slice operator only if it has a single input and a single output.
  if (!graph_editor.IsSingleInSingleOutNode(node)) {
    return false;
  }

  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
  if (!graph_editor.GetRepeatedNodeAttributeValues(node, "starts", starts) ||
      !graph_editor.GetRepeatedNodeAttributeValues(node, "ends", ends) ||
      starts.size() != ends.size()) {
    return false;
  }
  std::vector<int64_t> axes;
  if (!graph_editor.GetRepeatedNodeAttributeValues(node, "axes", axes)) {
    for (int i = 0; (size_t)i < starts.size(); ++i) {
      axes.push_back(i);
    }
  } else if (axes.size() != starts.size() || axes.size() != ends.size()) {
    return false;
  }

  // For now eliminate slice operators if starts=0 and ends=MAX_INT or -1.
  // TODO: Take into account the input's shape to get a tighter bound for the ends.
  for (int i = 0; i < axes.size(); ++i) {
    if (starts[i] > 0 || starts[i] < 0 ||
        (ends[i] > 0 && ends[i] < INT64_MAX)) {
      return false;
    }
  }

  return true;
}

}  // namespace onnxruntime
