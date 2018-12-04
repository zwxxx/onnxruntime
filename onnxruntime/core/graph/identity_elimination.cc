// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/rewrite_rule.h"
#include "core/graph/identity_elimination.h"
#include "core/graph/graph.h"
#include "core/graph/op.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {

Status EliminateIdentity::Apply(GraphEditor& graph_editor, Node& node, bool& modified) {
  if (graph_editor.RemoveSingleInSingleOutNode(node.Index())) {
    modified = true;
  }

  return Status::OK();
}

bool EliminateIdentity::SatisfyCondition(const GraphEditor& graph_editor, const Node& node) {
  return graph_editor.IsSingleInSingleOutNode(node);
}

}  // namespace onnxruntime
