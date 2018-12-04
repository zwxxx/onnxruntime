// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/rewrite_rule.h"

namespace onnxruntime {

// Rewrite rule that eliminates a slice operator if it redundant (does not cause data reduction).
class EliminateSlice : public RewriteRule {
 public:
  EliminateSlice() noexcept : RewriteRule("EliminateSlice", "Eliminate slice node") {}

 private:
  bool SatisfyCondition(const GraphEditor& graph_editor, const Node& node) override;

  Status Apply(GraphEditor& graph_editor, Node& node, bool& modified) override;
};

}  // namespace onnxruntime
