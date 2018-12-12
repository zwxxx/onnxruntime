// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/identity_elimination.h"

namespace onnxruntime {

Status EliminateIdentity::Apply(Graph& graph, Node& node, bool& modified) {
  std::map<const NodeArg*, NodeArg*> replacement_defs;
  auto id_input = node.InputDefs()[0];
  auto id_output = node.OutputDefs()[0];
  replacement_defs[id_output] = const_cast<NodeArg*>(id_input);

  // Replace (input) defs of the nodes following the Identity with the input to the Identity.
  for (auto it = node.OutputNodesBegin(), end = node.OutputNodesEnd(); it != end; ++it) {
    // TODO: Fix the Node API so this operation is supported without resorting to const_cast.
    const_cast<Node*>(&*it)->ReplaceDefs(replacement_defs);
    modified = true;
  }

  // Remove the Identity node.
  graph.RemoveNode(node.Index());

  // TODO: Make sure resolve is not required here.
  //ONNXRUNTIME_RETURN_IF_ERROR(graph_editor->Resolve());

  return Status::OK();
}

bool EliminateIdentity::SatisfyCondition(const Graph& /*graph*/, const Node& /*node*/) {
  return true;  // No additional condition required.
}

}  // namespace onnxruntime
