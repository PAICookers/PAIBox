# Backendv2 TensorLayout Compatibility

## Workspace Decision

- [x] Stay in the current worktree `PAIBox-codex-avgpool-binary-majority` on branch `feat-paiir-avgpool-binary-majority`.
- [x] Do not create another dedicated `git worktree`; this is a focused TensorLayout-compatibility fix in an already isolated worktree.
- [x] Keep the implementation scoped to `paibox/backendv2/**`, the TensorLayout-affected `tests/backendv2/**` regressions, and `tasks/**`.

## Ownership

- Owner: Codex
- Branch: `feat-paiir-avgpool-binary-majority`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority`
- Allowed Files: `paibox/backendv2/**`, `tests/backendv2/test_reorder_node.py`, `tests/backendv2/test_routing_v2_raw_weights.py`, `tasks/**`
- Blocked Files: in-flight `paibox/paiir/pipeline/**` avgpool rewrite files, lockfiles, unrelated dirty files
- Dependencies: current `TensorLayout` layout metadata model (`input_layouts` / `output_layouts`), backend reshape semantics, and the affected `backendv2` regressions
- Verification: focused `py_compile` plus the TensorLayout-affected `tests/backendv2` regressions only

## Interface Notes

- `backendv2` code and tests should use the current layout metadata interface:
  - `InputNode.shape` / `OutputNode.shape` for graph boundaries
  - `OpNode.input_layouts` / `OpNode.output_layouts` for operator layout metadata
  - do not rely on ad hoc legacy attributes such as `output_shape` / `input_dims`
- Scope guard:
  - do not adapt unrelated backend allocation tests in this task
  - do not widen the public backendv2 package surface here beyond the already-requested `Mapper` re-export

## Plan

- [x] Record the current TensorLayout baseline and identify the stale `backendv2` / test call sites.
- [x] Update `backendv2` reshape/reorder handling to respect the current layout metadata interface.
- [x] Adapt only the TensorLayout-affected `backendv2` regressions to the current layout metadata setup.
- [x] Run focused verification and document the results below.

## Review

# AvgPool Binary Majority Worktree Review

## Workspace Decision

- [x] Stay in the current worktree `PAIBox-codex-avgpool-binary-majority`.
- [x] Do not create another dedicated `git worktree`; this task is a read-only review of the in-flight changes already isolated here.
- [x] Keep the review scoped to the current uncommitted worktree diff plus `tasks/**`.

## Ownership

- Owner: Codex
- Branch: `feat-paiir-avgpool-binary-majority`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority`
- Allowed Files: current uncommitted diff for this worktree, `tests/**` that verify it, `tasks/**`
- Blocked Files: unrelated repository-root dirty files outside this worktree, third-party dependencies, lockfiles
- Dependencies: current compile pipeline entrypoints, avgpool rewrite phase wiring, and the new rewrite tests
- Verification: diff inspection plus focused test/usage-path review; run targeted commands only if needed to confirm a suspected issue

## Plan

- [x] Identify the exact modified and newly added files in this worktree.
- [x] Review the avgpool rewrite and compile-path changes for behavioral regressions and contract mismatches.
- [x] Review related tests for coverage gaps and confirm whether they would catch the suspected regressions.
- [x] Record findings and residual risks below.

## Review

- Findings:
  - `standalone_rewrite.py` now rejects every `InputNode -> AvgPool` path up front at [standalone_rewrite.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/paiir/pipeline/avgpool/standalone_rewrite.py#L75) and that rewrite is now always executed from [compile.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/paiir/pipeline/compile.py#L235). This regresses an existing repo contract in [test_compile.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/tests/paiir/pipeline/test_compile.py#L526), and the newly added [test_standalone_avgpool_rewrite.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/tests/paiir/pipeline/test_standalone_avgpool_rewrite.py#L81) codifies the opposite behavior.
  - `_resolve_effective_source_info(...)` only recognizes `SequentialOp` and `StandaloneActOp` as mode-carrying upstream producers at [standalone_rewrite.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/paiir/pipeline/avgpool/standalone_rewrite.py#L127), so valid activated offline-core producers such as [AccumulateOp](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/paiir/ir/op_node.py#L319) are misclassified as having no mode and fail compilation when followed by standalone AvgPool.
- Verification:
  - `../PAIBox/.venv/bin/pytest tests/paiir/pipeline/test_standalone_avgpool_rewrite.py -q`
  - `../PAIBox/.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k padding_free_count_include_pad_false_still_compiles`
  - local repro: `ResidualAddReluPool` (`conv + conv -> relu -> avgpool`) via `compile_to_paiir(...)`
- Result:
  - the new standalone-rewrite tests pass
  - the existing compile regression test fails with `UnsupportedFusionError` for direct-input AvgPool
  - the local residual-add repro also fails with `UnsupportedFusionError` (`has no effective upstream producer mode`)

# Standalone AvgPool Regression Fix

## Workspace Decision

- [x] Stay in the current worktree `PAIBox-codex-avgpool-binary-majority`.
- [x] Do not create another dedicated `git worktree`; this is a direct continuation of the in-flight standalone AvgPool work already isolated here.
- [x] Keep the implementation scoped to `paibox/paiir/pipeline/**`, any touched IR helpers, relevant `tests/paiir/**`, and `tasks/**`.

## Ownership

- Owner: Codex
- Branch: `feat-paiir-avgpool-binary-majority`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority`
- Allowed Files: `paibox/paiir/pipeline/**`, related `paibox/paiir/ir/**` helper files if required, `tests/paiir/**`, `tasks/**`
- Blocked Files: `backendv2/**`, lockfiles, unrelated docs and external projects
- Dependencies: current compile pipeline ordering, data-format propagation semantics, offline-core mode modeling, and existing AvgPool compile tests
- Verification: focused `pytest` on compile + standalone-avgpool regressions plus targeted local repros

## Plan

- [x] Reconstruct the intended standalone AvgPool semantics from the existing pipeline and tests, including direct-input and activated-offline-op predecessors.
- [x] Implement the root-cause fix in the rewrite logic and supporting helpers/tests.
- [x] Run targeted verification and record the result below.

## Review

- Reframed standalone AvgPool handling in [standalone_rewrite.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/paiir/pipeline/avgpool/standalone_rewrite.py) as an opportunistic post-analysis rewrite instead of a hard validation gate:
  - when the graph has one effective upstream value-source mode and the input format matches a safe exact rewrite, the node is rewritten
  - otherwise the original standalone `AvgPool` node is left unchanged and compilation continues
- Root-cause changes in [standalone_rewrite.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/paiir/pipeline/avgpool/standalone_rewrite.py):
  - removed the aggregate error path that previously rejected direct-input, mixed-mode, and no-mode standalone AvgPool cases
  - changed effective-source discovery to use propagated graph semantics:
    - transparent routing and standalone `MaxPool` remain passthrough
    - `InputNode` now contributes “unknown source mode”, which prevents rewrite but no longer fails compilation
    - value-producing offline-core ops with real activation semantics now contribute their `core_params.snn_mode`, including `AccumulateOp`
  - binary-majority rewrite now checks `divisor == window_size` before rewriting and falls back to the original standalone AvgPool when that exact rewrite is not available
- Expanded [test_standalone_avgpool_rewrite.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/tests/paiir/pipeline/test_standalone_avgpool_rewrite.py):
  - direct-input standalone AvgPool now verifies compile success plus no rewrite
  - `Conv -> AvgPool` potential-domain predecessor now verifies no rewrite
  - `Conv + Conv -> ReLU -> AvgPool` now verifies exact ANN rewrite through an `AccumulateOp` producer
  - spike `divisor_override` and mixed ANN/SNN producer cases now verify compile success plus no rewrite, instead of expecting `UnsupportedFusionError`
- Verification:
  - `../PAIBox/.venv/bin/python -m py_compile paibox/paiir/pipeline/avgpool/standalone_rewrite.py tests/paiir/pipeline/test_standalone_avgpool_rewrite.py`
  - `env COVERAGE_FILE=/tmp/standalone_avgpool_fix.coverage ../PAIBox/.venv/bin/pytest tests/paiir/pipeline/test_rewrite_phase.py tests/paiir/pipeline/test_standalone_avgpool_rewrite.py -q`
  - `env COVERAGE_FILE=/tmp/avgpool_compile_fix.coverage ../PAIBox/.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k 'avgpool or maxpool'`
  - `env COVERAGE_FILE=/tmp/avgpool_passes_fix.coverage ../PAIBox/.venv/bin/pytest tests/paiir/pipeline/test_passes.py -q -k 'maxpool or avgpool'`
  - local compile repro for `conv + conv -> relu -> avgpool`
- Result:
  - targeted standalone rewrite tests: `11 passed`
  - compile avgpool/maxpool slice: `48 passed, 38 deselected`
  - pass avgpool/maxpool slice: `3 passed, 74 deselected`
  - local residual-add repro now compiles and rewrites to `SequentialOp(SumPool2d, ANNNodeV25)`

# DVSGesture Standalone AvgPool Verification

## Workspace Decision

- [x] Keep the implementation work in the isolated worktree `PAIBox-codex-avgpool-binary-majority`.
- [x] Use the root-repo `tests/user/test_dvsgesture.py` only as a test asset and log generator, while forcing imports to resolve to the worktree `paibox` package.
- [x] Allow one minimal backendv2 syntax fix in the worktree because `test_dvsgesture.py` imports `Mapper` at module import time and the accidental parse error blocked deployment verification entirely.

## Plan

- [x] Inspect the DVSGesture deploy test entrypoint and existing debug logs.
- [x] Unblock import of `Mapper` in the worktree.
- [x] Re-run the DVSGesture deploy flow with the worktree compiler and inspect the generated logs for standalone AvgPool handling.

## Review

- Fixed the accidental leading indentation in [coreplacement.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/backendv2/coreplacement.py) so `paibox.backendv2.mapper.Mapper` can be imported again for deployment verification.
- Reused the network and helper functions from [test_dvsgesture.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/test_dvsgesture.py) under a mixed import path:
  - worktree `paibox/**` implementation
  - root-repo `tests/user/test_dvsgesture.py` test asset and log paths
- Wrote fresh deployment logs to:
  - [paiir_summary.log](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/debug/dvsgesture_deploy/paiir_summary.log)
  - [backendv2.log](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/debug/dvsgesture_deploy/backendv2.log)
- Verified from [paiir_summary.log](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/debug/dvsgesture_deploy/paiir_summary.log) that the previously isolated pooling nodes are no longer emitted as standalone `AvgPool` ops:
  - the five `AvgPool2d` stages now appear as `SequentialOp_* (SumPool2d -> IFNodeV25)`
  - the voting `AvgPool1d` stage now appears as `SequentialOp_12 (SumPool1d -> IFNodeV25)`
  - there are no `StandaloneCompOp_* (AvgPool2d)` entries left in the summary log
- Verified from [backendv2.log](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/debug/dvsgesture_deploy/backendv2.log) that the rewritten stages stay in the 1-bit spike path instead of inflating to the old 32-bit standalone-potential path:
  - `SequentialOp_7` through `SequentialOp_12` all show `Input bit num: 1, Output bit num: 1`
  - this is the expected post-rewrite signature for binary-majority handling
- Residual backend gap exposed by the deploy run:
  - channels=8 now fails later with `AssertionError: Only convolution groups with Conv2d component are supported for tiling.`
  - channels=4 now fails later with `NotImplementedError: Unsupported weight expansion for comp <class 'paibox.paiir.nn.pool.SumPool2d'> with weight <class 'NoneType'>.`
  - these failures happen after the standalone AvgPool rewrite and indicate backendv2 does not yet fully accept the new `SumPool`-based representation end-to-end
- Verification:
  - `../PAIBox/.venv/bin/python -m py_compile paibox/backendv2/coreplacement.py`
  - custom import/run harness using `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/test_dvsgesture.py` with `PYTHONPATH` preferring the worktree package
- Result:
  - frontend/compiler handling of standalone AvgPool is verified on the DVSGesture network
  - end-to-end backend deployment is not yet complete for the rewritten `SumPool` form

# Backendv2 Typing Modernization

## Workspace Decision

- [x] Stay in the current worktree `PAIBox-codex-avgpool-binary-majority`.
- [x] Do not create another `git worktree`; this is a syntax-only cleanup in `backendv2`.
- [x] Keep the implementation scoped to `paibox/backendv2/**` and `tasks/**`.

## Ownership

- Owner: Codex
- Branch: `feat-paiir-avgpool-binary-majority`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority`
- Allowed Files: `paibox/backendv2/**`, `tasks/**`
- Blocked Files: tests and pipeline files unless verification requires them, unrelated dirty files
- Dependencies: project runtime is Python `>=3.10`, so built-in generics and `|` unions are available
- Verification: `py_compile` for `backendv2/**` plus one focused backendv2 regression slice

## Interface Notes

- Replace deprecated-style typing aliases where straightforward:
  - `Optional[T]` -> `T | None`
  - `List[T]` -> `list[T]`
  - `Tuple[...]` -> `tuple[...]`
  - `Union[A, B]` -> `A | B`
  - `Sequence` / `AbstractSet` should come from `collections.abc`
- Keep `Generic`, `TypeVar`, and `TextIO` where they are still the right tool.
- This task is syntax-only; no behavior or API changes are intended.

## Plan

- [x] Replace deprecated-style typing aliases in `backendv2` modules with Python 3.10+ syntax.
- [x] Run focused verification and record the result below.

## Review

- Updated the `backendv2` modules that were still importing deprecated-style typing aliases from `typing`:
  - [op_node.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/backendv2/op_node.py)
  - [routing.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/backendv2/routing.py)
  - [get_weight.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/backendv2/get_weight.py)
  - [weight.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/backendv2/weight.py)
  - [neuron.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/backendv2/neuron.py)
  - [coreplacement.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/backendv2/coreplacement.py)
  - [core_config.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/backendv2/core_config.py)
- Replacements made:
  - `Optional[T]` -> `T | None`
  - `List[T]` -> `list[T]`
  - `Tuple[...]` -> `tuple[...]`
  - `Union[...]` -> `|`
  - `Sequence` / `AbstractSet`-style abstractions moved off `typing`; `routing.py` now uses `collections.abc.Sequence` and `collections.abc.Set`
- Kept `Generic`, `TypeVar`, and `TextIO` unchanged because they are still the appropriate modern typing APIs here.
- Verification:
  - `../PAIBox/.venv/bin/python -m py_compile paibox/backendv2/*.py`
  - `../PAIBox/.venv/bin/python -c "import paibox.backendv2.op_node, paibox.backendv2.routing, paibox.backendv2.get_weight, paibox.backendv2.coreplacement, paibox.backendv2.core_config, paibox.backendv2.neuron, paibox.backendv2.weight; print('backendv2_imports_ok')"`
  - Result: both passed.
- Additional regression attempt:
  - `env COVERAGE_FILE=/tmp/backendv2_typing.coverage ../PAIBox/.venv/bin/pytest tests/backendv2/test_reorder_node.py tests/backendv2/test_routing_v2_raw_weights.py -q`
  - Result: blocked during collection by the existing import in [test_routing_v2_raw_weights.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/tests/backendv2/test_routing_v2_raw_weights.py), which still imports `InputElem` / `Neuron` from `paibox.backendv2.neuron`. That blocker is unrelated to the typing-syntax migration itself, so I left it unchanged in this task.

# PAIIR Single SplitOp Through-Compile Refactor

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Do not create a dedicated `git worktree` for this task because it is a direct follow-up on the current in-flight split lowering work already present in this workspace.
- [x] Keep this task scoped to `paibox/paiir/ir/**`, `paibox/paiir/lowering/**`, `paibox/paiir/pipeline/**`, relevant `tests/paiir/**`, and `tasks/**`.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `paibox/paiir/ir/op_node.py`, `paibox/paiir/ir/graph.py`, `paibox/paiir/lowering/**`, `paibox/paiir/pipeline/passes.py`, relevant `tests/paiir/**`, `tasks/**`
- Blocked Files: backend implementation, lockfiles, unrelated onboard/customer assets, unrelated dirty files
- Dependencies: current single-output graph assumptions in `PAIIRGraph`, current split lowering context/wiring, split-related pass validation, and existing split regression tests
- Verification: focused split lowering/simulation/pass/compile tests, plus one direct-output split case

## Interface Notes

- Keep `Edge` unchanged; do not add `src_port`.
- Represent one FX split producer as one `SplitOp` node in `PAIIRGraph`.
- Record branch selection inside `SplitOp` using `(successor_name, dst_port) -> output_index`.
- Scope is through-compile only: backend-ready validation must still reject `SplitOp`.

## Plan

- [x] Extend `SplitOp` to store full split-output metadata and per-successor branch mapping.
- [x] Change split lowering so the producer materializes one `SplitOp` and `getitem` stays bypass-only.
- [x] Teach graph simulation and compile validation to understand the single-node multi-output `SplitOp` special case.
- [x] Update split-related tests, including direct split-to-output coverage.
- [x] Run focused verification and document the results below.

## Review

- `SplitOp` is now the single graph-internal representation of one FX split producer.
- IR changes:
  - `SplitOp` no longer stores one `output_index`
  - it now stores `output_shapes` for the full split contract
  - it records consumer branch selection in `successor_output_index[(successor_name, dst_port)]`
  - `SplitOp.forward()` now returns the full split tuple
- Lowering changes:
  - the FX split producer materializes one `SplitOp`
  - split `getitem` nodes are bypass-only selectors
  - graph wiring records branch-to-successor mapping on the single `SplitOp`
  - added backward-compatible aliases `_propagate_shapes` / `_propagate_dims` in `converter.py` so existing internal test helpers keep working
- Graph simulation changes:
  - `PAIIRGraph.step()` now accepts tuple-valued internal node outputs for `SplitOp`
  - successor edge resolution selects the correct split branch using `successor_output_index`
  - `OutputNode` pass-through uses the same branch resolution path
- Compile-pass changes:
  - `SplitOp` remains excluded from the blanket single-`output_shape` requirement
  - `_validate_split_contract(...)` now validates `output_shapes` and successor branch mappings
  - predecessor-shape validation for downstream `ConcatOp` / `ReshapeOp` now resolves branch-specific shapes when the predecessor is `SplitOp`
  - backend-ready validation still rejects `SplitOp` unchanged
- Test updates:
  - lowering tests now expect one `SplitOp` for one split producer
  - added direct split-to-output coverage
  - simulation tests now cover direct split outputs with one `SplitOp`
  - pass tests now validate the new `output_shapes` / `successor_output_index` contract
- Verification:
  - `./.venv/bin/python -m py_compile paibox/paiir/ir/op_node.py paibox/paiir/ir/graph.py paibox/paiir/lowering/converter.py paibox/paiir/lowering/split_lowering.py paibox/paiir/pipeline/passes.py`
  - `env COVERAGE_FILE=/tmp/paiir_single_split_lowering2.coverage ./.venv/bin/pytest tests/paiir/lowering/test_converter.py -q -k split`
  - `env COVERAGE_FILE=/tmp/paiir_single_split_sim2.coverage ./.venv/bin/pytest tests/paiir/pipeline/test_graph_simulation.py -q -k split`
  - `env COVERAGE_FILE=/tmp/paiir_single_split_passes2.coverage ./.venv/bin/pytest tests/paiir/pipeline/test_passes.py -q -k split`
  - `env COVERAGE_FILE=/tmp/paiir_single_split_compile2.coverage ./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k split`
  - result: all passed; `test_compile.py -k split` still emits the pre-existing unrelated `AutoOptimizationWarning` from AvgPool LIF scoring coverage

# PAIIR Single SplitOp Through-Compile Refactor

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Do not create a dedicated `git worktree` for this task because it is a direct continuation of the in-flight split lowering work already present in this workspace.
- [x] Keep this task scoped to `paibox/paiir/ir/**`, `paibox/paiir/lowering/**`, `paibox/paiir/pipeline/**`, relevant `tests/paiir/**`, and `tasks/**`.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `paibox/paiir/ir/op_node.py`, `paibox/paiir/ir/graph.py`, `paibox/paiir/lowering/converter.py`, `paibox/paiir/lowering/split_lowering.py`, `paibox/paiir/pipeline/passes.py`, relevant `tests/paiir/**`, `tasks/**`
- Blocked Files: `paibox/backendv2/**`, unrelated dirty files, lockfiles, unrelated onboard/customer suites
- Dependencies: current single-output graph model, FX split analysis, simulation path in `PAIIRGraph.step()`, and compile validation/annotation passes
- Verification: focused lowering/simulation/pass/compile split tests plus one lower-level `tests/paiir/ir/test_graph.py` regression pass

## Interface Notes

- Keep `Edge` unchanged; do not add `src_port`.
- Treat `SplitOp` as the one explicit graph-internal multi-output special case.
- Keep `SplitOp` frontend-only IR; backend-ready graph validation must still reject it.

## Plan

- [x] Change lowering so one FX split producer becomes one `SplitOp`, while `getitem` remains a selector/bypass node only.
- [x] Extend `SplitOp` and graph simulation to support one split node feeding multiple consumers by successor mapping.
- [x] Update compile/pass validation to understand split-specific multi-output metadata without upgrading the whole graph model to generic multi-output support.
- [x] Run targeted regression tests and document the result below.

## Review

- Lowering now materializes one `SplitOp` per FX split producer instead of one `SplitOp` per consumed branch.
- `SplitOp` now stores:
  - `output_shapes: tuple[torch.Size, ...]`
  - `successor_output_index: dict[tuple[str, int], int]`
  - shared `output_dims`
- FX `getitem` users are bypassed; during wiring, the corresponding `SplitOp -> consumer` edge is added and the branch mapping is recorded on the single `SplitOp`.
- `PAIIRGraph.step()` now allows tuple-valued intermediate outputs only for `SplitOp`, and resolves the correct branch per outgoing edge using `successor_output_index`.
- Compile/pass validation now understands split-specific multi-output state:
  - generic single-`output_shape` checks skip `SplitOp`
  - `_validate_split_contract(...)` validates `output_shapes` and successor mapping coverage/index validity
  - concat/reshape predecessor-shape checks use edge-level shape resolution so downstream routing ops can consume one branch of a single `SplitOp`
- `validate_deployable_graph(...)` still rejects `SplitOp` unchanged.
- Verification:
  - `./.venv/bin/python -m py_compile paibox/paiir/ir/op_node.py paibox/paiir/ir/graph.py paibox/paiir/lowering/converter.py paibox/paiir/lowering/split_lowering.py paibox/paiir/pipeline/passes.py tests/paiir/lowering/test_converter.py tests/paiir/pipeline/test_graph_simulation.py tests/paiir/pipeline/test_passes.py tests/paiir/pipeline/test_compile.py`
  - `env COVERAGE_FILE=/tmp/paiir_single_split_lowering.coverage ./.venv/bin/pytest tests/paiir/lowering/test_converter.py -q -k split`
  - `env COVERAGE_FILE=/tmp/paiir_single_split_sim.coverage ./.venv/bin/pytest tests/paiir/pipeline/test_graph_simulation.py -q -k split`
  - `env COVERAGE_FILE=/tmp/paiir_single_split_passes.coverage ./.venv/bin/pytest tests/paiir/pipeline/test_passes.py -q -k split`
  - `env COVERAGE_FILE=/tmp/paiir_single_split_compile.coverage ./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k split`
  - `env COVERAGE_FILE=/tmp/paiir_single_split_ir_graph.coverage ./.venv/bin/pytest tests/paiir/ir/test_graph.py -q`
  - result: all passed; `test_compile.py -k split` still emits the pre-existing unrelated `AutoOptimizationWarning` from AvgPool LIF scoring coverage

# PAIIR Edge Src Port Split Refactor

## Workspace Decision

- [x] Preserve the current repository root on `dev` as the integration baseline for unrelated dirty state.
- [x] Create an archival branch `feat-paiir-single-splitop` that carries only the current PAIIR-related dirty work.
- [x] Implement the `Edge.src_port` refactor in a dedicated worktree on branch `feat-paiir-edge-src-port`.

## Ownership

- Owner: Codex
- Archival Branch: `feat-paiir-single-splitop`
- Refactor Branch: `feat-paiir-edge-src-port`
- Refactor Worktree: `../PAIBox-kafcoppelia-edge-src-port`
- Allowed Files: `paibox/paiir/**`, `tests/paiir/**`, `tasks/**`
- Blocked Files: `pyproject.toml`, `poetry.lock`, `uv.lock`, docs, onboard/customer artifacts, backend implementation, unrelated dirty files
- Verification: focused `pytest` over split graph/lowering/passes/compile behavior plus `py_compile`

## Interface Notes

- The archival branch must preserve the current single-`SplitOp` plus node-local mapping implementation as-is.
- The refactor branch must replace node-local split branch mapping with `Edge.src_port`, keeping backend rejection of `SplitOp` unchanged.
- `Edge.src_port` means: source node output port index; `dst_port` remains destination input port index.

## Plan

- [ ] Export only the current PAIIR-related dirty changes into `feat-paiir-single-splitop` and commit them there.
- [ ] Create worktree `../PAIBox-kafcoppelia-edge-src-port` on `feat-paiir-edge-src-port`, based on the archival branch.
- [ ] Refactor graph/lowering/simulation/passes/tests from `SplitOp.successor_output_index` to `Edge.src_port`.
- [ ] Run focused verification in the new worktree and record the results below.

## Review

- Pending.

# PAIIR FX Helper Import Cleanup

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Do not create a dedicated `git worktree` for this task because it is a narrow consistency cleanup on the in-flight lowering refactor already present in this workspace.
- [x] Keep this task scoped to `paibox/paiir/lowering/**` and `tasks/**`.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `paibox/paiir/lowering/**`, `tasks/**`
- Blocked Files: unrelated `paiir` pipeline/backend modules, lockfiles, unrelated dirty files
- Dependencies: current `fx_utils.py`, lowering modules that import it, and focused lowering regression tests
- Verification: py_compile plus focused split/functional-conv/reshape lowering tests

## Interface Notes

- Remove redundant `as _xxx` aliases when importing shared helpers from `fx_utils.py` if there is no local name conflict.
- Keep this as a naming cleanup only; no behavior changes.

## Plan

- [ ] Replace aliased `fx_utils` imports with direct imports in lowering modules and update call sites.
- [ ] Run a small regression slice that covers split, generic unsupported diagnostics, and representative conv/reshape lowering.

## Review

- Pending.

# PAIIR + Backendv2 User Guide

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Do not create a dedicated `git worktree` for this task because it is a documentation-only change scoped to `docs/**` and `tasks/**`.
- [x] Keep this task scoped to `docs/**` and `tasks/**`; do not edit `paibox/paiir/**` or `paibox/backendv2/**`.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `docs/**`, `tasks/**`
- Blocked Files: `paibox/**`, `tests/**`, lockfiles, unrelated dirty files
- Dependencies: current `paibox.paiir.compile_to_paiir()` public entrypoint, current `paibox.backendv2.mapper.Mapper` workflow, current packaging/dependency metadata in `pyproject.toml`, and existing docs in `docs/**`
- Verification: source review plus one local smoke flow covering import checks, `compile_to_paiir()`, and `backendv2.Mapper.compile()` output generation

## Interface Notes

- The guide should describe the current user-visible path from PyTorch model to `PAIIRGraph` to backendv2 frame export.
- Use `paibox.backendv2.mapper.Mapper` in examples because the task explicitly targets `backendv2`, not the legacy `paibox.backend` mapper API.
- Call out only constraints confirmed in code or smoke verification:
  - compile sample inputs must use batch size 1
  - backendv2 depends on `paicorelib`, `ortools`, and `numba`
  - `Mapper.compile()` currently writes export artifacts to the current working directory under `./output/`

## Plan

- [x] Review existing docs plus `paiir` / `backendv2` entrypoints and extract the minimal end-user workflow.
- [x] Write a concise Markdown guide in `docs/` covering installation, compile flow, backend export, outputs, and common caveats.
- [x] Verify the guide against a local smoke run and record the results below.

## Review

- Added `docs/paiir_backendv2_quickstart.md` as a simple user-facing guide for the current `paiir -> backendv2` workflow.
- The guide is intentionally team-facing rather than internals-facing:
  - installation is written around a source environment that actually satisfies the current compile chain
  - examples use `from paibox.backendv2.mapper import Mapper`
  - common failure points are called out before deeper backend details
- Covered topics:
  - recommended installation flow with `uv`
  - fallback `venv + pip` install path
  - minimal smoke-tested compile example
  - `compile_to_paiir()` vs `torch_to_paiir()` usage boundary
  - export artifact overview and links to deeper docs
  - common caveats such as batch size 1 and `./output/` export behavior
- Verification:
  - `./.venv/bin/python -c "import paibox, torch, paicorelib, spikingjelly, numba; from paibox.paiir import compile_to_paiir; from paibox.backendv2.mapper import Mapper; from ortools.sat.python import cp_model; print('imports_ok')"`
  - `./.venv/bin/python -c "import torch, torch.nn as nn; from paibox.paiir import compile_to_paiir; m=nn.Sequential(nn.Linear(4,4), nn.ReLU(), nn.Linear(4,2)).eval(); g=compile_to_paiir(m, torch.randn(1,4)); print(type(g).__name__, len(g.nodes), len(g.edges))"`
  - `mkdir -p /tmp/paiir_backendv2_doc_smoke && ./.venv/bin/python -c "import os, torch, torch.nn as nn; from paibox.paiir import compile_to_paiir; from paibox.backendv2.mapper import Mapper; os.chdir('/tmp/paiir_backendv2_doc_smoke'); m=nn.Sequential(nn.Linear(4,4), nn.ReLU(), nn.Linear(4,2)).eval(); g=compile_to_paiir(m, torch.randn(1,4)); mapper=Mapper(); mapper.compile(g); print(sorted(os.listdir('output')))"`
  - result:
    - imports passed
    - minimal compile returned `PAIIRGraph 4 3`
    - backendv2 export produced `frame_type1/2/3` in both `.txt` and `.h` forms under `output/`

# PAIIR + Backendv2 Quickstart Style Revision

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Do not create a dedicated `git worktree` for this task because it is a follow-up documentation-only style revision.
- [x] Keep this task scoped to `docs/**`, `tasks/todo.md`, and `tasks/lessons.md`.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `docs/**`, `tasks/**`
- Blocked Files: `paibox/**`, `tests/**`, lockfiles, unrelated dirty files
- Dependencies: the already verified `paiir -> backendv2` flow and the new quickstart draft in `docs/paiir_backendv2_quickstart.md`
- Verification: markdown review against the current compile/export flow

## Interface Notes

- Rewrite the doc in GitHub project style: short sections, command-first, low prose density.
- Keep the verified workflow and constraints unchanged; change presentation only.

## Plan

- [x] Rewrite the quickstart into a concise GitHub-style structure.
- [x] Record the style follow-up in `tasks/todo.md`.
- [x] Add the lesson to `tasks/lessons.md`.

## Review

- Rewrote `docs/paiir_backendv2_quickstart.md` into a shorter README-style quickstart.
- Kept only the sections needed for first-use onboarding:
  - workflow
  - requirements
  - install
  - quick start
  - output
  - notes
  - next
- Removed longer narrative explanations and team-process guidance from the quickstart body.
- Verification:
  - reviewed the rewritten markdown against the already verified workflow
  - kept the same entrypoints and constraints:
    - `compile_to_paiir(...)`
    - `from paibox.backendv2.mapper import Mapper`
    - `batch_size=1`
    - export to current working directory `./output/`

# PAIIR + Backendv2 Quickstart Content Expansion

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Do not create a dedicated `git worktree` for this task because it is a documentation-only follow-up.
- [x] Keep this task scoped to `docs/**`, `tasks/todo.md`, and `tasks/lessons.md`.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `docs/**`, `tasks/**`
- Blocked Files: `paibox/**`, `tests/**`, lockfiles, unrelated dirty files
- Dependencies: current `register_neuron(...)` API, `compile_to_paiir(...)` signature, and deployment examples under `tests/onboard/modelcnn/**` and `tests/user/customer_spikeyolo/**`
- Verification: source review of signatures and examples plus markdown consistency review

## Interface Notes

- Keep the quickstart concise, but include the missing operational details:
  - how to prepare a deployment model
  - how `register_neuron(...)` is used
  - what `compile_to_paiir(...)` parameters mean
  - which parts of the flow the user is expected to customize

## Plan

- [x] Inspect local APIs and examples for deployment-model preparation and custom neuron registration.
- [x] Expand the quickstart with concise sections for model prep, registration, parameter meanings, and customization points.
- [x] Record the correction in `tasks/todo.md` and `tasks/lessons.md`.

## Review

- Expanded `docs/paiir_backendv2_quickstart.md` while keeping the GitHub-style layout.
- Added:
  - `Prepare Model`
  - `Custom Neurons`
  - `compile_to_paiir(...) Parameters`
  - `Customization`
- Grounded the new content in current local sources:
  - `paibox/paiir/lowering/converter.py` for `register_neuron(...)`
  - `paibox/paiir/pipeline/compile.py` for `compile_to_paiir(...)`
  - `tests/onboard/modelcnn/test_modelcnn_paiir_deploy.py` for quantized deployment examples
  - `tests/user/customer_spikeyolo/workflow.py` for dynamic custom-neuron registration examples
- Verification:
  - reviewed the updated markdown against the current code signatures and example usage
  - no production code changed

# PAIIR + Backendv2 Quickstart Chinese Rewrite

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Do not create a dedicated `git worktree` for this task because it is a documentation-only correction.
- [x] Keep this task scoped to `docs/**`, `tasks/todo.md`, and `tasks/lessons.md`.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `docs/**`, `tasks/**`
- Blocked Files: `paibox/**`, `tests/**`, lockfiles, unrelated dirty files
- Dependencies: current `compile_to_paiir(...)` signature, `TickOverride` / `DataFormat` definitions, and local example usage in docs/tests
- Verification: source review plus markdown consistency review

## Interface Notes

- This document is a Chinese-language exception.
- Parameter customization examples should be organized by common use case, not by arbitrary numbers.
- Keep the GitHub-style structure, but switch all user-facing narration to Chinese.

## Plan

- [x] Inspect `TickOverride`, `DataFormat`, and example usage in local tests/docs.
- [x] Rewrite the quickstart in Chinese.
- [x] Replace arbitrary parameter examples with scenario-based templates.
- [x] Record the correction in `tasks/todo.md` and `tasks/lessons.md`.

## Review

- Rewrote `docs/paiir_backendv2_quickstart.md` into Chinese for this special case.
- Kept the quickstart structure concise, but expanded the parameter section into common deployment scenarios:
  - 默认部署编译
  - 多输入模型
  - 排查不支持算子
  - `CompileConfig` 统一配置
  - 统一时序参数
  - 逐节点覆盖时序
  - 输入数据格式覆盖
  - `concrete_args` 固定非 Tensor 参数
  - 实验特性开关
- Replaced arbitrary numeric examples with semantic placeholders such as:
  - `deploy_window_steps`
  - `target_start_step`
  - `target_duration_steps`
- Grounded the new content in local code:
  - `paibox/paiir/pipeline/compile.py`
  - `paibox/paiir/pipeline/passes.py`
  - `paibox/paiir/pipeline/data_format.py`
  - `tests/paiir/pipeline/test_compile.py`
  - `tests/onboard/modelcnn/test_modelcnn_paiir_deploy.py`
  - `tests/onboard/modelscnn/test_modelscnn_paiir_deploy.py`
- Verification:
  - reviewed the rewritten markdown against current signatures and existing usage patterns
  - no production code changed

# PAIIR + Backendv2 Quickstart Final User-Facing Cleanup

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Do not create a dedicated `git worktree` for this task because it is a documentation-only correction.
- [x] Keep this task scoped to `docs/**`, `tasks/todo.md`, and `tasks/lessons.md`.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `docs/**`, `tasks/**`
- Blocked Files: `paibox/**`, `tests/**`, lockfiles, unrelated dirty files
- Dependencies: current tracer behavior in `paibox/paiir/lowering/converter.py` and the local `MultiSpike4`/`LutCustom` example semantics used in tests
- Verification: source review plus markdown review

## Interface Notes

- The file will be delivered to end users as a standalone document; remove repository-internal “see this file” guidance.
- Use a class-style `nn.Module` minimal example instead of `nn.Sequential(...)`.
- Clarify that `register_neuron(...)` is sufficient to make the module trace as a leaf in this path.
- Explain the `MultiSpike4 -> LutCustom` mapping as a semantic correspondence, not just a registration snippet.

## Plan

- [x] Replace the minimal example with a `class Model(nn.Module)` form.
- [x] Remove repository-internal cross-file guidance from the quickstart.
- [x] Clarify `register_neuron(...)` leaf behavior and expand the `MultiSpike4 -> LutCustom` explanation.
- [x] Record the correction in `tasks/todo.md` and `tasks/lessons.md`.

## Review

- Replaced the minimal example with an explicit `class Model(nn.Module)` implementation.
- Removed end-user references to other repository files so the document stands on its own.
- Updated the custom-neuron section to explain:
  - the intended software semantics of `MultiSpike4`
  - how those semantics are encoded into `LutCustom.thresholds` and `LutCustom.values`
  - why `register_neuron(...)` is enough for leaf handling in the normal custom-module path
- Verification:
  - reviewed the revised text against:
    - `paibox/paiir/lowering/converter.py`
    - `tests/paiir/conftest.py`
    - `tests/paiir/lowering/test_converter.py`
  - no production code changed

# WiderFace T1 Backbone Customer Flow Supplement

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Do not create a dedicated `git worktree` for this task because it is a documentation-only customer-flow addition.
- [x] Keep this task scoped to `docs/**`, `tasks/todo.md`, and `tasks/lessons.md`.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `docs/**`, `tasks/**`
- Blocked Files: `paibox/**`, `tests/**`, lockfiles, unrelated dirty files
- Dependencies: current WiderFace customer harness under `tests/user/customer_spikeyolo/**` and `tests/user/test_widerface_t1_backbone_sz160_paiir.py`
- Verification: source review plus markdown review

## Interface Notes

- The quickstart audience is the WiderFace customer, not a generic internal developer.
- Supplement the generic compile flow with the model-specific path:
  - FX pickle input
  - external repo path requirement
  - fixed sample shape
  - fixed cut spec
  - runtime `mem_update` registration
  - report-first, backend-second workflow

## Plan

- [x] Inspect the WiderFace customer deployment flow and its helper utilities.
- [x] Add a dedicated WiderFace customer section to the quickstart.
- [x] Record the correction in `tasks/todo.md` and `tasks/lessons.md`.

## Review

- Added a dedicated `WiderFace T1 backbone sz160 客户专用步骤` section to `docs/paiir_backendv2_quickstart.md`.
- Added the customer-specific constraints that were missing from the generic quickstart:
  - the deployment input is a quantized FX pickle, not only a handwritten `nn.Module`
  - the customer must provide the external `SpikeYOLOforPAICORE` repo path
  - the validated sample shape is `(1, 3, 160, 160)`
  - the validated cut point is `p3`
  - runtime `mem_update` must be mapped to a `MultiSpike4`-style `ANNNodeV25(LutCustom(...))`
  - recommended flow is report first (`strict="both"`), backend export second
- Included a concrete customer-flow snippet using the current helper API.
- Verification:
  - reviewed the new section against:
    - `tests/user/test_widerface_t1_backbone_sz160_paiir.py`
    - `tests/user/customer_spikeyolo/workflow.py`
    - `tests/user/customer_spikeyolo/loader.py`
    - `tests/user/customer_spikeyolo/cuts.py`
    - `tests/user/customer_spikeyolo/run_compile.py`
  - no production code changed

# User Quickstart Output-File Integration

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Do not create a dedicated `git worktree` for this task because it is a documentation-only integration change.
- [x] Keep this task scoped to `docs/**`, `tasks/todo.md`, and `tasks/lessons.md`.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `docs/**`, `tasks/**`
- Blocked Files: `paibox/**`, `tests/**`, lockfiles, unrelated dirty files
- Dependencies: current `docs/user_quickstart.md` and `docs/后端输出文件含义.md`
- Verification: content review after integration

## Interface Notes

- `docs/user_quickstart.md` is the user-facing standalone document.
- Important content from `docs/后端输出文件含义.md` should be integrated into the standalone quickstart rather than left as a second required document.

## Plan

- [x] Review `docs/后端输出文件含义.md` and extract the user-facing parts.
- [x] Integrate the relevant output-file explanations into `docs/user_quickstart.md`.
- [x] Record the correction in `tasks/todo.md` and `tasks/lessons.md`.

## Review

- Integrated the practical parts of `docs/后端输出文件含义.md` into `docs/user_quickstart.md`.
- Added to the quickstart:
  - what `frame_type1/2/3` each represent
  - the difference between `.txt` and `.h`
  - which files are typically handed to firmware / board-side users
  - the extra CPU-side addressing information needed for input and output integration
- Kept the added section concise and user-facing rather than copying the whole original note verbatim.
- Verification:
  - reviewed the merged output section for consistency with both source documents
  - no production code changed

# PAIIR FX Target Name Helper

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Do not create a dedicated `git worktree` for this task because it is a narrow follow-up cleanup on the in-flight lowering refactor already present in this workspace.
- [x] Keep this task scoped to `paibox/paiir/lowering/**`, relevant `tests/paiir/lowering/**`, and `tasks/**`.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `paibox/paiir/lowering/**`, relevant `tests/paiir/lowering/**`, `tasks/**`
- Blocked Files: unrelated `paiir` pipeline/backend modules, lockfiles, unrelated dirty files
- Dependencies: current `fx_utils.py`, `split_lowering.py`, `converter.py`, and lowering diagnostics tests
- Verification: focused lowering tests covering split and generic unsupported function diagnostics

## Interface Notes

- If introduced, `get_fx_call_target_name(...)` should be a generic FX call-target display helper, not a split-specific helper renamed without widening semantics.
- The extraction must not change the current unsupported-op messages for supported split, unsupported split siblings, or generic unsupported functions.

## Plan

- [x] Add `get_fx_call_target_name(...)` to `fx_utils.py`.
- [x] Replace the duplicated target-name formatting in `split_lowering.py` and `converter.py`.
- [x] Run focused lowering tests and record the result below.

## Review

- Added `get_fx_call_target_name(...)` to `paibox/paiir/lowering/fx_utils.py`.
- Replaced duplicated FX target-name formatting in:
  - `paibox/paiir/lowering/split_lowering.py`
  - `paibox/paiir/lowering/converter.py`
- This helper is worth keeping because it centralizes one specific diagnostic concern that already had multiple call sites:
  - for `call_function`, prefer `target.__name__`
  - otherwise fall back to `str(node.target)`
- The extraction was kept narrow:
  - no behavioral change to split lowering
  - no broad “all node kinds” naming abstraction beyond the current diagnostic need
- Verification:
  - `env COVERAGE_FILE=/tmp/paiir_fx_target_name_split.coverage ./.venv/bin/pytest tests/paiir/lowering/test_converter.py -q -k split`
  - `env COVERAGE_FILE=/tmp/paiir_fx_target_name_unsupported.coverage ./.venv/bin/pytest tests/paiir/lowering/test_converter.py -q -k "strict_mode_raises or non_strict_mode_warns"`
  - result: all passed

# PAIIR Split Lowering Scope Cleanup

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Do not create a dedicated `git worktree` for this task because it is a direct follow-up cleanup on the in-flight lowering refactor already in this workspace.
- [x] Keep this task scoped to `paibox/paiir/lowering/**`, relevant `tests/paiir/lowering/**`, and `tasks/**`.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `paibox/paiir/lowering/**`, relevant `tests/paiir/lowering/**`, `tasks/**`
- Blocked Files: unrelated `paiir` pipeline/backend modules, lockfiles, unrelated dirty files
- Dependencies: current `split_lowering.py`, `converter.py`, and split-related lowering tests
- Verification: focused split lowering tests

## Interface Notes

- `split_lowering.py` should classify only the currently supported split operator family.
- Unsupported siblings such as `chunk` / `tensor_split` should fall back to generic unsupported-op handling in `converter.py`.

## Plan

- [x] Remove unsupported split-family operators from split-lowering dispatch constants.
- [x] Verify lowering still rejects `chunk` / `tensor_split` through the generic unsupported path and keeps supported `split` behavior unchanged.

## Review

- Removed `UNSUPPORTED_SPLIT_FUNCTIONS` and `UNSUPPORTED_SPLIT_METHODS` from `paibox/paiir/lowering/split_lowering.py`.
- `is_split_like_node(...)` now classifies only the currently supported split operator family:
  - `torch.split`
  - `Tensor.split`
- Resulting behavior is cleaner:
  - supported `split` continues to use the dedicated split-lowering path
  - unsupported siblings such as `chunk` and `tensor_split` now fall through `converter.py`'s generic unsupported-op handling instead of being mixed into the split-specific dispatch
- Verification:
  - `env COVERAGE_FILE=/tmp/paiir_split_scope_cleanup.coverage ./.venv/bin/pytest tests/paiir/lowering/test_converter.py -q -k split`
  - result: passed

# PAIIR Lowering Shared FX Utils Extraction

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Do not create a dedicated `git worktree` for this task because it is a direct cleanup follow-up on the in-flight split-lowering refactor already present in this workspace.
- [x] Keep this task scoped to `paibox/paiir/lowering/**` and `tasks/**`.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `paibox/paiir/lowering/**`, `tasks/**`
- Blocked Files: unrelated `paibox/paiir/pipeline/**`, backend code, lockfiles, unrelated dirty files
- Dependencies: current `converter.py`, `split_lowering.py`, `conv_lowering.py`, `shape_analysis.py`, and split-related regression tests
- Verification: focused lowering/split tests plus one functional-conv and one reshape/compile smoke

## Interface Notes

- Extract only pure FX helper utilities.
- Keep `converter.py` as lowering orchestrator.
- Remove callback-style plumbing from `split_lowering.py` where the shared helper can be imported directly.

## Plan

- [x] Extract shared FX helper utilities into a dedicated module.
- [x] Update lowering modules to import those helpers directly and simplify split helper signatures.
- [x] Run focused regression tests and document the result below.

## Review

- Added the shared lowering helper module `paibox/paiir/lowering/fx_utils.py`.
- Moved these pure FX utilities out of `converter.py`:
  - `get_call_arg`
  - `get_output_shape`
  - `get_input_shapes`
  - `get_output_dims`
  - `get_input_dims`
- Updated the lowering modules to import the shared helpers directly:
  - `converter.py`
  - `split_lowering.py`
  - `conv_lowering.py`
  - `shape_analysis.py`
- Simplified `split_lowering.py` by removing the callback-style helper injection from its public functions; it now imports the shared FX helpers directly, which reduces signature noise and keeps the split module focused on split semantics.
- This extraction keeps the architectural boundary cleaner:
  - `fx_utils.py` owns pure FX metadata/argument reads
  - `converter.py` remains the lowering orchestrator
  - operator-specific helpers (`split_lowering.py`, `conv_lowering.py`, `shape_analysis.py`) stop reimplementing the same FX access logic
- Verification:
  - `./.venv/bin/python -m py_compile paibox/paiir/lowering/converter.py paibox/paiir/lowering/split_lowering.py paibox/paiir/lowering/fx_utils.py paibox/paiir/lowering/conv_lowering.py paibox/paiir/lowering/shape_analysis.py`
  - `env COVERAGE_FILE=/tmp/paiir_fx_utils_split.coverage ./.venv/bin/pytest tests/paiir/lowering/test_converter.py -q -k split`
  - `env COVERAGE_FILE=/tmp/paiir_fx_utils_compile.coverage ./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k "quantized_functional_conv2d_supported_in_strict_mode or quantized_functional_conv1d_supported_in_strict_mode or shape_only_reshape_args_do_not_become_data_predecessors"`
  - `env COVERAGE_FILE=/tmp/paiir_fx_utils_split_sim.coverage ./.venv/bin/pytest tests/paiir/pipeline/test_graph_simulation.py -q -k split`
  - result: all passed

# PAIIR Split Lowering Module Refactor

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Do not create a dedicated `git worktree` for this task because it is a direct follow-up refactor on the in-flight split lowering edits already present in this workspace.
- [x] Keep this task scoped to `paibox/paiir/lowering/**`, `tests/paiir/lowering/**`, and `tasks/**` unless the extraction reveals a hard dependency that must move with it.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `paibox/paiir/lowering/converter.py`, new split-lowering helper module under `paibox/paiir/lowering/**`, relevant `tests/paiir/lowering/**`, `tasks/**`
- Blocked Files: `paibox/backendv2/**`, unrelated `paiir` pipeline/backend modules, lockfiles, unrelated dirty files
- Dependencies: current FX lowering flow in `converter.py`, `DimsProp`/shape meta helpers, `SplitOp` IR semantics, and existing split regression coverage
- Verification: targeted `pytest` for split lowering plus one compile/simulation smoke to confirm extraction did not change behavior

## Interface Notes

- Keep `converter.py` as the orchestration entry point for FX -> PAIIR lowering.
- Extract split-specific analysis and IR-node construction helpers into a dedicated lowering submodule.
- Do not change the supported split contract during this refactor.

## Plan

- [x] Identify the minimal split-specific helper set that can move without creating circular imports.
- [x] Extract those helpers into a dedicated lowering submodule and keep `converter.py` focused on orchestration/context wiring.
- [x] Run targeted split regression tests and document the result below.

## Review

- Extracted the split-specific lowering logic from `paibox/paiir/lowering/converter.py` into the new helper module `paibox/paiir/lowering/split_lowering.py`.
- `converter.py` now stays in the orchestration role:
  - it owns the generic FX helper functions (`_get_call_arg`, `_get_output_shape`, `_get_output_dims`)
  - it owns the lowering context and the overall analysis/lower/wire sequence
  - it delegates split-specific producer analysis, diagnostics, and IR-node construction to the new submodule
- The extracted module now owns:
  - split-producer contract parsing and validation
  - unsupported split diagnostics
  - split `getitem` consumer collection
  - `SplitOp` IR construction with copied shape/dims metadata
- Import direction remains one-way: `split_lowering.py` depends on IR and FX primitives only and does not import `converter.py`. `converter.py` passes the small generic helper callbacks it already owns, so the extraction did not introduce a circular dependency.
- Behavior was kept unchanged:
  - supported split contract remains the same
  - `SplitOp` stays frontend-only IR
  - the pending-layout rejection path remains intact
- Verification:
  - `./.venv/bin/python -m py_compile paibox/paiir/lowering/converter.py paibox/paiir/lowering/split_lowering.py`
  - `env COVERAGE_FILE=/tmp/paiir_split_refactor_lowering.coverage ./.venv/bin/pytest tests/paiir/lowering/test_converter.py -q -k split`
  - `env COVERAGE_FILE=/tmp/paiir_split_refactor_passes.coverage ./.venv/bin/pytest tests/paiir/pipeline/test_passes.py -q -k split`
  - `env COVERAGE_FILE=/tmp/paiir_split_refactor_sim.coverage ./.venv/bin/pytest tests/paiir/pipeline/test_graph_simulation.py -q -k split`
  - `env COVERAGE_FILE=/tmp/paiir_split_refactor_compile.coverage ./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k split`
  - result: all passed; `test_compile.py -k split` still emits the pre-existing unrelated `AutoOptimizationWarning` from AvgPool LIF scoring coverage

# PAIIR Split Review And Gap Fix

## Split Lowering Refactor

### Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Do not create a dedicated `git worktree` for this task because the refactor is a direct follow-up to the in-flight split lowering changes already present in this workspace, and moving away from those uncommitted edits would add unnecessary merge risk.
- [x] Keep this task scoped to `paibox/paiir/lowering/**`, relevant `tests/paiir/lowering/**`, and `tasks/**`.

### Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `paibox/paiir/lowering/converter.py`, new split-lowering helper module(s) under `paibox/paiir/lowering/`, relevant `tests/paiir/lowering/**`, `tasks/**`
- Blocked Files: `paibox/backendv2/**`, unrelated `paibox/paiir/pipeline/**` / `paibox/paiir/ir/**` changes unless the extraction proves they are required, lockfiles, unrelated dirty files
- Dependencies: current `_LoweringContext`, FX node metadata helpers, `SplitOp`, and existing split lowering tests
- Verification: focused `pytest` for split lowering and at least one compile/simulation split smoke to ensure extraction preserves behavior

### Interface Notes

- `converter.py` should remain the lowering orchestrator.
- Split-specific producer analysis, metadata filling, and IR registration are candidates to move into a dedicated submodule.
- The refactor must not change the current supported split contract or the existing unsupported-path errors.

### Plan

- [x] Identify the smallest extraction boundary that removes split-specific detail from `converter.py` without creating circular imports.
- [x] Move split-only helpers/types into a dedicated lowering submodule and update `converter.py` to call that API.
- [x] Run focused split-related tests and record the outcome below.

### Review

- Extracted split-specific lowering logic into `paibox/paiir/lowering/split_lowering.py`.
- Kept `converter.py` as the orchestrator:
  - it still owns the lowering pipeline order
  - it still owns `_LoweringContext`
  - it still owns generic IR registration and graph wiring
- Moved split-only responsibilities out of `converter.py`:
  - split producer analysis
  - unsupported split diagnostics
  - split getitem user collection
  - split IR-node construction
- Chose a dependency direction that avoids circular imports:
  - `split_lowering.py` depends on `SplitOp` and `DimsType`
  - `converter.py` injects generic helper callables such as `_get_call_arg`, `_get_output_shape`, and `_get_output_dims`
  - `split_lowering.py` does not import `converter.py`
- Resulting structure is narrower and easier to extend:
  - future split-specific support (`chunk`, richer multi-output routing, etc.) now has a dedicated file
  - `converter.py` is less burdened by operator-specific details
- Verification:
  - `./.venv/bin/python -m py_compile paibox/paiir/lowering/converter.py paibox/paiir/lowering/split_lowering.py`
  - `./.venv/bin/pytest tests/paiir/lowering/test_converter.py -q -k split`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_graph_simulation.py -q -k split`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k split`
  - result: all passed; compile split suite still emits the pre-existing unrelated `AutoOptimizationWarning` from AvgPool LIF scoring coverage

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Do not create a dedicated `git worktree` for this task because the user asked to review the current in-flight PAIIR split implementation, and the target files under `paibox/paiir/**` already contain local edits in this workspace that define the review baseline.
- [x] Keep this task scoped to `paibox/paiir/ir/**`, `paibox/paiir/lowering/**`, `paibox/paiir/pipeline/**`, `tests/paiir/**`, and `tasks/**`.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `paibox/paiir/ir/op_node.py`, `paibox/paiir/ir/graph.py`, `paibox/paiir/lowering/converter.py`, `paibox/paiir/pipeline/passes.py`, relevant `tests/paiir/**`, `tasks/**`
- Blocked Files: `paibox/backendv2/**`, lockfiles, unrelated dirty files, and unrelated onboard/customer suites
- Dependencies: current FX split analysis, single-output `PAIIRGraph` semantics, routing-op simulation, compile-pass validation/propagation, and existing split tests under `tests/paiir/**`
- Verification: focused `pytest` for split lowering, pass behavior, simulation, and compile rejection

## Interface Notes

- `SplitOp` remains frontend-only IR and must never survive into backend-ready compiled graphs.
- The supported frontend contract is still limited to `torch.split` / `Tensor.split` with static Python `split_size_or_sections`, integer `dim`, and direct integer `getitem` consumers.
- This review must check both functional correctness and metadata propagation: shape, dims, signal domain, data format, and tick-depth behavior.

## Plan

- [x] Audit the current split implementation across lowering, IR execution, and compile passes.
- [x] Identify concrete behavioral gaps or unsupported edge cases that should be rejected more clearly or handled correctly.
- [x] Patch the minimal production code required to close the confirmed gaps.
- [x] Add focused regression tests for every fixed gap.
- [x] Run targeted verification and record the outcome below.

## Review

- Confirmed current split contract in this workspace is:
  - frontend-only IR (`SplitOp`) for `torch.split` / `Tensor.split`
  - static Python `split_size_or_sections`
  - integer `dim`
  - direct non-negative integer `getitem` consumers only
- Gap fixed this turn: `split` after a pending `permute/transpose` layout transform could previously lower into PAIIR and then fail later with inconsistent simulation/runtime behavior because current PAIIR only materializes layout metadata at reshape boundaries. The lowerer now rejects that pattern early with a clear `UnsupportedOpError` instead of producing a broken graph.
- Regression coverage added:
  - `tests/paiir/lowering/test_converter.py` verifies method-form `x.split([1, 3], dim=1)` still canonicalizes to tuple `sections`
  - `tests/paiir/lowering/test_converter.py` verifies `permute -> split` is rejected with the new explicit unsupported-path message
- Targeted verification:
  - `./.venv/bin/pytest tests/paiir/lowering/test_converter.py -q -k split`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_passes.py -q -k split`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_graph_simulation.py -q -k split`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k split`
  - result: all passed when rerun sequentially; `test_compile.py -k split` still emits the pre-existing unrelated `AutoOptimizationWarning` from AvgPool LIF scoring coverage

# Backendv2 Weight Fill NumPy Evaluation

# Backendv2 Multi-RG Routing Feasibility Analysis

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Keep this task analysis-only in `tasks/**` because the user asked whether the current backendv2 OR-Tools model can express simultaneous routing feasibility constraints; no production edit is required to answer that.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `tasks/**`
- Blocked Files: all production source unless a follow-up implementation is requested
- Dependencies: current `paibox/backendv2/mapper.py`, `paibox/backendv2/route_solver.py`, `paibox/backendv2/routing.py`, and installed `paicorelib` routing primitives in `./.venv`
- Verification: inspect the current CP-SAT model, trace how routing-group adjacency enters it, and verify whether route encoding imposes additional hard feasibility bounds

## Interface Notes

- `Mapper.routing()` passes `next_rg_group` into `route_solve(...)` as the only routing-group relationship input.
- In `route_solve(...)`, those relationships currently affect only the distance term in the objective; they are not modeled as hard feasibility constraints.
- Actual destination programming happens later in `RoutingGroup.set_detail_dest()`, which computes a `CoordZXYOffset` from each source core to the destination group's `base_coord` and copies the destination group's multicast shape.
- In the current hive window, the derived `CoordZXYOffset` values remain well inside the hardware `CORE_Z/X/Y` bounds exposed by `paicorelib`.

## Plan

- [x] Inspect the current OR-Tools model in `route_solver.py`.
- [x] Trace how routing-group dependencies are produced and consumed in `mapper.py` / `routing.py`.
- [x] Verify whether post-placement route encoding introduces separate hard pairwise feasibility limits.
- [x] Summarize whether the requested multi-pair simultaneous routing condition is supported today and what model change would be needed if not.

## Review

- `paibox/backendv2/route_solver.py` places each routing group exactly once, prevents overlap, and minimizes center-to-center Manhattan distance for pairs in `next_area_id`; it does not add a boolean or linear constraint that says a specific routing-group pair must be routable.
- `paibox/backendv2/mapper.py` builds `next_rg_group` only from the actual dataflow edges discovered by `toposort_for_rg(...)`, so arbitrary extra pair requirements are not part of the current solver input contract.
- `paibox/backendv2/routing.py` programs per-neuron destinations after placement by using `find_coordxy_shortest_path(dest_coord, start=core_coord)` plus the destination group's multicast copy config; this is downstream encoding, not a CP-SAT feasibility check.
- Using the installed `paicorelib` in `./.venv`, the current hive window `(rows 2..8, cols 0..9)` yields a worst-case offset `(0, 6, -9)`, while hardware bounds are `CORE_Z/X/Y in [-31, 31]`, so current placements are not close to exhausting addressability.
- Conclusion: the current backendv2 OR-Tools model cannot guarantee a custom set of pairs like `(rg1, rg2)`, `(rg3, rg4)`, `(rg1, rg3)`, `(rg2, rg4)` as hard simultaneous routing conditions unless that pair set is added explicitly to the model and encoded as feasibility constraints rather than as soft distance preferences.

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Keep this task scoped to `tests/backendv2/**` and `tasks/**` because the user asked to benchmark first without touching production source and the repository already contains unrelated in-flight changes.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `tests/backendv2/**`, `tasks/**`
- Blocked Files: `paibox/backendv2/get_weight.py`, unrelated dirty files under `paibox/paiir/**`, `tests/paiir/**`, `pyproject.toml`, `uv.lock`, and all untracked non-task artifacts
- Dependencies: current `get_raw_weights()` call path in `paibox/backendv2/routing.py`, existing `target_cache` layout, and available local Python environment for micro-benchmarks
- Verification: inspect call flow, benchmark `_fill_weights_numba` against NumPy-only alternatives from test-only code, and report whether source changes are justified

## Interface Notes

- `get_raw_weights()` first expands each predecessor path into dense `[target_out, pred_out]` matrices and caches them per `(target, predecessor)`.
- `_fill_weights_numba()` does not build these dense matrices; it only copies indexed sub-blocks from `target_cache` into the final routing-group `weights` matrix.
- Any NumPy replacement must preserve overwrite semantics `weights[i, j] = matrix[mi, mj]` for every `(output target, input target)` pair.

## Plan

- [x] Inspect `_fill_weights_numba()` and surrounding data preparation to identify the true hot path and data-shape constraints.
- [x] Add a test-only benchmark harness for the current numba path, the plain Python path, and one or more NumPy-only indexing strategies.
- [x] Run the benchmark harness and decide whether numba is materially faster in this project context or whether a NumPy-only path is close enough to justify removing the dependency.
- [x] Summarize the results and recommend whether to keep numba, make it optional, or be replaced in a follow-up source change.
- [x] Extend the benchmark with larger routing-group profiles and compare multiple optimized NumPy-only variants.

## Review

- Added `tests/backendv2/test_get_weight_fill_benchmark.py` as a test-only harness that:
  - verifies correctness equality across `build_weights`, `build_weights_numba`, and a NumPy block-assignment candidate
  - benchmarks warm `numba`, plain Python, and pure-NumPy paths on sparse-to-dense synthetic routing-group profiles
  - measures a fresh-process numba cold-start cost
- Extended the harness with larger profiles and three NumPy-only strategies:
  - `numpy_fullscan`: scan every input target and use `path_matrices.get(...)`
  - `numpy_connected`: iterate only connected `path_matrices.items()`
  - `numpy_rowcache`: prefetch matrix rows before column gather
- Added a fourth NumPy-only strategy, `numpy_hybrid`, that:
  - precomputes per-target metadata once
  - uses direct slice-to-slice block copies whenever source/destination indices are contiguous
  - falls back to `np.ix_` only for fragmented/shuffled layouts
- Added shuffled profiles to avoid overfitting the benchmark to ideal contiguous target-major ordering.
- Benchmark results from `PAIBOX_RUN_BENCHMARKS=1 ./.venv/bin/pytest tests/backendv2/test_get_weight_fill_benchmark.py -q -s`:
  - `small_sparse`: python `2.005 ms`, numba warm `3.769 ms`, fullscan `1.213 ms`, connected `1.175 ms`, rowcache `1.271 ms`, hybrid `1.180 ms`
  - `medium_mixed`: python `9.109 ms`, numba warm `8.602 ms`, fullscan `3.755 ms`, connected `3.496 ms`, rowcache `3.209 ms`, hybrid `2.654 ms`
  - `large_blocky`: python `18.732 ms`, numba warm `9.117 ms`, fullscan `7.161 ms`, connected `6.992 ms`, rowcache `9.212 ms`, hybrid `3.841 ms`
  - `medium_mixed_shuffled`: python `9.333 ms`, numba warm `9.628 ms`, fullscan `3.680 ms`, connected `3.688 ms`, rowcache `3.567 ms`, hybrid `4.656 ms`
  - `xlarge_blocky`: python `52.276 ms`, numba warm `15.437 ms`, fullscan `19.125 ms`, connected `19.118 ms`, rowcache `47.738 ms`, hybrid `8.038 ms`
  - `xxlarge_blocky`: python `103.942 ms`, numba warm `39.476 ms`, fullscan `104.846 ms`, connected `99.018 ms`, rowcache `208.453 ms`, hybrid `43.419 ms`
  - `xxlarge_high_fanin`: python `176.352 ms`, numba warm `54.026 ms`, fullscan `152.801 ms`, connected `173.877 ms`, rowcache `403.823 ms`, hybrid `39.334 ms`
  - `many_targets_large`: python `144.531 ms`, numba warm `47.852 ms`, fullscan `67.519 ms`, connected `65.898 ms`, rowcache `142.588 ms`, hybrid `27.486 ms`
  - `xlarge_blocky_shuffled`: python `52.491 ms`, numba warm `35.537 ms`, fullscan `30.620 ms`, connected `28.446 ms`, rowcache `43.480 ms`, hybrid `27.111 ms`
  - numba cold start end-to-end: `2433.121 ms`
- Additional ad hoc ten-thousand-scale measurements using the same harness helpers:
  - `tenk_balanced`: weights shape `(10240, 10240)`, about `200.0 MiB`
    - numba warm `42.434 ms`
    - numpy connected `79.229 ms`
    - numpy hybrid `36.495 ms`
  - `tenk_balanced_shuffled`: weights shape `(10240, 10240)`, about `200.0 MiB`
    - numba warm `71.703 ms`
    - numpy connected `86.970 ms`
    - numpy hybrid `78.969 ms`
  - `twelvek_high_fanin`: weights shape `(12288, 12288)`, about `288.0 MiB`
    - numba warm `87.850 ms`
    - numpy connected `98.793 ms`
    - numpy hybrid `36.928 ms`
  - `fourteenk_balanced`: weights shape `(14336, 14336)`, about `392.0 MiB`
    - numba warm `73.113 ms`
    - numpy hybrid `43.269 ms`
- Conclusion from the benchmark:
  - numba is not the fastest option for small and medium routing-block workloads in this code path
  - the new `hybrid` NumPy algorithm is the strongest light-dependency candidate
  - on contiguous target-major layouts, `hybrid` often beats warm numba because it collapses the work into block slice copies instead of gathered element writes
  - even on shuffled layouts, `hybrid` remains competitive by falling back to `np.ix_`
  - among the older NumPy-only variants, `rowcache` is not a good direction for large blocks
  - when process cold-start matters, numba still loses badly because the first-call compile cost is on the order of seconds
- Additional risk observed during verification:
  - current `_fill_weights_numba()` emits `NumbaPendingDeprecationWarning` for reflected Python lists, so the present implementation depends on a numba feature that is already on a deprecation path

# ModelCNN Customer LUT Alignment

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Keep this task scoped to `tests/onboard/modelcnn/**` and `tasks/**` because the customer-specific LUT rule only affects the modelcnn deploy smoke and its task records.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `tests/onboard/modelcnn/**`, `tasks/**`
- Blocked Files: `paibox/paiir/**`, `tests/paiir/**`, unrelated dirty files
- Dependencies: existing `LutReLU` lowering path for explicit activation modules and current backendv2 compile/export flow
- Verification: rebuild `mnist_fc` and `mnist_lenet5` artifacts under `tests/onboard/modelcnn/debug/paiir_modelcnn_deploy/` and confirm backend compile still succeeds

## Interface Notes

- The customer `readme.txt` files define per-layer LUT parameters for every fused `*ReLU` block.
- `ReLU -> LutReLU` mapping already exists globally, but it uses the default `LutReLU()` range and therefore does not match the customer rule.
- For this task, each deploy-time activation must be an explicit `LutReLU(min_val=-5, max_val=o_s / (w_s * in_s) * 255, output_sign=0)`.

## Plan

- [x] Add explicit per-layer LUT configuration helpers to `test_modelcnn_paiir_deploy.py`.
- [x] Replace every deploy-time `ReLU` in the two customer models with the exact customer `LutReLU` instances.
- [x] Rebuild both modelcnn compile outputs and record the results.

## Review

- Added customer LUT config in [test_modelcnn_paiir_deploy.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/onboard/modelcnn/test_modelcnn_paiir_deploy.py):
  - global `min_val = -5`
  - per-layer `max_val = output_scale / (weight_scale * input_scale) * 255`
  - explicit layer tables for:
    - FC: `layer1_0`, `layer2_0`
    - LeNet5: `features_0`, `features_3`, `classifier_0`, `classifier_2`
- Directly replacing eager `ReLU` with real `LutReLU` modules broke FX shape propagation because `LutReLU.forward()` returns integer-coded outputs, which then fed into downstream `Linear` layers and triggered dtype mismatch.
- Final implementation uses a repo-local `CustomerLutReLU` float-forward shim that:
  - behaves as `torch.relu(x)` during eager shape propagation
  - is registered via `register_neuron(...)` so lowering emits `ANNNodeV25(LutReLU(min=-5, max=..., output_sign=0))`
- Simplified the final test file to keep only the two end-to-end deployment cases:
  - `mnist_fc`
  - `mnist_lenet5`
  - removed intermediate integer-domain and LUT-config unit tests because the user only needs the final deployment smoke
- Validation:
  - `python -m py_compile tests/onboard/modelcnn/test_modelcnn_paiir_deploy.py`
    - result: passed
  - `./.venv/bin/pytest tests/onboard/modelcnn/test_modelcnn_paiir_deploy.py -q -k 'customer_lut_configuration_matches_readme or integer_domain'`
    - result: `4 passed, 2 deselected`
  - `./.venv/bin/python - <<'PY' ...`
    - rebuilt persistent artifacts for both customer models under `tests/onboard/modelcnn/debug/paiir_modelcnn_deploy/`
    - result: `mnist_fc` compiled with `6` graph nodes and `3` routing groups
    - result: `mnist_lenet5` compiled with `10` graph nodes and `7` routing groups
  - `./.venv/bin/pytest tests/onboard/modelcnn/test_modelcnn_paiir_deploy.py -q`
    - result: `2 passed` in about `3m55s`
- Backend evidence:
  - [mnist_fc_backendv2.log](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/onboard/modelcnn/debug/paiir_modelcnn_deploy/mnist_fc_backendv2.log) now starts LUT thresholds with `[-5, 0, ...]`
  - [mnist_lenet5_backendv2.log](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/onboard/modelcnn/debug/paiir_modelcnn_deploy/mnist_lenet5_backendv2.log) now starts LUT thresholds with `[-5, 0, ...]`

# PAIIR IF/LIF Lowering Surrogate Fidelity

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Keep this task scoped to `paibox/paiir/lowering/converter.py`, `tests/paiir/lowering/test_converter.py`, and `tasks/**`.
- [x] Do not create a dedicated `git worktree` for this change because the target lowering files already have in-flight local edits in the current workspace; preserving and extending that active context is safer than branching from `HEAD` and losing the uncommitted state.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `paibox/paiir/lowering/converter.py`, `tests/paiir/lowering/test_converter.py`, `tasks/**`
- Blocked Files: unrelated dirty files, especially `paibox/paiir/ir/core_neuron.py`, `paibox/paiir/ir/graph.py`, `tests/paiir/ir/test_graph.py`, `pyproject.toml`, `uv.lock`
- Dependencies: existing `IFNodeV25` / `LIFNodeV25` constructor support for `surrogate_function` and `detach_reset`
- Verification: targeted `pytest` for converter lowering coverage around SpikingJelly IF/LIF conversion

## Interface Notes

- `IFNodeV25` and `LIFNodeV25` already accept `surrogate_function` and `detach_reset`.
- The current loss of fidelity happens in lowering: SpikingJelly `IFNode` / `LIFNode` are converted without forwarding those two attributes.
- This task should preserve both activation-based and legacy `clock_driven` compatibility paths.

## Plan

- [x] Update IF/LIF lowering helpers in `converter.py` so converted `IFNodeV25` / `LIFNodeV25` receive the source neuron's `surrogate_function` and `detach_reset`.
- [x] Extend converter tests to assert those attributes are preserved after lowering.
- [x] Run focused verification and record the outcome below.

## Review

- Lowering now forwards `surrogate_function` and `detach_reset` for both activation-based and legacy `clock_driven` SpikingJelly `IFNode` / `LIFNode` mappings in [converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/converter.py).
- Added focused regression coverage in [test_converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/lowering/test_converter.py) that asserts lowered `IFNodeV25` / `LIFNodeV25` keep the original neuron's `surrogate_function` object and `detach_reset` flag.
- Verification:
  - `python -m py_compile paibox/paiir/lowering/converter.py tests/paiir/lowering/test_converter.py`
    - result: passed
  - `./.venv/bin/pytest tests/paiir/lowering/test_converter.py -q`
    - result: `11 passed`

# ModelCNN MNIST PAIIR Deploy Smoke

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Keep this task scoped to `tests/onboard/modelcnn/**` and `tasks/**` because the MNIST model assets and task ledger are already local to this workspace, while the task does not need to edit the in-flight `paibox/paiir/**` changes currently present in the tree.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `tests/onboard/modelcnn/**`, `tasks/**`
- Blocked Files: `paibox/paiir/**`, `tests/paiir/**`, unrelated dirty files
- Dependencies: existing `paiir` lowering / backendv2 compile support for `Linear`, `Conv2d`, `ReLU`, `MaxPool2d`, `Flatten`
- Verification: targeted `pytest` for the new onboard test file plus an explicit `compile_to_paiir + Mapper.compile` deployment smoke for both MNIST models

## Interface Notes

- `paiir` can lower the plain ANN topology used by these MNIST models (`Linear`, `Conv2d`, `ReLU`, `MaxPool2d`, `Flatten`) today.
- The exported `manual_quant_*` artifacts are raw `torch.int8` weight tensors plus `torch.int32` bias tensors; they are not `paiir`-native modules and therefore must be loaded into a host-side reconstruction model for compile smoke.
- This task should prove deployability of the operator topology and exported integer parameters without claiming that the current path is an exact on-chip requantization implementation.

## Plan

- [x] Add a focused onboard deployment test module for the MNIST FC and LeNet5 quantized exports.
- [x] Implement helpers to load the exported `int8` / `int32` tensors, reconstruct deployable host-side models, and assert parameter-domain expectations.
- [x] Run `paiir` compile plus `backendv2` deployment smoke for both reconstructed models and capture the operator-support conclusion in test assertions / logs.
- [x] Run targeted verification and record the results in the review section below.

## Review

- Added [test_modelcnn_paiir_deploy.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/onboard/modelcnn/test_modelcnn_paiir_deploy.py) with two verification layers:
  - integer-domain export checks for all saved `weight_int8` / `bias_int32` tensors
  - deployment smoke that reconstructs host-side ANN graphs from those tensors and runs `paiir -> backendv2`
- Operator-support conclusion:
  - The exported `manual_quant_*` modules are not lowered by `paiir` directly today.
  - The underlying quantized MNIST topologies are deployable because `paiir` already supports the plain operators they reduce to:
    - FC: `view/flatten`, `Linear`, `ReLU`, final `Linear`
    - LeNet5: `Conv2d`, `ReLU`, `MaxPool2d`, `flatten`, `Linear`
  - The test therefore reconstructs those deployable operator graphs with the exported `torch.int8` weights and `torch.int32` biases attached as buffers for graph-side weight inspection.
- Observed compiled operator shapes:
  - FC compiled to `2 x SequentialOp`, `1 x StandaloneCompOp`, `1 x ReshapeOp`; backendv2 routing groups = `3`
  - LeNet5 compiled to `4 x SequentialOp`, `3 x StandaloneCompOp`, `1 x ReshapeOp`; direct smoke verification routing groups = `7`
- Verification:
  - `python -m py_compile tests/onboard/modelcnn/test_modelcnn_paiir_deploy.py`
    - result: passed
  - `./.venv/bin/pytest tests/onboard/modelcnn/test_modelcnn_paiir_deploy.py -q -k 'integer_domain'`
    - result: `2 passed, 2 deselected`
  - `./.venv/bin/pytest tests/onboard/modelcnn/test_modelcnn_paiir_deploy.py -q -k 'mnist_fc_quantized_exports_compile_to_paiir_and_backend'`
    - result: `1 passed, 3 deselected`
    - note: backend emitted existing `NumbaPendingDeprecationWarning` warnings from `paibox/backendv2/get_weight.py`
  - `./.venv/bin/python - <<'PY' ...`
    - result: LeNet5 direct deployment smoke passed with `node_type_counts {'sequential': 4, 'standalone_comp': 3, 'reshape': 1}`, `routing_groups 7`
    - note: repeated pytest session output for the LeNet5 case was flaky in the terminal tool, but the generated PAIIR/backend logs and the direct smoke run both completed successfully

# PAIIR output_domain Placement Review

## Workspace Decision

- [x] Continue in the current workspace on `feat/layout-two-pass-canonicalization`.
- [x] Keep this task review-only: inspect `paibox/paiir/**` abstractions and avoid product-code edits while evaluating future online-core / RV CPU operator support.

## Plan

- [x] Audit what `output_domain` currently means and where it is consumed.
- [x] Evaluate whether `output_domain` belongs to all `PAIIRNode` variants or should move to a narrower abstraction.
- [x] Record a recommendation that keeps future `OnlineCoreOp` / `CPUOp` support clean.

## Review

- Current semantics:
  - `output_domain` in [ir_base.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/ir_base.py#L20) is a graph-wide annotation shared by `InputNode`, `OutputNode`, and all operator nodes.
  - The current propagation logic in [passes.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/passes.py#L607) treats it as signal semantics (`POTENTIAL` vs `VALUE`), not as execution placement.
  - `OfflineCoreOp` output can already derive this annotation from neuron params in [passes.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/passes.py#L696), while routing/boundary nodes simply forward or constrain predecessor domains.
- Architecture assessment:
  - Defining `output_domain` directly on `PAIIRNode` is acceptable as a short-term simplification because the current IR is a single-output tensor graph.
  - It is not the best long-term abstraction boundary, because `PAIIRNode` is otherwise just graph identity, while `output_domain` is really output-tensor contract metadata.
  - Moving it down to `OpNode` alone would be too narrow, because `InputNode` / `OutputNode` also participate in signal-domain validation and propagation.
- Recommendation:
  - Keep `PAIIRNode` focused on node identity.
  - Introduce a dedicated output-metadata layer such as `NodeOutputSpec` / `SingleOutputTensorNode`, owned by `InputNode`, `OutputNode`, and `OpNode`.
  - Rename the field to `output_signal_domain` if it stays scalar, to avoid confusion with future execution domains such as offline core / online core / RV CPU.
  - Keep execution placement orthogonal, either through the existing class families (`OfflineCoreOp`, `OnlineCoreOp`, `CPUOp`) or a separate explicit enum/property.
- Future-facing implications:
  - `OnlineCoreOp` and `CPUOp` already sit under the shared operator layer in [op_node.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/op_node.py#L125), so the real expansion pressure is not node inheritance but semantic richness.
  - The current `SignalDomain` enum (`POTENTIAL` / `VALUE`) may be too narrow once online learning cores or CPU-side custom ops need richer output semantics.
  - If PAIIR later supports multi-output ops or non-tensor side channels, the correct long-term shape is per-output spec metadata rather than a single scalar field on the base node.

# Rebase Current Branch Onto Remote Dev

## Workspace Decision

- [x] Continue in the current workspace because `feat-paiir-layout-ir-cleanup` is already inside an interactive rebase onto remote `dev`.
- [x] Keep the scope limited to resolving the paused rebase, preserving unrelated untracked files, and verifying the rebased `paibox/paiir/**` plus `tests/paiir/**` result.

## Plan

- [x] Inspect the paused rebase state and confirm the intended resolution for the currently replayed commit.
- [x] Stage the resolved files and continue the rebase onto `dev`.
- [x] Resolve any remaining replay conflicts from later commits if they appear.
- [x] Run focused verification on the rebased branch and record the outcome.

## Review

- The paused rebase stopped while replaying `874db47` (`✨ Feat(paiir): canonicalize layout-only routing before fusion`) onto remote `dev`.
- Conflict resolution outcome:
  - staged the five modified files that had already been resolved in the working tree
  - staged the five newly introduced layout-canonicalization files/tests that still needed to be added to the index
  - continued the rebase successfully and replayed the final commit `9e7d9ae` (`♻️ refactor(paiir): align shape metadata and graph validation`) without needing further manual conflict edits
- Final branch tip after rebase:
  - `62577c3` `♻️ refactor(paiir): align shape metadata and graph validation`
  - `b4f0b61` `✨ Feat(paiir): canonicalize layout-only routing before fusion`
  - `2348f92` `♻️ refactor(paiir): harden IR naming and graph rewrites`
- Verification:
  - `./.venv/bin/pytest tests/paiir/ir/test_reshape_semantics.py tests/paiir/pipeline/test_layout_chain_canonicalization.py tests/paiir/pipeline/test_layout_cross_node_elision.py tests/paiir/lowering/test_converter.py tests/paiir/pipeline/test_passes.py -q`
    - result: `90 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py tests/paiir/pipeline/test_graph_simulation.py -q -k 'maxpool_after_reshape_compiles or function_unsqueeze_before_linear_compiles or tuple_repeat_all_ones_before_linear_compiles or method_squeeze_before_linear_compiles or function_squeeze_before_linear_compiles or test_view_and_view_as_routing_before_linear or test_function_unsqueeze_before_linear or test_tuple_repeat_all_ones_before_linear or test_method_squeeze_before_linear or test_function_squeeze_before_linear or test_flatten_then_reshape_chain_before_linear'`
    - result: `11 passed, 128 deselected`

# PAIIR Commit Split And Branch Cleanup

## Workspace Decision

- [x] Continue in the current workspace because the user explicitly wants the already accumulated `paibox/paiir/**` and `tests/paiir/**` changes reviewed and split in place.
- [x] Limit staging, review, and commits to `paibox/paiir/**` plus `tests/paiir/**`; ignore unrelated working-tree changes from other Codex sessions.

## Plan

- [ ] Review the current `paibox/paiir/**` and `tests/paiir/**` diff and group it into coherent commit themes.
- [ ] Optionally rename the branch to better match the final scope of the grouped commits.
- [ ] Create a small sequence of focused commits with messages matching the existing project style.
- [ ] Run targeted verification after each thematic commit and one broader `tests/paiir` pass at the end.

# PAIIR Test Fallout From Shape/Canonicalization Tightening

## Workspace Decision

- [x] Continue in the current workspace on `feat/layout-two-pass-canonicalization`.
- [x] Keep the scope on affected `tests/paiir/**` assertions only; the runtime behavior is already correct and the failures come from outdated graph-shape expectations.

## Plan

- [x] Reproduce the affected test failures after the shape-type and reshape canonicalization tightening.
- [x] Update stale test expectations to match the new canonicalized graph topology.
- [x] Re-run the affected targeted tests and the broader `tests/paiir` suite.

## Review

- The fallout was not a new runtime bug; it was a stale expectation in compile/simulation tests that still assumed two routing reshapes would remain for:
  - `unsqueeze -> flatten -> linear`
  - identity `repeat((1,...)) -> flatten -> linear`
  - `squeeze -> flatten -> linear`
  - `view -> view_as -> linear`
- Current canonicalization correctly collapses those patterns to a single remaining reshape before the `Linear`, so the tests were updated accordingly in:
  - [test_compile.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_compile.py)
  - [test_graph_simulation.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_graph_simulation.py)
- Focused verification:
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k 'function_unsqueeze_before_linear_compiles or tuple_repeat_all_ones_before_linear_compiles or method_squeeze_before_linear_compiles or function_squeeze_before_linear_compiles'`
    - result: `4 passed`

# PAIRV SDK runmode.mk Usage Analysis

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Keep this task review-only and scoped to the sibling SDK at `../PAIRV/application/**`, `../PAIRV/Build/**`, `../PAIRV/NMSIS/**`, `../PAIRV/OS/**`, `../PAIRV/SoC/**`, the SDK root build entry files, and `tasks/**`.
- [x] Do not create a dedicated `git worktree` because this task only needs static build-system inspection and no product-code edits are planned.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `../PAIRV/application/**`, `../PAIRV/Build/**`, `../PAIRV/NMSIS/**`, `../PAIRV/OS/**`, `../PAIRV/SoC/**`, `../PAIRV/Makefile`, `tasks/**`
- Blocked Files: unrelated dirty files outside the PAIRV SDK analysis scope
- Dependencies: existing PAIRV CLI make entrypoints and the local include chain among root makefiles and `SoC/evalsoc/*.mk`
- Verification: prove conclusions from actual include/reference paths and variable consumers instead of inference only

## Plan

- [x] Inspect the PAIRV root build entry and the relevant makefile include graph.
- [x] Trace every reference to `SoC/evalsoc/runmode.mk` and the variables it defines.
- [x] Decide whether `runmode.mk` has any effect on the current CLI application build flow and record the evidence below.

## Review

- CLI build entry:
  - SDK root [Makefile](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/Makefile) only forwards targets such as `all` / `showflags` into the selected app directory via `make -C $(VALID_PROGRAM) $@`.
  - Every inspected application Makefile includes [Makefile.base](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/Build/Makefile.base), which includes [Makefile.conf](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/Build/Makefile.conf), which includes [Makefile.soc](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/Build/Makefile.soc), which includes [SoC/evalsoc/build.mk](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/build.mk#L30).
- `runmode.mk` is not dead code:
  - [SoC/evalsoc/build.mk](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/build.mk#L96) uses `-include $(NUCLEI_SDK_SOC)/runmode.mk`, so the file is always attempted in the evalsoc build path.
  - [runmode.mk](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/runmode.mk#L6) maps `RUNMODE=lm/icdlm/dcilm/cache/bus/clm` into `RUNMODE_*` macros added to `COMMON_FLAGS`.
  - Those macros are actually consumed in [system_evalsoc.c](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Common/Source/system_evalsoc.c#L847) to enable or disable ILM, DLM, cache ECC, L2, and BPU during `_premain_init()`.
  - [evalsoc.h](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Common/Include/evalsoc.h#L403) also uses `RUNMODE_IC_EN` / `RUNMODE_DC_EN` / `RUNMODE_CCM_EN` to override `__ICACHE_PRESENT`, `__DCACHE_PRESENT`, and `__CCM_PRESENT`.
- But for the current default CLI application flow, it is dormant:
  - No inspected application, root makefile, or build helper sets `RUNMODE` or any `XLCFG_*` variable by default; the only definitions are inside [runmode.mk](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/runmode.mk).
  - `make -C ../PAIRV PROGRAM=application/baremetal/helloworld showflags` produced `CFLAGS/ASMFLAGS` without any `-DRUNMODE_*` macro.
  - `make -C ../PAIRV PROGRAM=application/baremetal/helloworld RUNMODE=cache showflags` immediately added `-DRUNMODE_CONTROL -DRUNMODE_ILM_EN=0 -DRUNMODE_DLM_EN=0 -DRUNMODE_IC_EN=1 -DRUNMODE_DC_EN=1 -DRUNMODE_CCM_EN=1`, proving the file only affects CLI builds when the user explicitly passes these knobs or when another makefile sets them.
- Conclusion:
  - `SoC/evalsoc/runmode.mk` is useful as an optional evalsoc-specific runtime/feature override layer.
  - It is not required for the default “just compile application from CLI” path in this SDK; default builds still work and do not use any of its macros.
  - If your customized SDK does not intend to support CLI knobs like `RUNMODE=cache` or `XLCFG_PLIC=1`, the file is effectively optional for your current use case, though not meaningless.
- Additional higher-priority inconsistency found during verification:
  - [cpufeature.h](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Common/Include/cpufeature.h) says `CFG_CPU_NAME "n307"`.
  - The current local [SoC/evalsoc/build.mk](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/build.mk#L30) now hardcodes `override CORE := n307fd`, while [cpufeature.mk](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/cpufeature.mk#L2) still defaults to `CORE ?= n300fd`.
  - Verified with both `make -C ../PAIRV PROGRAM=application/baremetal/helloworld CORE=n300fd info` and `CORE=n307fd info`: the effective configuration is always `CORE=n307fd` because the SoC build file overrides it.
  - Follow-up impact check:
    - [Makefile.core](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/Build/Makefile.core#L17) and [Makefile.core](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/Build/Makefile.core#L20) map `n300fd` and `n307fd` to the same tuple: `rv32imafdc ilp32d nuclei-300-series`.
    - Therefore, in this SDK, changing only `CORE` between those two values does not change `-march`, `-mabi`, `-mtune`, `CPU_SERIES`, linker script selection, or normal upload/build flow.
    - What it can change is the simulator/model CPU name strings assembled in [build.mk](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/build.mk#L203) and [build.mk](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/build.mk#L214):
      - QEMU: `-cpu nuclei-$(CORE),ext=...`
      - XLModel: `--cpu=$(CORE)`
    - `xl_spike` does not use `CORE` directly; it only uses `RISCV_ARCH`, so it remains unaffected here.
  - For a truly customized N307 SDK, the more meaningful hardware difference is carried by generated SoC config such as [cpufeature.h](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Common/Include/cpufeature.h), not by the `CORE` selector alone.

# PAIRV Default Eval Naming Rename Impact Review

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Keep this task review-only and scoped to the primary SDK source tree under `../PAIRV/application/**`, `../PAIRV/Build/**`, `../PAIRV/NMSIS/**`, `../PAIRV/OS/**`, `../PAIRV/SoC/**`, the SDK root files, and `tasks/**`.
- [x] Do not create a dedicated `git worktree` because this phase is impact assessment only and no product-code edits are planned.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `../PAIRV/application/**`, `../PAIRV/Build/**`, `../PAIRV/NMSIS/**`, `../PAIRV/OS/**`, `../PAIRV/SoC/**`, `../PAIRV/Makefile`, `../PAIRV/README.md`, `tasks/**`
- Blocked Files: unrelated dirty files and IDE-export/workspace artifacts outside the primary SDK source tree
- Dependencies: actual CLI SDK include graph and the current SoC/board naming conventions rooted at `evalsoc` and `nuclei_fpga_eval`
- Verification: classify every relevant default-eval identifier occurrence by whether it affects build paths, headers/APIs, board assets, docs, or generated/secondary artifacts

## Plan

- [x] Inventory primary-source references to `evalsoc`, `nuclei_fpga_eval`, `BOARD_NUCLEI_FPGA_EVAL`, and adjacent default-eval identifiers.
- [x] Group the findings into rename units with concrete breakage risk and coupling.
- [x] Recommend whether to rename now or defer based on impact and the current customization maturity.

## Review

- Scope audited:
  - primary SDK source only, per user guidance: `../PAIRV/application/**`, `../PAIRV/Build/**`, `../PAIRV/NMSIS/**`, `../PAIRV/OS/**`, `../PAIRV/SoC/**`, plus `../PAIRV/Makefile` and `../PAIRV/README.md`
  - explicitly not counted into the main recommendation: `../PAIRV/Nuclei/**` IDE-export workspaces and generated build products
- High-level size:
  - `24` primary files contain `evalsoc`-style names or identifiers
  - `5` primary files contain board-name identifiers such as `nuclei_fpga_eval` or `board_nuclei_fpga_eval`
  - the active SoC tree under `../PAIRV/SoC/evalsoc` contains `31` non-generated source/resource files
  - current public SoC headers expose `41` distinct `EVALSOC_*` macro names across `cpufeature.h` and `evalsoc.h`
  - application/OS sources include the stable umbrella header `nuclei_sdk_soc.h` in `9` places, which means compatibility wrappers are possible if renaming is ever staged

- Impact grouping:
  - `1. Build identity and directory layout`
    - changing `SOC=evalsoc` means renaming the SoC directory itself (`SoC/evalsoc`) plus root/default references in:
      - [Makefile.base](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/Build/Makefile.base#L20)
      - [Makefile.soc](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/Build/Makefile.soc#L1)
      - [Makefile.rules](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/Build/Makefile.rules#L59)
      - [README.md](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/README.md#L84)
    - this is not just text replacement, because the build system resolves `$(NUCLEI_SDK_ROOT)/SoC/$(SOC)/build.mk` dynamically
  - `2. Board identity`
    - changing `BOARD=nuclei_fpga_eval` is comparatively small and localized:
      - [build.mk](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/build.mk#L31)
      - board directory `SoC/evalsoc/Board/nuclei_fpga_eval`
      - [board_nuclei_fpga_eval.h](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Board/nuclei_fpga_eval/Include/board_nuclei_fpga_eval.h)
      - [nuclei_sdk_hal.h](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Board/nuclei_fpga_eval/Include/nuclei_sdk_hal.h#L9)
    - this is the easiest safe rename slice
  - `3. SoC file names and internal include graph`
    - the SoC tree uses `evalsoc` heavily in file names, not only contents:
      - [evalsoc.h](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Common/Include/evalsoc.h)
      - [system_evalsoc.h](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Common/Include/system_evalsoc.h)
      - [evalsoc_uart.h](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Common/Include/evalsoc_uart.h)
      - [system_evalsoc.c](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Common/Source/system_evalsoc.c)
      - [evalsoc_uart.c](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Common/Source/Drivers/evalsoc_uart.c)
      - [startup_evalsoc.S](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Common/Source/GCC/startup_evalsoc.S)
      - [intexc_evalsoc.S](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Common/Source/GCC/intexc_evalsoc.S)
      - [intexc_evalsoc_s.S](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Common/Source/GCC/intexc_evalsoc_s.S)
      - [evalsoc_common.c](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Common/Source/evalsoc_common.c)
    - [nuclei_sdk_soc.h](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Common/Include/nuclei_sdk_soc.h#L9) currently includes `evalsoc.h` and `evalsoc_uart.h`, so a hard rename would require either compatibility wrappers or coordinated include changes
  - `4. Public macro/API surface`
    - [cpufeature.h](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Common/Include/cpufeature.h) and [evalsoc.h](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Common/Include/evalsoc.h) export `41` distinct `EVALSOC_*` macros
    - changing these names is an API break for any external code, BSP layer, or future app code that references those macros directly
    - this is the main reason the rename is much broader than a board-directory cleanup
  - `5. Linker/openocd/script asset names`
    - board assets also embed `evalsoc` in filenames:
      - linker scripts `gcc_evalsoc_*.ld`
      - IAR scripts `iar_evalsoc_*.icf`
      - [openocd_evalsoc.cfg](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Board/nuclei_fpga_eval/openocd_evalsoc.cfg)
      - [evalsoc.memory](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/Board/nuclei_fpga_eval/Source/GCC/evalsoc.memory)
    - these are referenced by [build.mk](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/build.mk#L75) and [build.mk](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/build.mk#L84), so renaming them is feasible but must be done as a coordinated set
  - `6. External tool contracts`
    - [build.mk](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/build.mk#L193) sets `QEMU_MACHINE ?= nuclei_evalsoc,download=$(DOWNLOAD)`
    - [build.mk](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/SoC/evalsoc/build.mk#L213) sets `XLMODEL_OPT += -M nuclei_evalsoc`
    - those names are not just SDK labels; they are machine/model names expected by external tools
    - if you rename them without matching support in QEMU / XLModel, simulation targets will break even if compilation still works
  - `7. Docs/help/comments`
    - user-facing text in [README.md](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/README.md) and [Makefile.rules](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/Build/Makefile.rules#L59) would also need refresh for consistency
    - application code only has a couple of harmless comments mentioning `system_evalsoc.c`, for example [main.c](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/application/baremetal/uart/main.c#L65)

- Recommendation:
  - a full hard rename of all `evalsoc` / `EVALSOC_*` / `nuclei_fpga_eval` identifiers is a broad refactor, not a cosmetic cleanup
  - the risky part is not the board name; it is the SoC/API/tooling identity
  - for the current SDK state, this is too much for a casual rename pass and should be deferred unless you first choose an explicit compatibility strategy
  - the safest staged approach would be:
    - stage 1: rename board identity only (`BOARD`, board dir, board header, board docs)
    - stage 2: if still needed, rename SoC-internal filenames while keeping compatibility shim headers/macros
    - stage 3: only consider renaming `EVALSOC_*` macros and `nuclei_evalsoc` tool model names if you control all downstream code and the simulator/model support

# PAIRV Root Makefile Retention Review

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Keep this task review-only and scoped to the sibling SDK root `../PAIRV/Makefile`, related app makefiles, root docs, and `tasks/**`.
- [x] Do not create a dedicated `git worktree` because this phase is only evaluating retention and simplification, with no planned source edits yet.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `../PAIRV/Makefile`, `../PAIRV/application/**/Makefile`, `../PAIRV/Build/**`, `../PAIRV/README.md`, `tasks/**`
- Blocked Files: unrelated dirty files and the broader rename candidates outside this focused review
- Dependencies: current CLI app layout and the root Makefile forwarding model
- Verification: base the recommendation on concrete entrypoint behavior and actual references, not preference alone

## Plan

- [x] Inspect the root Makefile behavior and responsibility boundaries.
- [x] Trace where the root entrypoint is documented or relied on.
- [x] Recommend keep/remove plus a minimal simplification set if it stays.

## Review

- Current value if kept:
  - a repo-root CLI entrypoint for single-app operations such as `make PROGRAM=... all/info/showflags/upload`
  - repo-root bulk operations `buildall` / `cleanall`
  - a single user-facing command surface that matches the README positioning of [Makefile](/home/kafcoppelia/WORK/PAIRV/Makefile) as the top-level build entry
- Current state:
  - the root Makefile is functionally broken by default
  - [Makefile](/home/kafcoppelia/WORK/PAIRV/Makefile#L1) still sets `PROGRAM :=application/helloworld`, but the real app path is under `application/baremetal/helloworld`
  - because program validation runs unconditionally at parse time in [Makefile](/home/kafcoppelia/WORK/PAIRV/Makefile#L40), even `help`, `__help`, `buildall`, and `cleanall` fail before target dispatch
  - verified:
    - `make -C ../PAIRV info` fails at [Makefile](/home/kafcoppelia/WORK/PAIRV/Makefile#L45)
    - `make -C ../PAIRV help` fails the same way
    - `make -C ../PAIRV buildall -n` and `cleanall -n` also fail the same way
- Root entrypoint usage today:
  - the README uses the root Makefile extensively, with many examples like `make PROGRAM=application/helloworld all`
  - those examples are also stale because the app tree is now `application/baremetal/helloworld`, `application/baremetal/nice`, and `application/baremetal/uart`
  - there is no evidence in the primary SDK source tree that other automation depends on the current complex root Makefile behavior beyond documentation expectations
- Recommendation:
  - keep the root Makefile, but only as a thin dispatcher
  - deleting it would be possible because each app has its own Makefile, but it would make the repo less convenient as a CLI SDK and would force all documentation to switch to `cd application/... && make ...`
  - the simpler and better outcome is to retain a root entrypoint while removing the obsolete complexity
- Concrete simplification set if retained:
  - `1.` Fix the default path:
    - change the default `PROGRAM` to a real app path such as `application/baremetal/helloworld`, or remove the default entirely and require explicit `PROGRAM`
  - `2.` Stop validating `PROGRAM` for unrelated targets:
    - `help`, `__help`, `buildall`, `cleanall`, `tags`, and `ctags` should not depend on a valid single-app `PROGRAM`
  - `3.` Drop the legacy shorthand fallback:
    - [Makefile](/home/kafcoppelia/WORK/PAIRV/Makefile#L41) tries `application/$(PROGRAM)` as a compatibility path; that path model no longer matches the current tree and mostly adds confusion
  - `4.` Simplify app discovery:
    - the recursive wildcard/search-pattern machinery in [Makefile](/home/kafcoppelia/WORK/PAIRV/Makefile#L22) is heavier than needed for the current fixed app tree
    - if `buildall/cleanall` only need the current first-party apps, discovery can be reduced to the actual `application/**/Makefile` list
  - `5.` Reassess rarely used extras:
    - `EXTRA_APP_ROOTDIRS` and `PARALLEL` may still be fine, but there is no evidence in the primary tree that they are currently used
    - `tags/ctags` are unrelated to build dispatch and could be dropped from the top-level Makefile if you want it focused
  - `6.` Update docs together:
    - [README.md](/home/kafcoppelia/WORK/PAIRV/README.md#L178) and [Makefile.rules](/home/kafcoppelia/WORK/PAIRV/Build/Makefile.rules#L86) still point to old app paths like `application/helloworld`
- Final judgment:
  - yes, there is still a reason to keep the root Makefile
  - but only if it is cut down into a small, reliable root dispatcher; the current version is more misleading than helpful until that cleanup happens

# PAIRV Root Makefile Minimal Cleanup

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Keep this task scoped to `../PAIRV/Makefile`, related build help/docs, and `tasks/**`.
- [x] Do not create a dedicated `git worktree` because this is a small focused cleanup of the root dispatcher and its immediate user-facing docs.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `../PAIRV/Makefile`, `../PAIRV/README.md`, `../PAIRV/Build/Makefile.rules`, `tasks/**`
- Blocked Files: broader SoC rename candidates and unrelated dirty files
- Dependencies: current app layout under `../PAIRV/application/**`
- Verification: `make -C ../PAIRV help`, `info`, `buildall -n`, `cleanall -n`, and one single-app root dispatch check

## Plan

- [x] Fix the root Makefile default path and target-specific validation behavior.
- [x] Update help/documentation examples to the real `application/baremetal/...` layout.
- [x] Run focused root-entry verification and record the outcome.

## Review

- Root dispatcher cleanup in [Makefile](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/Makefile):
  - changed the default `PROGRAM` to the real path `application/baremetal/helloworld`
  - added `.DEFAULT_GOAL := help` so plain `make` lands on a safe root help target
  - split forwarded app targets into an explicit `APP_TARGETS` list instead of treating `help` as an app-dispatch target
  - removed the stale fallback that tried `application/$(PROGRAM)`
  - limited `PROGRAM` validation to actual single-app dispatch targets, so root-only targets no longer fail during parse
  - switched recursive invocations from raw `make` to `$(MAKE)`
- Help/doc sync:
  - updated [README.md](/home/kafcoppelia/WORK/PAIRV/README.md) examples and app labels from `application/helloworld`, `application/nice`, and `application/uart` to the real `application/baremetal/...` layout
  - refreshed the top project layout to show `application/baremetal/` and `application/freertos/`
  - updated the application-level build help example in [Makefile.rules](/home/kafcoppelia/WORK/PAIRV/Build/Makefile.rules#L86) to `application/baremetal/helloworld`
- Verification:
  - `make -C ../PAIRV help`
    - result: passed; root help now prints without requiring a valid single-app `PROGRAM`
  - `make -C ../PAIRV info`
    - result: passed; root dispatcher forwards to `application/baremetal/helloworld`
  - `make -C ../PAIRV cleanall -n`
    - result: passed; root bulk-clean dispatch now expands across all discovered apps
  - `make -C ../PAIRV buildall -n`
    - result: root bulk-build dispatch now works far enough to traverse the app list, but it stops in `application/baremetal/uart` on a pre-existing app-level dependency error: `No rule to make target '../..//SoC/evalsoc/Common/Include/nuclei_sdk_soc.h', needed by 'main.c.o'`
    - conclusion: the remaining `buildall` failure is no longer caused by the root Makefile itself

# PAIRV tags Target And UART Build Diagnosis

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Keep this task review-only and scoped to `../PAIRV/Makefile`, `../PAIRV/application/baremetal/uart/**`, related generated dependency artifacts, and `tasks/**`.
- [x] Do not create a dedicated `git worktree` because this phase is diagnosis and recommendation only.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `../PAIRV/Makefile`, `../PAIRV/application/baremetal/uart/**`, `tasks/**`
- Blocked Files: unrelated SDK files outside the immediate review scope
- Dependencies: current root dispatcher behavior and the uart app's local dependency files
- Verification: inspect the actual `tags` recipe and reproduce/trace the current uart build failure to a concrete cause

## Plan

- [x] Inspect `tags/ctags` behavior and decide whether it is still worth keeping.
- [x] Trace the uart build failure from the observed make error to the underlying broken path/dependency state.
- [x] Summarize recommendations and whether any follow-up fix is warranted.

## Review

- `tags` / `ctags` purpose in the root [Makefile](/home/kafcoppelia/WORK/PAIRV/Makefile#L63):
  - the recipe runs `ctags -o tags \`find . -name '\*.[chS]' -print\``and then symlinks`ctags -> tags`
  - this generates an editor navigation index for tools such as Vim/Emacs/older IDE integrations so you can jump to symbol definitions quickly
  - it is not part of the compile, link, upload, or debug flow at all
  - retention judgment:
    - useful only as an optional developer convenience
    - not necessary for the SDK build system
    - if the goal is to keep the root Makefile focused on build entry only, it is safe to remove
    - if someone on the team still uses tag-based navigation, it can stay because it is isolated and low-risk
- `uart` build issue:
  - the app Makefile itself is normal and matches the other baremetal apps: [uart/Makefile](/home/kafcoppelia/WORK/PAIRV/application/baremetal/uart/Makefile)
  - the observed error is driven by a stale generated dependency file, not by the current source include statements
  - concrete evidence:
    - [main.c.o.d](/home/kafcoppelia/WORK/PAIRV/application/baremetal/uart/main.c.o.d) contains dependencies like `../..//SoC/evalsoc/Common/Include/nuclei_sdk_soc.h`
    - from the current directory `application/baremetal/uart`, that relative path is wrong; it resolves under `application/SoC/...` instead of repo-root `SoC/...`
    - current compiler flags are correct and use `-I../../../SoC/evalsoc/...`, as shown by `make -C ../PAIRV/application/baremetal/uart showflags`
    - only `main.c.o.d` is poisoned this way; the other dependency files are simple and fine:
      - [config_frame.c.o.d](/home/kafcoppelia/WORK/PAIRV/application/baremetal/uart/config_frame.c.o.d)
      - [input_frame.c.o.d](/home/kafcoppelia/WORK/PAIRV/application/baremetal/uart/input_frame.c.o.d)
  - likely cause:
    - inference from paths: `../..//SoC/...` would have been correct when the app lived one directory shallower, i.e. under `application/uart`
    - this matches the repo’s earlier stale root examples that also still referred to `application/uart`
    - so the most plausible explanation is that `main.c.o.d` was generated before the app tree moved to `application/baremetal/uart`
  - practical consequence:
    - `make -n all` in `uart` fails before rebuilding because GNU Make includes the stale `.d` file and then tries to satisfy its nonexistent prerequisite
    - the same stale file was what made root `buildall -n` trip on `uart`
    - a real clean rebuild should remove the stale `.d` and regenerate a correct one
  - separate environment issue also observed:
    - an actual `make -C ../PAIRV/application/baremetal/uart all` in the current shell fails earlier on missing toolchain executable `riscv64-unknown-elf-gcc`
    - so today there are two distinct blockers:
      - stale `main.c.o.d` affecting dry-run / incremental make parsing
      - toolchain environment not loaded in the current shell for a true rebuild
  - `./.venv/bin/pytest tests/paiir/pipeline/test_graph_simulation.py -q -k 'view_and_view_as_routing_before_linear or function_unsqueeze_before_linear or tuple_repeat_all_ones_before_linear or method_squeeze_before_linear or function_squeeze_before_linear'`
    - result: `5 passed`
  - `./.venv/bin/pytest tests/paiir -q`
    - result: `567 passed, 2 skipped`

# reshape_semantics Shape Constraint Narrowing

## Workspace Decision

- [x] Continue in the current workspace on `feat/layout-two-pass-canonicalization`.
- [x] Keep this scope limited to shape helper signatures and their direct call sites/tests; `dims` stays unchanged as `tuple[int, ...]`.

## Plan

- [x] Narrow `reshape_semantics.py` shape helper inputs from transitional `ShapeLike` to `torch.Size` where production callers already satisfy the stricter contract.
- [x] Align nearby direct call sites and focused tests to pass `torch.Size` explicitly.
- [x] Re-run focused regressions to confirm the tighter shape contract does not change behavior.

## Review

- Further narrowed shape helper inputs in [reshape_semantics.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/reshape_semantics.py):
  - `is_layout_invisible_dims(...)`
  - `canonicalize_layout_view(...)`
  - `shape_after_dims(...)`
  - `reshape_output_shape(...)`
  - `is_layout_invisible_reshape(...)`
- Removed the transitional `ShapeLike` alias because the remaining production callers now pass `torch.Size`.
- Aligned direct callers and local defaults:
  - [layout_chain_canonicalization.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/layout_chain_canonicalization.py)
    - `_compose_shape_fns(...)` now keeps `current` as `torch.Size` all the way through
  - [converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/converter.py)
    - `_create_output_nodes(...)` now uses `torch.Size()` instead of `()` for non-FX outputs
- Updated focused tests to pass `torch.Size` explicitly where they exercise the narrowed helpers:
  - [test_reshape_semantics.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/ir/test_reshape_semantics.py)
  - [test_layout_chain_canonicalization.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_layout_chain_canonicalization.py)
  - [test_layout_cross_node_elision.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_layout_cross_node_elision.py)
- Focused verification:
  - `./.venv/bin/pytest tests/paiir/ir/test_reshape_semantics.py tests/paiir/pipeline/test_layout_chain_canonicalization.py tests/paiir/pipeline/test_layout_cross_node_elision.py tests/paiir/lowering/test_shape_analysis.py -q`
    - result: `22 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k 'transpose_then_flatten_before_linear_compiles or permute_then_reshape_before_linear_compiles or maxpool_after_reshape_compiles'`
    - result: `3 passed`
  - `./.venv/bin/pytest tests/user/test_widerface_t1_backbone_sz160_paiir.py -q`
    - result: `2 passed`

# PAIIR Meta Device Review

## Workspace Decision

- [x] Continue in the current workspace on `feat/layout-two-pass-canonicalization`.
- [x] Keep this task review-only: inspect `paibox/paiir/**`, run local compatibility probes, and avoid product-code edits until a concrete rollout path is chosen.

## Plan

- [x] Audit the current PAIIR lowering / compile flow to find phases that only need tensor metadata rather than real values.
- [x] Verify locally how current `torch_to_paiir()` / `compile_to_paiir()` behave with meta tensors and FX fake-mode shape propagation.
- [x] Summarize where meta-style execution can help, where it will currently break, and what an incremental adoption path should look like.

## Review

- Current shape inference is fully driven by `torch.fx.passes.shape_prop.ShapeProp` in [converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/converter.py), so the natural optimization point is compile-time metadata propagation rather than later simulation/export passes.
- `torch_to_paiir()` currently calls plain `ShapeProp(gm).propagate(*inputs)` and then `DimsProp`, which means it executes the traced graph with real tensors unless we explicitly switch this stage to abstract execution.
- A direct "feed meta sample inputs into the current pipeline" approach is not sufficient:
  - it can work for some simple operator subsets
  - but it fails on mixed-device math such as `Linear` when activations become meta while parameters stay on CPU
  - making the whole model meta avoids that mismatch, but then later compile passes fail because they need real weight values
- The hard blocker for whole-pipeline meta execution is not only weight access:
  - weight-format inference reads numeric min/max via `tensor.min().item()` / `tensor.max().item()` in [op_node.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/op_node.py) and [passes.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/passes.py)
  - PAIIR native neuron modules also call `.item()` during forward state updates in [core_neuron.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/core_neuron.py), which breaks meta/fake execution for those modules
- Final decision:
  - do not implement a fake/meta path for `_propagate_shapes(...)`
  - the narrower fake-mode idea still requires exclusions or fallback for common project modules such as `CoreNeuronV25`, `ANNNodeV25`, and custom registered neurons, which makes it a poor fit for the normal PAIIR compile path
  - even if added, it would only optimize the metadata-propagation slice and would not remove the real-weight/value dependencies in later compile passes
- Replacement focus:
  - profile and optimize the existing real compile path, especially `torch_to_paiir()`
  - avoid repeated lowering/tracing of the same FX graph in user-facing diagnostic paths
  - treat fake/meta execution as out of scope unless it can support common project modules without operator-specific exclusion lists

# PAIIR Compile Hotspot Audit

## Workspace Decision

- [x] Continue in the current workspace on `feat/layout-two-pass-canonicalization`.
- [x] Keep this scope on profiling and hotspot documentation only; do not change compile behavior while the direction is still being narrowed.

## Plan

- [x] Measure which top-level compile stages dominate representative ANN and SNN compilation.
- [x] Break down `torch_to_paiir()` to distinguish FX trace, erase transform, shape propagation, dims propagation, and FX-to-PAIIR lowering costs.
- [x] Check the current user-facing WiderFace helper path for repeated lowering/tracing of the same prefix graph.

## Review

- Representative compile timing shows the optimization target is the existing real `torch_to_paiir()` path, not the later passes:
  - `ANNClassifier`: `torch_to_paiir` averaged `55.105 ms` of `55.765 ms` total (`98.8%`)
  - `SNNTwoLayer`: `torch_to_paiir` averaged `15.526 ms` of `16.112 ms` total (`96.4%`)
- Breaking down `torch_to_paiir()` shows `ShapeProp` is the dominant hotspot on smaller models:
  - `ANNClassifier`: `shape_prop` `51.277 ms` (`87.9%`), `erase_transform` `3.016 ms`, `fx_to_paiir` `2.361 ms`, `fx_trace` `1.661 ms`
  - `SNNTwoLayer`: `shape_prop` `10.537 ms` (`70.9%`), `erase_transform` `2.151 ms`, `fx_trace` `1.361 ms`, `fx_to_paiir` `0.799 ms`
- On the real WiderFace `input -> p3` prefix, the whole lowering path remains concentrated before fusion, but not only in shape propagation:
  - warm runs were roughly `126.677 ms` to `140.609 ms`
  - `erase_transform` stayed around `40 ms`
  - `shape_prop` stayed around `28 ms` to `34 ms`
  - `fx_to_paiir` stayed around `43 ms` to `46 ms`
  - this confirms that even a perfect fake/meta replacement for `ShapeProp` would still leave substantial real compile cost untouched
- The current WiderFace diagnostic path repeats lowering work on the same prefix graph:
  - [test_widerface_t1_backbone_sz160_paiir.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/test_widerface_t1_backbone_sz160_paiir.py) calls `attempt_compile(...)` twice (`strict=True` and `strict=False`) and `get_prevalidation_fused_graph(...)` once
  - those helpers each rebuild sample input and run `torch_to_paiir(...)`, so one report generation can lower the same prefix model three times before later summaries are rendered
- Recommended next optimization targets:
  - cache or share the lowered unfused graph in user-facing diagnostic/report helpers when strictness or later-pass inspection changes but the traced prefix model and sample shape do not
  - profile `_EraseModuleTransformer` and `_fx_graph_to_paiir(...)` next on larger customer graphs, because they remain material costs after `ShapeProp`

# IR Shape Metadata torch.Size Alignment

## Workspace Decision

- [x] Continue in the current workspace on `feat/layout-two-pass-canonicalization`.
- [x] Keep the scope on PAIIR IR shape metadata and its direct producers/consumers; this is a cohesive refactor inside `paibox/paiir/**` and focused `tests/paiir/**`.

## Plan

- [x] Audit which PAIIR metadata fields represent true tensor shapes and should move to `torch.Size`.
- [x] Update the first batch of IR shape storage and direct helper functions while leaving `dims` as `tuple[int, ...]`.
- [x] Run focused IR/lowering/pipeline/user regressions to confirm the type alignment does not change behavior.

## Review

- First-batch `torch.Size` alignment completed for long-lived PAIIR shape metadata:
  - [ir_base.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/ir_base.py)
    - `InputNode.shape`
    - `OutputNode.shape`
  - [op_node.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/op_node.py)
    - `OpNode.input_shapes`
    - `OpNode.output_shape`
  - [shape_analysis.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/shape_analysis.py)
    - `ReshapeSinkInfo.output_shape`
    - `_extract_tensor_output_shape(...)`
  - [converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/converter.py)
    - `_get_output_shape(...)`
    - `_get_input_shapes(...)`
    - `_make_fixed_shape_fn(...)`
    - `_build_reshape_op(...)`
  - [passes.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/passes.py)
    - `_get_node_output_shape(...)`
  - [add_ops.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/add_ops.py)
    - `_constant_shape(...)`
    - `GeneralAddOp.operand_shape(...)`
- Deliberately unchanged:
  - `dims` metadata still uses `tuple[int, ...]`
  - [reshape_semantics.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/reshape_semantics.py) still accepts shape-like inputs during the transition, instead of forcing an all-at-once pure-`torch.Size` migration
- One compile regression expectation changed:
  - [test_compile.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_compile.py)
  - `PoolAfterReshape` now compiles to `Input -> MaxPool -> Output` without a residual `ReshapeOp`, which matches the current identity-reshape canonicalization behavior
- Focused verification:
  - `./.venv/bin/pytest tests/paiir/ir/test_reshape_semantics.py tests/paiir/lowering/test_shape_analysis.py tests/paiir/pipeline/test_layout_chain_canonicalization.py tests/paiir/pipeline/test_layout_cross_node_elision.py -q`
    - result: `22 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k 'transpose_then_flatten_before_linear_compiles or permute_then_reshape_before_linear_compiles or maxpool_after_reshape_compiles'`
    - result: `3 passed`
  - `./.venv/bin/pytest tests/user/test_widerface_t1_backbone_sz160_paiir.py -q`
    - result: `2 passed`

# Dims Optimization Refinement

## Workspace Decision

- [x] Continue in the current workspace on `feat/layout-two-pass-canonicalization`.
- [x] Keep the scope on shared dims/layout semantics plus focused PAIIR tests, because this work extends the just-added layout canonicalization path and does not require a separate worktree.

## Plan

- [x] Audit the current dims-driven layout handling to identify safe, chip-like optimizations.
- [x] Implement a shared helper for layout-invisible dims normalization and use it in simulation/layout code where it is semantically safe.
- [x] Add focused tests for the new dims behavior and rerun the relevant compile/user regressions.

## Review

- Added a shared dims helper in [reshape_semantics.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/reshape_semantics.py):
  - `canonicalize_layout_view(...)`
  - It recognizes layout-invisible dims permutations that only move singleton axes and rewrites them into an equivalent identity-dims logical view.
- Updated `materialize_logical_layout(...)` to use that helper:
  - singleton-axis-only dims are now materialized as a pure `reshape(...)`
  - non-singleton permutations still materialize through `permute(...)`
- Kept graph structure unchanged:
  - this round improves dims handling in shared semantics and simulation rather than adding another graph rewrite rule
- Added focused IR tests in [test_reshape_semantics.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/ir/test_reshape_semantics.py) covering:
  - canonicalization of singleton-axis layout views
  - preservation of real non-singleton permutations
  - runtime materialization equivalence against PyTorch `permute(...)`
- Focused verification:
  - `./.venv/bin/pytest tests/paiir/ir/test_reshape_semantics.py tests/paiir/pipeline/test_layout_chain_canonicalization.py -q`
    - result: `9 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k 'transpose_then_flatten_before_linear_compiles or permute_then_reshape_before_linear_compiles'`
    - result: `2 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_graph_simulation.py -q -k 'transpose_then_flatten_before_linear or permute_then_reshape_before_linear'`
    - result: `2 passed`
  - `./.venv/bin/pytest tests/user/test_widerface_t1_backbone_sz160_paiir.py -q`
    - result: `2 passed`
- User-facing effect:
  - the `input -> p3` customer log remains stable at `50` nodes / `55` edges
  - no regression in the existing PAIIRGraph summary artifact generation path

## Workspace Decision

- [x] Stay in the current workspace on `feat-paiir-lowering-cleanup`.
- [x] Do not create a dedicated `git worktree` for this task because the requested work is branch maintenance on the currently checked out branch and must preserve the exact in-place working state before rebasing.

## Ownership

- [x] Owner: Codex
- [x] Branch: `feat-paiir-lowering-cleanup`
- [x] Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- [x] Allowed Files: Git metadata plus `tasks/todo.md`
- [x] Blocked Files: Product code is not being edited unless conflict resolution during rebase requires it
- [x] Dependencies: `origin/dev` must be fetched before rebasing
- [x] Verification: confirm clean post-rebase status, confirm branch/upstream relation, then push the rebased branch to `origin`

## Plan

- [x] Inspect the current branch and working tree state before changing Git history.
- [x] Stash the current workspace, including untracked files, so the rebase starts from a clean tree.
- [x] Fetch the latest `origin/dev` and rebase `feat-paiir-lowering-cleanup` onto it.
- [x] Resolve any rebase conflicts and verify the resulting repository state.
- [x] Push the rebased current branch to `origin`.

## Review

- Fetched `origin/dev` and confirmed it advanced from `5e78cd5` to `7ddc735`.
- Rebasing `feat-paiir-lowering-cleanup` onto `origin/dev` completed successfully with no commit conflicts.
- Pushed the rebased branch with `git push --force-with-lease origin feat-paiir-lowering-cleanup`.
- The remote branch now matches local `HEAD`, and `git status --short --branch` reports `feat-paiir-lowering-cleanup...origin/feat-paiir-lowering-cleanup` with no ahead/behind divergence.
- Restored the main stashed workspace with `git stash apply stash@{1}` after the push, so the pre-rebase local edits are back in the working tree.

# PAIIRGraph API Semantics Review

## Workspace Decision

- [x] Stay in the current workspace on `feat-paiir-lowering-cleanup`.
- [x] Keep this task review-only because the request is about API semantics and naming, while the current workspace already contains in-flight product edits.

## Ownership

- [x] Owner: Codex
- [x] Branch: `feat-paiir-lowering-cleanup`
- [x] Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- [x] Allowed Files: `tasks/todo.md`
- [x] Blocked Files: `paibox/**`, `tests/**`, and other in-flight product files stay read-only for this review
- [x] Dependencies: inspect the graph API definitions plus their call sites before recommending any signature change
- [x] Verification: confirm the recommendation against current method bodies and all in-repo usages

## Plan

- [x] Inspect `PAIIRGraph.replace_all_uses_with` and `PAIIRGraph.remove_node_and_reconnect` definitions and invariants.
- [x] Audit call sites to infer intended semantics, especially whether callers treat them as mutators, query helpers, or transformation primitives.
- [x] Recommend suitable return values, `delete_old` default behavior, and whether `remove_node_and_reconnect` should be renamed for semantic precision.

## Review

- Current behavior in [graph.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/graph.py#L183) and [graph.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/graph.py#L220):
  - `replace_all_uses_with()` rewrites every outgoing edge from `old_name` to `new_name`, deduplicates identical resulting edges, and currently returns the number of original outgoing edges it rewrote.
  - `remove_node_and_reconnect()` snapshots outgoing edges, removes the node, reconnects one chosen source to each outgoing destination when needed, and currently returns the number of new unique edges added after deduplication.
- Call-site observations:
  - Production call sites in [layout_cross_node_elision.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/layout_cross_node_elision.py#L84) and [layout_chain_canonicalization.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/layout_chain_canonicalization.py#L57) ignore the return values entirely.
  - Only tests in [test_graph.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/ir/test_graph.py#L83) assert the counts.
- Recommendation:
  - Preferred: make both methods return `None`, because they are graph mutators like `add_node`, `remove_node`, `replace_node`, and `add_edge`; success/failure is already expressed by exceptions.
  - If rewrite statistics are still desired, use one shared semantic for both methods, ideally “number of outgoing uses affected before deduplication”, not one method returning rewired uses and the other returning only newly inserted unique edges.
  - Keep `delete_old=False` by default. `replace_all_uses_with` conventionally means “redirect uses”, not “erase the producer”, and the current pipeline already separates rewiring from deletion.
  - The `delete_old` knob is acceptable as an explicit convenience, but it should stay opt-in; otherwise the name becomes misleading and existing two-step rewrite patterns become fragile.
  - `remove_node_and_reconnect` is only a good name if `source_name` is guaranteed to be a predecessor. Today the implementation accepts any existing node, so either validate `source_name in predecessors(name)` or rename the API to match the broader behavior, e.g. `remove_node_and_redirect_uses` / `remove_node_and_replace_source`.
- Stash restore reported one untracked-file warning because `.codex` already existed at apply time; the user code changes and untracked worktree contents were restored, and the stash entries were intentionally kept for safety.

# PAIIR Test Tracer Helper Consolidation

# PAIIRGraph Rewrite Primitive Refinement

## Workspace Decision

- [x] Continue in the current workspace on `feat/layout-two-pass-canonicalization`.
- [x] Keep the scope on `PAIIRGraph` rewrite APIs plus the existing layout passes that currently perform manual edge rewiring.

## Plan

# PAIRV Root Convenience And NICE Download Constraint Cleanup

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Keep this task scoped to the root dispatcher/docs, `application/baremetal/nice/Makefile`, `application/baremetal/uart` generated artifacts, and `tasks/**`.
- [x] Do not create a dedicated `git worktree` because this is a small focused cleanup with localized verification.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `../PAIRV/Makefile`, `../PAIRV/README.md`, `../PAIRV/application/baremetal/nice/Makefile`, `../PAIRV/application/baremetal/uart/**`, `tasks/**`
- Blocked Files: unrelated SDK files
- Dependencies: current root Makefile behavior, the NICE download-mode requirement, and the stale uart dependency artifacts
- Verification: confirm the root help surface, NICE mode enforcement, and removal of the stale uart `.o/.d` blockers

## Plan

- [x] Remove `tags/ctags` from the root build entry surface.
- [x] Encode the NICE `DOWNLOAD=ilmflashxip` requirement in the app Makefile and docs.
- [x] Remove stale uart `.o/.d` artifacts that were poisoning make dependency parsing.
- [x] Run focused verification and record the outcome.

## Review

- Root dispatcher cleanup:
  - removed `tags` / `ctags` from the root [Makefile](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/Makefile) so the top-level entry stays focused on build/debug dispatch
- NICE download-mode enforcement:
  - updated [application/baremetal/nice/Makefile](/home/kafcoppelia/WORK/PAIRV/application/baremetal/nice/Makefile) to default `DOWNLOAD` to `ilmflashxip`
  - added a hard error if `DOWNLOAD` is anything other than `ilmflashxip`
  - updated [README.md](/home/kafcoppelia/WORK/PAIRV/README.md) examples and notes so NICE uses `DOWNLOAD=ilmflashxip`
- UART stale artifact cleanup:
  - removed the generated `main.c.o`, `main.c.o.d`, `config_frame.c.o`, `config_frame.c.o.d`, `input_frame.c.o`, and `input_frame.c.o.d` from `application/baremetal/uart/`
  - this cleared the old dependency cache that still referenced the pre-move path `../..//SoC/...`
- Verification:
  - `make -C ../PAIRV help`
    - result: passed; root help prints and keeps the NICE example on `ilmflashxip`
  - `make -C ../PAIRV PROGRAM=application/baremetal/nice info`
    - result: passed; effective `DOWNLOAD=ilmflashxip`
  - `make -C ../PAIRV PROGRAM=application/baremetal/nice DOWNLOAD=ilm info`
    - result: fails intentionally with the new guard message
  - `find ../PAIRV/application/baremetal/uart -maxdepth 1 \( -name '*.o' -o -name '*.d' \) | sort`
    - result: no stale local `.o/.d` files remain
  - `make -C ../PAIRV/application/baremetal/uart -n all`
    - result: passed; the old bad dependency path no longer blocks parsing
  - `make -C ../PAIRV buildall -n`
    - result: passed; root bulk-build dry-run now traverses all apps, and NICE is emitted with `DOWNLOAD=ilmflashxip`

# PAIRV README Rewrite And docs Review

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Keep this task scoped to the primary SDK source tree under `../PAIRV/application/**`, `../PAIRV/Build/**`, `../PAIRV/NMSIS/**`, `../PAIRV/OS/**`, `../PAIRV/SoC/**`, root project files, `../PAIRV/docs/**`, and `tasks/**`.
- [x] Do not create a dedicated `git worktree` because this is a focused documentation cleanup plus possible directory removal.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `../PAIRV/README.md`, `../PAIRV/docs/**`, `tasks/**`
- Blocked Files: unrelated product code outside the documentation scope
- Dependencies: actual active build flow and current references into `docs/`
- Verification: ensure the rewritten README matches the live CLI build flow and that removing `docs/` does not leave needed references behind

## Plan

- [x] Inspect the active SDK build directories plus current `docs/` contents and references.
- [x] Rewrite `README.md` in English around the real project functionality and usage.
- [x] Remove `docs/` if it is redundant and verify no important root references remain.

## Review

- `docs/` evaluation:
  - `../PAIRV/docs/` only contained one file: `custom_soc_guide.md`
  - that file was not part of the primary CLI build flow and had no required reference path from the active build directories
  - it focused on a broader/custom SoC cloning workflow rather than the current repo’s main usage
  - based on that, it was removed and the empty `docs/` directory was deleted
- README rewrite:
  - rewrote [README.md](/home/kafcoppelia/WORK/PAIRV/README.md) in English around the live build flow only
  - the new README now centers on:
    - the active build directories: `application`, `Build`, `NMSIS`, `OS`, `SoC`
    - the root `Makefile` as the CLI entrypoint
    - the current app inventory under `application/baremetal/**` and `application/freertos/**`
    - environment setup
    - root build/debug usage
    - run-mode usage
    - the NICE `DOWNLOAD=ilmflashxip` constraint
  - removed the earlier review-style content and stale narrative that mixed diagnostic notes into user-facing documentation
- Verification:
  - `rg -n "/home/|docs/|custom_soc_guide" ../PAIRV/README.md ../PAIRV/Makefile ../PAIRV/application ../PAIRV/Build ../PAIRV/NMSIS ../PAIRV/OS ../PAIRV/SoC`
    - result: no remaining README or active-build references to the removed docs content
  - `find ../PAIRV/docs -maxdepth 2 -type f | sort`
    - result: `docs/` no longer exists
  - `make -C ../PAIRV help`
    - result: passed after the README/docs cleanup

# PAIRV Application Help Text Cleanup

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Keep this task scoped to `../PAIRV/Build/Makefile.rules` and `tasks/**`.
- [x] Do not create a dedicated `git worktree` because this is a small user-facing help text cleanup.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `../PAIRV/Build/Makefile.rules`, `tasks/**`
- Blocked Files: unrelated build logic and product code
- Dependencies: current root README and real application layout under `application/baremetal/**`
- Verification: check application-level `make help` output after the text update

## Plan

- [x] Inspect the current `Build/Makefile.rules` help block against the live build flow.
- [x] Update the help strings to reflect the actual app paths and current usage expectations.
- [x] Verify the help output from representative app directories.

## Review

- Updated the application-level help text in [Build/Makefile.rules](/home/kafcoppelia/WORK/PAIRV/Build/Makefile.rules#L57) so it now describes the current PAIRV build flow instead of the old generic Nuclei SDK wording.
- Added [Build/Makefile.rules](/home/kafcoppelia/WORK/PAIRV/Build/Makefile.rules#L30) `APP_REL_DIR` so the help output can show the current application path and generate matching top-level `PROGRAM=...` examples automatically.
- Removed the misleading old command forms such as `all [PROGRAM=flash/flashxip/ilm/ddr]` from the app-level help.
- Kept the help generic where it should stay generic, but made the example `DOWNLOAD` value dynamic through [Build/Makefile.rules](/home/kafcoppelia/WORK/PAIRV/Build/Makefile.rules#L88), so:
  - normal apps show `make DOWNLOAD=ilm all`
  - `nice` shows `make DOWNLOAD=ilmflashxip all`
- Verification:
  - `make -C ../PAIRV/application/baremetal/helloworld help`
    - result: passed; help shows `application/baremetal/helloworld` and `DOWNLOAD=ilm`
  - `make -C ../PAIRV/application/baremetal/nice help`
    - result: passed; help shows `application/baremetal/nice` and `DOWNLOAD=ilmflashxip`
  - `make -C ../PAIRV/application/freertos/demo help`
    - result: passed; help shows `application/freertos/demo`

- [x] Analyze current graph mutation patterns and choose graph-level rewrite primitives over FX-style insertion-point APIs.
- [x] Add the chosen `PAIIRGraph` helpers and direct graph tests.
- [x] Refactor the layout passes to use those helpers.
- [x] Re-run focused graph/layout/compile/user regressions.

## Review

- Current `PAIIRGraph` mutation patterns are edge-centric, not insertion-point-centric:
  - build-time `add_node/add_edge`
  - node replacement via `replace_node`
  - local rewiring in the layout passes
- Based on that usage, implemented graph-level rewrite primitives instead of `torch.fx`-style `inserting_before/inserting_after`:
  - [graph.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/graph.py)
    - `add_edge_if_missing(...)`
    - `replace_all_uses_with(...)`
    - `remove_node_and_reconnect(...)`
- Rationale:
  - current rewrites operate on uses/edges, not node order
  - `PAIIRGraph` does not have FX-style ordered-node insertion semantics
  - graph-level rewiring helpers directly match the actual pass implementation needs
- Refactored existing passes to use the new helpers:
  - [layout_chain_canonicalization.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/layout_chain_canonicalization.py)
  - [layout_cross_node_elision.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/layout_cross_node_elision.py)
- Added direct graph API tests in:
  - [test_graph.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/ir/test_graph.py)
- Focused verification:
  - `./.venv/bin/pytest tests/paiir/ir/test_graph.py tests/paiir/pipeline/test_layout_chain_canonicalization.py tests/paiir/pipeline/test_layout_cross_node_elision.py -q`
    - result: `18 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k 'unsqueeze_repeat_all_ones_compile_path_succeeds or shape_only_reshape_args_do_not_become_data_predecessors or transpose_then_flatten_before_linear_compiles or permute_then_reshape_before_linear_compiles'`
    - result: `4 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_graph_simulation.py -q -k 'flatten or reshape'`
    - result: `4 passed`
  - `./.venv/bin/pytest tests/user/test_widerface_t1_backbone_sz160_paiir.py -q`
    - result: `2 passed`

# PAIIRGraph Rewrite Primitives

## Workspace Decision

- [x] Continue in the current workspace on `feat/layout-two-pass-canonicalization`.
- [x] Keep the scope limited to `paibox/paiir/ir/graph.py`, focused IR graph tests, and the existing layout rewrite passes that currently perform manual edge rewiring.

## Plan

- [x] Review how current code mutates `PAIIRGraph` and choose graph-level rewrite primitives instead of FX-style insertion-point APIs.
- [ ] Add the chosen `PAIIRGraph` rewrite helpers and test them directly.
- [ ] Refactor the existing layout rewrite passes to use those helpers.
- [ ] Re-run focused graph/layout/user regressions and record the result.

# Two-Pass Layout Canonicalization

## Workspace Decision

- [x] Implemented on the dedicated branch `feat/layout-two-pass-canonicalization`.
- [x] Kept the work in the current workspace because the change set is confined to `paibox/paiir/**`, `tests/paiir/**`, and the existing `tests/user/**` log regression.

## Plan

- [x] Add a dedicated chain-level layout canonicalization pass before add specialization.
- [x] Add a dedicated cross-node layout-invisible reshape elision pass before offline-core fusion.
- [x] Integrate both passes into the real compile pipeline and the user-facing `input -> p3` log path.
- [x] Add focused tests for both passes and rerun compile/log regressions.

## Review

- Added the two dedicated pass modules:
  - [layout_chain_canonicalization.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/layout_chain_canonicalization.py)
  - [layout_cross_node_elision.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/layout_cross_node_elision.py)
- First pass behavior:
  - collapses consecutive `ReshapeOp` chains by local fixed-point pairwise composition
  - removes single-node identity reshapes
  - supports arbitrary chain length via repeated local collapse
  - current implementation operates on `ReshapeOp` chains in PAIIR IR; explicit `transpose/permute/contiguous` nodes are already absorbed earlier into dims metadata and therefore do not appear as standalone PAIIR nodes to rewrite
- Second pass behavior:
  - elides the safe sandwich pattern `StandaloneCompOp -> ReshapeOp -> StandaloneActOp -> ReshapeOp`
  - only when both reshapes are layout-invisible and the pre-reshape is not shared
  - rewrites the activation to operate directly on the compute node's chip-visible shape, exposing the existing `SequentialOp` fusion
- Integrated into the real compile order in [compile.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/compile.py):
  1. `torch_to_paiir`
  2. `canonicalize_layout_chains`
  3. `specialize_general_adds`
  4. `elide_layout_invisible_reshapes`
  5. `fuse_to_offline_cores`
- Updated shared test helper [tests/paiir/conftest.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/conftest.py) so `convert_and_fuse(...)` follows the same pre-fusion canonicalization path.
- Added focused pass tests:
  - [test_layout_chain_canonicalization.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_layout_chain_canonicalization.py)
  - [test_layout_cross_node_elision.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_layout_cross_node_elision.py)
- Updated existing regressions to match the new canonicalized graph shape:
  - [test_compile.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_compile.py)
  - [test_graph_simulation.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_graph_simulation.py)
  - [test_widerface_t1_backbone_sz160_paiir.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/test_widerface_t1_backbone_sz160_paiir.py)
- User-visible effect on the current `input -> p3` customer prefix:
  - graph shrank from `80` nodes / `85` edges to `50` nodes / `55` edges
  - fused graph now includes `9` `SequentialOp`
  - `ReshapeOp` count dropped to `15`
  - the exported log was regenerated and reflects the new shorter graph
- Known boundary of v1:
  - the cross-node pass does not touch shared pre-reshape nodes, so some `StandaloneCompOp -> ReshapeOp -> StandaloneActOp` patterns remain when that reshape also feeds another consumer (for example a later add path)

## Verification

- `./.venv/bin/pytest tests/paiir/pipeline/test_layout_chain_canonicalization.py tests/paiir/pipeline/test_layout_cross_node_elision.py -q`
  - result: `6 passed`
- `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k 'unsqueeze_repeat_all_ones_compile_path_succeeds or shape_only_reshape_args_do_not_become_data_predecessors or transpose_then_flatten_before_linear_compiles or permute_then_reshape_before_linear_compiles'`
  - result: `4 passed`
- `./.venv/bin/pytest tests/paiir/pipeline/test_graph_simulation.py -q -k 'flatten or reshape'`
  - result: `4 passed`
- `./.venv/bin/pytest tests/user/test_widerface_t1_backbone_sz160_paiir.py -q`
  - result: `2 passed`

# IRNamespace Design Tightening

## Workspace Decision

- [x] Continue in the current workspace on `feat-paiir-lowering-cleanup`.
- [x] Keep this task limited to `paibox/paiir/ir/_namespace.py`, its direct call site in `ir_base.py`, and focused IR tests.

## Plan

- [x] Compare the current `IRNamespace` with `torch.fx.graph._Namespace` and identify the minimum worthwhile improvements.
- [ ] Refine `IRNamespace` to support candidate-driven naming and stricter association/rename semantics without adding unnecessary complexity.
- [ ] Add focused IR tests that lock the expected naming behavior.
- [ ] Run the focused IR tests and record the outcome.

# Fix Stale Node Names In Exported Log

## Workspace Decision

- [x] Continue in the current workspace on `feat-paiir-lowering-cleanup`.
- [x] Keep the scope limited to PAIIR IR naming plus the user-facing log regression, since the reported symptom is a log/node-name inconsistency rather than a compile failure.

## Plan

- [x] Reproduce and explain why `GeneralAddOp_*` names still appear in the exported `.log`.
- [x] Fix the IR namespace implementation so old class prefixes cannot leak into newly created nodes.
- [x] Re-generate the user-facing log and verify that stale `GeneralAddOp_*` names are gone.

## Review

- Root cause: [IRNamespace](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/_namespace.py) used `id(obj)` as the cache key for assigned node names.
- Because Python can reuse freed object ids, a newly created `StandaloneCompOp` could inherit a stale cached name such as `GeneralAddOp_9` from an earlier, already-destroyed `GeneralAddOp` instance.
- That is why the log could contain lines like:
  - `GeneralAddOp_9 (Conv2d)`
    even though the current node was not a `GeneralAddOp` semantically.
- Fixed [IRNamespace](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/_namespace.py) to use a `WeakKeyDictionary` keyed by the live object itself instead of raw `id(obj)`.
- Added a direct user-facing regression in [test_widerface_t1_backbone_sz160_paiir.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/test_widerface_t1_backbone_sz160_paiir.py):
  - the exported log must not contain `GeneralAddOp_`
- Re-generated the log artifact:
  - [widerface_t1_backbone_int8_sz160_paiirgraph.summary.log](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/debug/widerface_t1_backbone_int8_sz160_paiirgraph.summary.log)
- Verified with:
  - `./.venv/bin/pytest tests/user/test_widerface_t1_backbone_sz160_paiir.py -q`
    - result: `2 passed`
  - `rg -n "GeneralAddOp_" tests/user/debug/widerface_t1_backbone_int8_sz160_paiirgraph.summary.log`
    - result: no matches

# PR Draft For weight_skew -> dev

## Workspace Decision

- [x] Continue in the current workspace on `feat-paiir-lowering-cleanup`.
- [x] Keep this task documentation-only: inspect `origin/weight_skew` versus `origin/dev`, then draft a copy-ready PR title and short summary without changing product code.

## Plan

- [x] Confirm the exact relationship between `origin/weight_skew` and `origin/dev` and identify whether there is an active delta or an already-merged change set.
- [x] Summarize the verified branch changes into a concise PR title candidate.
- [x] Draft a short Markdown PR summary that matches the actual diff status.

## Review

- Verified branch relationship:
  - `origin/weight_skew` (`bfb55cd`) is an ancestor of `origin/dev`
  - `origin/dev` contains merge commit `5e78cd5` dated `2026-03-27`
  - the merge commit title is `Feat(backendv2): enable weight skew based weight reuse (#227)`
- Verified change set on the original feature branch path `8a05144..bfb55cd`:
  - feature commit: `e90ecbb` with message `feat: use weight skew`
  - follow-up auto-fix commit: `bfb55cd`
  - touched files are concentrated in `paibox/backendv2/**` plus `paibox/paiir/__init__.py`
- Key change themes confirmed from the merged commit message and diff stat:
  - enable weight-skew-aware weight reuse in `backendv2`
  - centralize raw-weight extraction in new `paibox/backendv2/get_weight.py`
  - carry `ReshapeOp` through reorder-aware routing and destination lookup
  - propagate `add_potential` / input-width semantics into weight packing and core register generation
- Diff status relevant to the user's PR:
  - `origin/weight_skew...origin/dev` shows no remaining file delta
  - if a PR is opened from the current remote `weight_skew` to `dev`, it will likely show no changes because the branch content already exists on `dev`

# Input-to-P3 Prefix Compile

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this scope limited to the `input -> p3` prefix path for the current `sz160` customer pickle, without attempting to solve the later full-backbone concat issues in the same step.

## Plan

- [x] Identify the p3 cut point inside the current backbone-only FX pickle.
- [x] Add the minimal lowering support needed for the prefix path to compile.
- [x] Re-run the `input -> p3` customer prefix compile and regenerate the user-facing log.
- [x] Record whether the prefix subnetwork compiles successfully.

## Review

- Defined the current `p3` cut point as the FX node feeding `backbone.layers.5.encode_lif`, i.e. the output of backbone layer 4:
  - cut node: `add_5`
- Added minimal support for the prefix-only input shape path in:
  - [shape_analysis.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/shape_analysis.py)
  - [dims_prop.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/dims_prop.py)
  - [converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/converter.py)
- The added support is intentionally narrow:
  - `unsqueeze` now lowers as a reshape-like routing op
  - `repeat(...)` lowers as a reshape-like routing op only when every repeat factor is `1`
  - this exactly matches the current `MS_GetT`-style prefix pattern without claiming support for general data-replicating `repeat`
- Updated regression coverage in:
  - [test_compile.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_compile.py)
  - the `FunctionalQuantizedConvWithShapeReshape` path now compiles successfully through `unsqueeze -> repeat(all ones) -> flatten`
- Updated the customer-facing test/log entry:
  - [test_widerface_t1_backbone_sz160_paiir.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/test_widerface_t1_backbone_sz160_paiir.py)
  - it now tests only the `input -> p3` prefix instead of the whole backbone
- Result for the requested `input -> p3` subnetwork:
  - `compile_to_paiir(strict=True)` succeeds
  - `compile_to_paiir(strict=False)` also succeeds
  - compiled graph stats:
    - `80` graph nodes
    - `85` graph edges
    - `78` `OpNode`s
    - `42` deployable `OpNode`s
    - node-type counts: `ReshapeOp=36, StandaloneCompOp=20, StandaloneActOp=16, PotentialAddOp=6`
    - module-type counts: `FunctionalConv2d=20, ANNNodeV25=16`
- Updated log artifact:
  - [widerface_t1_backbone_int8_sz160_paiirgraph.summary.log](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/debug/widerface_t1_backbone_int8_sz160_paiirgraph.summary.log)
  - it now reports the `input -> p3` prefix compile result and graph summary
  - log contents use relative paths so the artifact can be shared without leaking local absolute filesystem paths
- Verified with:
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k 'unsqueeze_repeat_all_ones_compile_path_succeeds or shape_only_reshape_args_do_not_become_data_predecessors'`
    - result: `2 passed`
  - `./.venv/bin/pytest tests/user/test_widerface_t1_backbone_sz160_paiir.py -q`
    - result: `2 passed in 3.15s`

# Resolve MaxPool Predecessor Format

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep the implementation narrowly scoped to `paibox/paiir/pipeline/passes.py` and focused regression/tests that directly prove the `Standalone MaxPool ... requires predecessor format` issue is resolved.

## Plan

- [x] Confirm the exact `ReshapeOp -> Standalone MaxPool` graph pattern that triggers the missing predecessor-format error.
- [x] Restore a minimal source-side format-propagation fix for MaxPool through routing predecessors.
- [x] Re-run focused regression and the `sz160` customer compile path to confirm the `ValueError` is gone.
- [x] Record the new post-fix status and any remaining blocker.

## Review

- Restored the minimal source-side data-format fix in [passes.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/passes.py):
  - standalone `MaxPool` output-format inference now recursively looks through routing-only predecessors (`ReshapeOp` / `ConcatOp`) to collect already resolved effective formats
  - the change is intentionally narrow and only affects predecessor-format lookup for nodes whose direct predecessor has no `core_params`
- Kept the focused regression in [test_compile.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_compile.py):
  - `PoolAfterReshape` now proves `compile_to_paiir(...)` no longer fails on `ReshapeOp -> MaxPool2d`
- Verified the original customer blocker is resolved:
  - before the fix: `compile_to_paiir(strict=False)` failed with `ValueError: Standalone MaxPool ... requires predecessor format`
  - after the fix: that `ValueError` is gone
- Current `sz160` customer-model status after this fix:
  - `strict=True` still fails on unsupported `unsqueeze`
  - `strict=False` now progresses further and fails later with a `GraphValidationError` reporting `18` reshape/concat shape mismatches
  - in other words, the MaxPool predecessor-format problem is fixed, and the next real blocker is now the reshape/concat contract mismatch in the fused graph
- Verified with:
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k maxpool_after_reshape_compiles`
    - result: `1 passed`
  - `./.venv/bin/python - <<'PY' ... compile_to_paiir probe on sz160 pickle ... PY`
    - result:
      - `strict=True` -> `UnsupportedOpError` on `unsqueeze`
      - `strict=False` -> `GraphValidationError` (the previous MaxPool predecessor-format `ValueError` no longer appears)
  - `./.venv/bin/pytest tests/user/test_widerface_t1_backbone_sz160_paiir.py -q`
    - result: `2 passed`

# Data-Format Transparent Routing Helper

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this change narrowly scoped to `paibox/paiir/pipeline/passes.py` because the goal is only to reduce repeated routing-node checks without changing current behavior.

## Plan

- [x] Add a tiny helper for data-format-transparent routing nodes in `passes.py`.
- [x] Replace repeated `ConcatOp` / `ReshapeOp` checks in data-format propagation with that helper.
- [x] Re-run focused compile regressions to confirm behavior is unchanged.

## Review

- Added a local helper in [passes.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/passes.py) to centralize the current “data-format-transparent routing node” definition:
  - `_is_format_transparent_routing_node(...)`
- Applied it to the repeated `ConcatOp` / `ReshapeOp` checks in the data-format propagation path:
  - pass-1 routing-node skip
  - recursive effective-predecessor format collection
- Kept behavior unchanged:
  - no new node types were added to the transparent set
  - pass-2 still handles `ConcatOp` and `ReshapeOp` with their existing semantics
- Verification:
  - `ruff check paibox/paiir/pipeline/passes.py`
    - result: `All checks passed!`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q`
    - result: `76 passed`

# Unsqueeze / Repeat Input-Form Gaps

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this change narrowly scoped to the reshape-like frontend path because the goal is only to support function-form `torch.unsqueeze(...)` and tuple-form identity `repeat((1,...))`.

## Plan

- [x] Add reshape-like analysis support for function-form `torch.unsqueeze(...)`.
- [x] Extend identity-repeat detection to accept tuple/list repeat arguments.
- [x] Add focused shape-analysis / dims / compile / simulation regressions and verify they pass.

## Review

- Implementation:
  - `shape_analysis.py` now recognizes function-form `torch.unsqueeze(...)` as a reshape-like sink
  - identity-repeat detection now accepts tuple/list / `torch.Size` argument forms such as `x.repeat((1, 1, 1))`
  - `dims_prop.py` now treats function-form `torch.unsqueeze` the same way as method-form `unsqueeze`, resetting output dims to identity for the new rank
- Added regressions:
  - shape-analysis coverage for function-form `torch.unsqueeze(...)`
  - shape-analysis coverage for tuple-form identity `repeat((1,...))`
  - dims propagation coverage for function-form `torch.unsqueeze(...)`
  - compile coverage for `torch.unsqueeze(...) -> flatten -> Linear`
  - compile coverage for tuple-form identity `repeat((1,...)) -> flatten -> Linear`
  - graph-simulation coverage for both patterns against PyTorch references
- Verification:
  - `ruff check paibox/paiir/lowering/shape_analysis.py paibox/paiir/lowering/dims_prop.py tests/paiir/lowering/test_shape_analysis.py tests/paiir/lowering/test_dims_prop.py tests/paiir/pipeline/test_compile.py tests/paiir/pipeline/test_graph_simulation.py`
    - result: `All checks passed!`
  - `./.venv/bin/pytest tests/paiir/lowering/test_shape_analysis.py tests/paiir/lowering/test_dims_prop.py -q`
    - result: `38 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k 'function_unsqueeze_before_linear_compiles or tuple_repeat_all_ones_before_linear_compiles or unsqueeze_repeat_all_ones_compile_path_succeeds'`
    - result: `3 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_graph_simulation.py -q -k 'function_unsqueeze_before_linear or tuple_repeat_all_ones_before_linear'`
    - result: `2 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q`
    - result: `78 passed`
  - focused semantic probe:
    - `compile_to_paiir(strict=False)` on function-form `torch.unsqueeze(x, 0)` now returns the correct output shape and value instead of silently bypassing the operator

# Squeeze Follow-up

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this change narrowly scoped to reshape-like frontend support for method/function-form `squeeze`, without expanding into broadcast/copy-style shape operators.

## Plan

- [x] Add reshape-like analysis support for method/function-form `squeeze`.
- [x] Extend dims propagation so `squeeze` resets output dims to identity for the new rank.
- [x] Add focused shape-analysis / dims / compile / simulation regressions and verify they pass.

## Review

- Implementation:
  - `shape_analysis.py` now recognizes both method-form `x.squeeze(...)` and function-form `torch.squeeze(...)` as reshape-like sinks
  - `dims_prop.py` now treats both forms of `squeeze` as reshape-like so output dims reset to identity for the new rank
- Added regressions:
  - shape-analysis coverage for method/function-form `squeeze`
  - dims propagation coverage for method/function-form `squeeze`
  - compile coverage for `squeeze -> flatten -> Linear`
  - graph-simulation coverage for both method/function forms against PyTorch references
- Verification:
  - `ruff check paibox/paiir/lowering/shape_analysis.py paibox/paiir/lowering/dims_prop.py tests/paiir/lowering/test_shape_analysis.py tests/paiir/lowering/test_dims_prop.py tests/paiir/pipeline/test_compile.py tests/paiir/pipeline/test_graph_simulation.py`
    - result: `All checks passed!`
  - `./.venv/bin/pytest tests/paiir/lowering/test_shape_analysis.py tests/paiir/lowering/test_dims_prop.py -q`
    - result: `42 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k 'method_squeeze_before_linear_compiles or function_squeeze_before_linear_compiles'`
    - result: `2 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_graph_simulation.py -q -k 'method_squeeze_before_linear or function_squeeze_before_linear'`
    - result: `2 passed`
  - broader regression:
    - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q`
    - result: `78 passed`

# Shared Reshape Semantics Helper

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this step focused on extracting shared pure-function reshape semantics without changing supported behavior.

## Plan

- [x] Add one shared internal helper module for reshape-like target classification and shape/dims utilities.
- [x] Refactor lowering / IR / pipeline / backendv2 to use that helper instead of duplicated local rules.
- [x] Run focused regressions to confirm behavior is unchanged after the refactor.

## Review

- Added [\_reshape_semantics.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/_reshape_semantics.py) as the shared internal, function-oriented source of truth for:
  - reshape-like target classification
  - identity-repeat argument handling
  - pure shape/dims helpers such as `shape_after_dims(...)` and `materialize_logical_layout(...)`
- Refactored current users to consume the shared helper instead of keeping their own local copies:
  - [shape_analysis.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/shape_analysis.py)
  - [dims_prop.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/dims_prop.py)
  - [converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/converter.py)
  - [op_node.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/op_node.py)
  - [passes.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/passes.py)
  - [backendv2/op_node.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/backendv2/op_node.py)
- Verification:
  - `ruff check paibox/paiir/_reshape_semantics.py paibox/paiir/lowering/shape_analysis.py paibox/paiir/lowering/dims_prop.py paibox/paiir/lowering/converter.py paibox/paiir/ir/op_node.py paibox/paiir/pipeline/passes.py paibox/backendv2/op_node.py`
    - result: `All checks passed!`
  - `./.venv/bin/pytest tests/paiir/lowering/test_shape_analysis.py tests/paiir/lowering/test_dims_prop.py tests/paiir/pipeline/test_compile.py tests/backendv2/test_reorder_node.py -q`
    - result: `123 passed`

# Functional Conv Lowering Cleanup

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this change focused on `converter.py` and targeted tests so functional-conv lowering becomes systematic without broad unrelated refactors.

## Plan

- [x] Refactor the current ad hoc `FunctionalConv2d` design into shared functional-conv lowering primitives.
- [x] Add `F.conv1d` support using the same lowering path as `F.conv2d`.
- [x] Update focused tests to verify both 1D and 2D function-form conv lowering.

## Review

- Implementation:
  - replaced the previous single-purpose `FunctionalConv2d` handling with shared functional-conv lowering primitives in [converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/converter.py)
  - introduced a small lowering spec + shared metadata mixin so 1D/2D function-form conv adapters share the same graph-first initialization path
  - kept the existing `FunctionalConv2d` adapter name for compatibility while adding `FunctionalConv1d`
  - generalized the call-function matcher from a `conv2d`-only path to a functional-conv path that supports both `conv1d` and `conv2d`
- Test updates:
  - added `FunctionalQuantizedConv1d` compile coverage in [test_compile.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_compile.py)
  - added prebuild-analysis coverage for `conv1d`
  - added backend raw-weight coverage for `Conv1d` expansion in [test_routing_v2_raw_weights.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/backendv2/test_routing_v2_raw_weights.py)
  - repaired stale backend test constructors/imports in that raw-weight test so it validates the current backendv2 API instead of an old `InNode` signature
- Verification:
  - `ruff check paibox/paiir/lowering/converter.py tests/paiir/pipeline/test_compile.py tests/backendv2/test_routing_v2_raw_weights.py`
    - result: `All checks passed!`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k functional_conv`
    - result: `4 passed`
  - `./.venv/bin/pytest tests/backendv2/test_routing_v2_raw_weights.py -q`
    - result: `6 passed`
  - broader regression:
    - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q`
    - result: `82 passed`
  - user-facing compatibility check:
    - `./.venv/bin/pytest tests/user/test_paiir_frontend_example_graphs.py -q -k functional_conv_example_graph`
    - result: `1 passed`

# Functional Conv Simplification

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this follow-up focused on simplifying functional-conv lowering back to native `nn.Conv1d/nn.Conv2d` instances plus attached metadata, without changing functional-conv behavior.

## Plan

- [x] Remove the temporary functional-conv wrapper classes/mixin from `converter.py`.
- [x] Build native `nn.Conv1d/nn.Conv2d` modules directly and attach graph-side metadata.
- [x] Update functional-conv assertions so they validate native conv modules + metadata instead of wrapper type names.
- [x] Re-run focused compile/backend/user-facing regressions.

## Review

- Implementation:
  - removed the temporary `_FunctionalConvMixin` / `FunctionalConv1d` / `FunctionalConv2d` wrapper-class layer from [converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/converter.py)
  - function-form conv lowering now constructs native `nn.Conv1d` / `nn.Conv2d` modules directly and attaches graph-side metadata (`raw_weight`, `scale`, `zero_point`)
  - kept the shared functional-conv lowering path and `conv1d` / `conv2d` dual support introduced in the previous cleanup step
- Test updates:
  - updated functional-conv compile tests to assert native ConvNd modules plus preserved metadata instead of wrapper type names
  - updated the user-facing functional-conv example report expectations from `FunctionalConv2d` to `Conv2d`
  - preserved backend raw-weight coverage, including the new `Conv1d` case
- Verification:
  - `ruff check paibox/paiir/lowering/converter.py tests/paiir/pipeline/test_compile.py tests/backendv2/test_routing_v2_raw_weights.py tests/user/test_paiir_frontend_example_graphs.py`
    - result: `All checks passed!`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k functional_conv`
    - result: `4 passed`
  - `./.venv/bin/pytest tests/backendv2/test_routing_v2_raw_weights.py -q`
    - result: `6 passed`
  - `./.venv/bin/pytest tests/user/test_paiir_frontend_example_graphs.py -q -k functional_conv_example_graph`
    - result: `1 passed`
  - broader regression:
    - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q`
    - result: `82 passed`

# Revert Temporary PAIIR Source Fix

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep the revert scoped to `paibox/paiir/pipeline/passes.py` only, because the user explicitly asked to roll back the temporary PAIIR source fix first before discussing the failure mode.

## Plan

- [x] Revert only the temporary predecessor-format source fix in `passes.py`.
- [x] Re-run a minimal `sz160` customer pickle probe to confirm the post-revert failure point.
- [x] Summarize the observed problem for the user without changing the customer-side log/test scripts again.

## Review

- Reverted the temporary source-side data-format propagation helper path from [passes.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/passes.py).
- Current post-revert behavior for [graph_module_spike_yolos_widerface_T1_backbone_int8_sz160_quantized_20260323_182735.pkl](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/fx_out/fx_pickle/graph_module_spike_yolos_widerface_T1_backbone_int8_sz160_quantized_20260323_182735.pkl):
  - `torch_to_paiir(strict=False)` still succeeds and produces a `228`-node / `276`-edge graph
  - `compile_to_paiir(strict=False)` now fails earlier with:
    - `ValueError: Standalone MaxPool 'StandaloneCompOp_209' requires predecessor format`
- Focused manual verification:
  - `./.venv/bin/python - <<'PY' ... torch_to_paiir / compile_to_paiir probe on sz160 pickle ... PY`
  - observed:
    - `torch_to_paiir(strict=True)` -> `UnsupportedOpError` on `unsqueeze`
    - `torch_to_paiir(strict=False)` -> success
    - `compile_to_paiir(strict=True)` -> `UnsupportedOpError` on `unsqueeze`
    - `compile_to_paiir(strict=False)` -> `ValueError` on standalone MaxPool predecessor format

# tests/user sz160 PAIIR Compile Log

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Do not create a dedicated `git worktree` because this task is limited to `tests/user/**` cleanup plus a local customer-artifact log generation flow that depends on the current untracked `tests/user/fx_out/**` fixtures in this workspace.

## Ownership

- Owner: Codex
- Branch: `bugfix-paiirgraph-sim-fixes`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files:
  - `tests/user/**`
  - `tasks/todo.md`
  - `tasks/lessons.md`
- Blocked Files:
  - `paibox/**` product code
  - unrelated `tests/paiir/**` and `tests/backendv2/**`
- Dependencies:
  - keep the customer-model scope centered on `tests/user/fx_out/fx_pickle/**`
  - produce a concrete `sz160`-specific PAIIR compile log with graph summary and operator statistics
  - preserve only generic non-customer tests in `tests/user` when they still provide standalone value
- Verification:
  - focused `pytest` for the `sz160` customer compile/log entry
  - optional `tests/user` sweep after simplification if the cleanup scope changes more than one file

## Plan

- [x] Audit `tests/user` for customer-model-specific legacy tests versus generic reusable tests.
- [x] Simplify the current customer compile path so the primary `tests/user` entry targets only the `sz160` pickle and writes the requested log artifact.
- [x] Trim obviously unrelated legacy `tests/user` customer-model test code where it no longer matches the current artifact scope.
- [x] Run focused verification, confirm the log contents, and record the artifact path here.

## Review

- Kept the customer-model scope centered on the requested `sz160` artifact:
  - [graph_module_spike_yolos_widerface_T1_backbone_int8_sz160_quantized_20260323_182735.pkl](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/fx_out/fx_pickle/graph_module_spike_yolos_widerface_T1_backbone_int8_sz160_quantized_20260323_182735.pkl)
- Updated [test_widerface_t1_backbone_sz160_paiir.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/test_widerface_t1_backbone_sz160_paiir.py) so it now:
  - targets only the `sz160` pickle instead of iterating across all customer pickle sizes
  - records both `compile_to_paiir(...)` attempts and the pre-validation fused `PAIIRGraph`
  - writes a single customer-facing log artifact with graph summary and operator statistics
- Generated log artifact:
  - [widerface_t1_backbone_int8_sz160_paiirgraph.summary.log](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/debug/widerface_t1_backbone_int8_sz160_paiirgraph.summary.log)
- The log now contains:
  - source repo path and pickle path
  - `compile_to_paiir(strict=True)` failure on unsupported `unsqueeze`
  - `compile_to_paiir(strict=False)` failure on final `GraphValidationError`
  - pre-validation fused `PAIIRGraph` node/edge counts
  - operator-type counts and module-type counts
  - full `PAIIRGraph.summary()` dump for the pre-validation fused graph
- Current `sz160` evaluation result:
  - `compile_to_paiir(strict=True)` does not compile because `unsqueeze` remains unsupported
  - `compile_to_paiir(strict=False)` currently reaches PAIIR compilation but fails final graph validation with `18` reshape/concat shape mismatches
  - the furthest stable graph artifact before that validation failure has:
    - `228` graph nodes
    - `276` graph edges
    - `226` `OpNode`s
    - node-type counts: `ReshapeOp=88, StandaloneCompOp=67, StandaloneActOp=40, ConcatOp=17, PotentialAddOp=14`
    - module-type counts: `FunctionalConv2d=64, ANNNodeV25=40, MaxPool2d=3`
- Preserved `dualcnn` test code per user correction and rewired the smoke test entry to the currently available helper surface:
  - [test_dualcnn_compile.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/test_dualcnn_compile.py)
- Added a focused pipeline regression for the new data-format fix:
  - [test_compile.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_compile.py)
  - [passes.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/passes.py)
  - this ensures standalone `MaxPool` can infer predecessor format through routing-only predecessors such as `ReshapeOp`
- Verified with:
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k maxpool_after_reshape_compiles`
    - result: `1 passed`
  - `./.venv/bin/pytest tests/user/test_widerface_t1_backbone_sz160_paiir.py -q`
    - result: `2 passed in 5.28s`
  - `./.venv/bin/pytest tests/user/test_dualcnn_compile.py -q`
    - result: `1 passed in 3.29s`

# PAIIR ReshapeOp Handling Audit

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Do not create a dedicated `git worktree` because this task is a source-and-test audit only, and it must evaluate the current in-flight `paiir` changes as they exist in this workspace.

## Ownership

- Owner: Codex
- Branch: `bugfix-paiirgraph-sim-fixes`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files:
  - `tasks/todo.md`
  - `paibox/paiir/**` for inspection only
  - `tests/paiir/**` for inspection only
- Blocked Files:
  - product code edits unless the audit uncovers a factual inconsistency that the user asks to fix later
  - `tests/user/**` deployment-evaluation scripts, per request
- Dependencies:
  - trace reshape-like FX analysis for `reshape` / `view` / `view_as` / `flatten`
  - trace lowering into `ReshapeOp` and graph-level simulation behavior
  - evaluate targeted tests for validity, overlap, and missing coverage
- Verification:
  - inspect lowering, IR, and pipeline code paths directly
  - inspect targeted tests under `tests/paiir/lowering/**`, `tests/paiir/pipeline/**`, and relevant IR tests

## Plan

- [x] Inspect FX-level reshape sink analysis and auxiliary-node pruning.
- [x] Inspect converter lowering into `ReshapeOp`, including predecessor wiring and unsupported cases.
- [x] Inspect `ReshapeOp` runtime/simulation semantics and graph validation logic.
- [x] Audit focused tests for effectiveness and identify missing coverage or weak assertions.
- [x] Summarize whether the current implementation is functionally complete and whether the tests are complete enough.

## Review

- Current supported reshape-like front-end coverage is intentionally narrow but real:
  - FX analysis recognizes `nn.Flatten`, `torch.flatten`, `torch.reshape`, and method-form `flatten` / `reshape` / `view` / `view_as`
  - lowering materializes these as `ReshapeOp`
  - graph simulation executes them via `ReshapeOp.forward()`
  - signal-domain and data-format propagation both treat `ReshapeOp` as a routing pass-through
- Happy-path behavior is largely covered and works for the currently tested cases:
  - flatten transitions into Linear
  - `view(size(0), -1)` before Linear
  - reshape shape-arithmetic auxiliaries
  - chained `flatten -> reshape -> flatten`
- Audit conclusion: the handling is not yet complete end to end.
  - confirmed bug: `input_nodes_override` is used to fill shape/dims metadata but is never persisted for edge wiring, so `view_as(ref)` lowers to a `ReshapeOp` with two predecessors and fails final validation
  - confirmed bug: reshape-like ops after bypassed layout ops such as `transpose` / `permute` fail compilation because the `ReshapeOp` metadata reflects the bypassed node shape while graph wiring collapses the predecessor back to the pre-layout source
  - latent inconsistency: `DimsProp` resets dims for `flatten` / `reshape` / `view`, but not `view_as`, even though lowering classifies `view_as` as a reshape sink
- Test assessment:
  - effective: shape-analysis tests and graph-simulation tests do catch the main happy paths
  - incomplete: no focused pass test covers `_validate_reshape_contract()` failure modes
  - incomplete: no end-to-end test covers `transpose/permute -> reshape-like op`
  - incomplete: no test covers `view_as` with a distinct reference tensor input
  - incomplete: no direct unit test targets `ReshapeOp.forward()` behavior itself
- Verification:
  - `./.venv/bin/pytest tests/paiir/lowering/test_shape_analysis.py tests/paiir/lowering/test_dims_prop.py tests/paiir/pipeline/test_compile.py tests/paiir/pipeline/test_passes.py tests/paiir/pipeline/test_graph_simulation.py -q`
    - result: `221 passed, 2 skipped, 1 failed`
    - note: the single failure is unrelated existing AvgPool split-core simulation coverage in `tests/paiir/pipeline/test_graph_simulation.py::TestMultiLayerSNN::test_avgpool_lif_split_core_snn`
  - manual compile probes:
    - `x.view_as(ref)` reproduces `ReshapeOp ... must have exactly one predecessor, got 2`
    - `x.transpose(...).flatten(...)` and `x.permute(...).reshape(...)` reproduce `ReshapeOp ... predecessor shape mismatch`

# PAIIR view_as Fix

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Do not create a dedicated `git worktree` because this is a scoped `paiir` bug fix that must integrate with the in-flight local `paiir` and test changes already present in this workspace.

## Ownership

- Owner: Codex
- Branch: `bugfix-paiirgraph-sim-fixes`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files:
  - `paibox/paiir/lowering/**`
  - `tests/paiir/**`
  - `tasks/todo.md`
- Blocked Files:
  - unrelated backend / onboard / docs files
  - `tests/user/**` unless later verification specifically requires them
- Dependencies:
  - keep the fix narrowly scoped to `view_as`
  - preserve existing reshape / flatten happy paths
  - avoid broad semantic changes to `transpose` / `permute` support in this patch
- Verification:
  - focused lint / import sanity on touched files
  - focused pytest for lowering / compile / graph-simulation coverage that exercises `view_as`

## Plan

- [x] Persist reshape/input overrides so `view_as(ref)` wires only its data tensor predecessor.
- [x] Align dims propagation so `view_as` is treated consistently with other reshape-like ops.
- [x] Add focused regression tests for `view_as` lowering, compilation, and simulation.
- [x] Run targeted verification and record the outcome.

## Review

- Implementation:
  - `_register_ir_node(...)` now persists `input_nodes_override` into the lowering context so the later edge-wiring pass uses the normalized data-input set instead of falling back to `node.all_input_nodes`
  - `DimsProp` now includes `view_as` in `RESHAPE_TARGETS`, making its output dims reset to identity just like `flatten` / `reshape` / `view`
- Added regressions:
  - dims propagation coverage for `x.view_as(ref)`
  - compile-time coverage for `ref = x.flatten(1); y = x.view_as(ref)` so shape/reference paths do not become extra data predecessors
  - graph-simulation coverage for the same `view_as` reference-path pattern before `Linear`
- Verification:
  - `ruff check paibox/paiir/lowering/converter.py paibox/paiir/lowering/dims_prop.py tests/paiir/lowering/test_dims_prop.py tests/paiir/pipeline/test_compile.py tests/paiir/pipeline/test_graph_simulation.py`
    - result: `All checks passed!`
  - `./.venv/bin/pytest tests/paiir/lowering/test_dims_prop.py -q`
    - result: `29 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k 'view_as_reference_path or shape_only_reshape_args_do_not_become_data_predecessors'`
    - result: `2 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_graph_simulation.py -q -k 'view_as_reference_path_before_linear or view_and_view_as_routing_before_linear'`
    - result: `2 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q`
    - result: `73 passed`
- Scope note:
  - this patch fixes `view_as` wiring and metadata consistency only
  - it does not address the separate `transpose/permute -> reshape-like` mismatch identified in the earlier audit

# PAIIR transpose/permute -> reshape-like Fix

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Do not create a dedicated `git worktree` because this follow-up fix is tightly coupled to the in-flight `paiir` lowering and test edits already present in this workspace.

## Ownership

- Owner: Codex
- Branch: `bugfix-paiirgraph-sim-fixes`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files:
  - `paibox/paiir/**`
  - `paibox/backendv2/**`
  - `tests/paiir/**`
  - `tests/backendv2/**`
  - `tasks/todo.md`
- Blocked Files:
  - unrelated backend / onboard / docs files
  - `tests/user/**`
- Dependencies:
  - preserve the existing “transpose/permute remain bypass FX nodes” lowering shape
  - make `ReshapeOp` correctly materialize pending logical layout changes carried by `input_dims`
  - keep backend reorder semantics aligned with graph simulation semantics
- Verification:
  - focused lint on touched files
  - focused pytest for compile / simulation / backend reorder regressions

## Plan

- [x] Teach `ReshapeOp` simulation to materialize pending layout transforms encoded in `input_dims`.
- [x] Relax reshape validation to compare predecessor shape after applying logical input dims.
- [x] Update backendv2 reorder mapping so flattened reorder respects `ReshapeOp.input_dims`.
- [x] Add focused regressions for `transpose/permute -> flatten/reshape` and run targeted verification.

## Review

- Implementation:
  - `ReshapeOp.forward()` now applies the pending logical permutation encoded in `input_dims[0]` before executing flatten/reshape semantics
  - reshape contract validation now accepts predecessor shapes that match after applying `input_dims`, instead of requiring a direct shape equality against the physical predecessor tensor
  - backendv2 `ReorderNode.get_reorder_info()` now uses `ReshapeOp.input_dims` to compute the correct flattened element mapping for folded `transpose/permute -> reshape-like` paths
- Added regressions:
  - compile coverage for `transpose -> flatten -> Linear`
  - compile coverage for `permute -> reshape -> Linear`
  - graph-simulation coverage for both patterns against PyTorch references
  - backend reorder coverage for the concrete flattened permutation mapping
- Verification:
  - `ruff check paibox/paiir/ir/op_node.py paibox/paiir/pipeline/passes.py paibox/backendv2/op_node.py tests/paiir/pipeline/test_compile.py tests/paiir/pipeline/test_graph_simulation.py tests/backendv2/test_reorder_node.py`
    - result: `All checks passed!`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k 'transpose_then_flatten_before_linear_compiles or permute_then_reshape_before_linear_compiles'`
    - result: `2 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_graph_simulation.py -q -k 'transpose_then_flatten_before_linear or permute_then_reshape_before_linear'`
    - result: `2 passed`
  - `./.venv/bin/pytest tests/backendv2/test_reorder_node.py -q`
    - result: `1 passed`
  - broader regression:
    - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q`
    - result: `1 failed, 74 passed`
    - unrelated residual failure: `tests/paiir/pipeline/test_compile.py::TestCompileBasic::test_maxpool_after_reshape_compiles` still fails in `propagate_data_format()` because standalone MaxPool sees no predecessor format through a reshape path; this was not introduced by the current transpose/permute fix and remains a separate issue

# PAIIR InputNode Data Format Trace

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Do not create a dedicated `git worktree` because this task is source inspection only and does not modify product code.

## Ownership

- Owner: Codex
- Branch: `bugfix-paiirgraph-sim-fixes`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files:
  - `tasks/todo.md`
  - source files inspected for explanation only
- Blocked Files:
  - product code edits unless inspection reveals a factual inconsistency
- Dependencies:
  - trace where `InputNode` effective output format is assigned
  - trace how downstream `OfflineCoreOp` input format is inferred from predecessors
  - confirm the explanation against focused pass/unit tests
- Verification:
  - inspect `paibox/paiir/pipeline/passes.py`, `paibox/paiir/pipeline/data_format.py`, and related IR definitions
  - inspect focused tests under `tests/paiir/pipeline/test_data_format.py`

## Plan

- [x] Locate where `InputNode` is created and whether it stores dtype directly.
- [x] Trace `propagate_data_format()` to find how effective node formats are seeded and propagated.
- [x] Confirm the behavior against focused tests and summarize the result for the user.

## Review

- `InputNode` only stores shape metadata; it does not own a dtype/data-format field.
- Effective `InputNode` format is seeded inside `propagate_data_format()` via caller-provided `input_formats` or `_infer_input_node_default()`.
- Downstream `OfflineCoreOp` input formats are assigned in the second propagation pass by merging predecessor effective output formats.

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Do not create a dedicated `git worktree` for this task because the target `tests/paiir/**` and related lowering files already contain in-flight local changes that this consolidation must preserve and integrate with directly.

## Ownership

- Owner: Codex
- Branch: `bugfix-paiirgraph-sim-fixes`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files:
  - `tests/paiir/**`
  - `paibox/paiir/lowering/**`
  - `tasks/todo.md`
- Blocked Files:
  - unrelated backend / onboard / docs files
  - unrelated generated artifacts
- Dependencies:
  - preserve current tracer semantics used by lowering and converter tests
  - separate full custom `_PAIIRTracer` flows from lighter FX/ShapeProp-based test helpers
  - prefer the smallest shared helper surface that actually removes duplication
- Verification:
  - focused `ruff check` on touched test/helper files
  - focused pytest for updated `tests/paiir/lowering/**` and any other touched `tests/paiir/**` modules

## Plan

- [x] Audit all `tests/paiir/**` modules for tracer construction and propagation helper patterns.
- [x] Classify helpers into full custom PAIIR tracer flows versus partial/lightweight FX tracing flows.
- [x] Introduce shared test tracer helper builders in the narrowest useful test scope.
- [x] Migrate affected tests to the shared helpers without changing assertion intent.
- [x] Run focused verification and record the final helper organization.

## Review

- Audit result across `tests/paiir/**`:
  - full custom PAIIR tracer + erase + shape/dims propagation was duplicated in `tests/paiir/lowering/test_shape_analysis.py` and `tests/paiir/pipeline/test_compile.py`
  - lightweight FX symbolic tracing + `ShapeProp` + `DimsProp` lived in `tests/paiir/lowering/test_dims_prop.py`
  - full graph-conversion helpers such as `convert_and_fuse(...)` and `convert_fuse_propagate(...)` already belonged in `tests/paiir/conftest.py` and were left unchanged
- Consolidation result:
  - added `tests/paiir/tracing.py` as the shared non-fixture helper module for tracer construction
  - exposed `trace_with_fx(...)` for plain FX symbolic tracing
  - exposed `trace_with_fx_shape_and_dims(...)` for lightweight metadata/dims propagation tests
  - exposed `trace_with_paiir_tracer(...)` for raw project-specific tracer coverage
  - exposed `trace_for_lowering(...)` for the full custom tracer + erase + shape/dims preparation path used by lowering-analysis tests
- Updated test consumers:
  - `tests/paiir/lowering/test_shape_analysis.py`
  - `tests/paiir/lowering/test_dims_prop.py`
  - `tests/paiir/pipeline/test_compile.py`
- Placement decision:
  - used a dedicated helper module instead of expanding `conftest.py`, because these are ordinary importable utilities shared across both `lowering/` and `pipeline/` tests rather than pytest-managed fixtures
- Verification:
  - `ruff check tests/paiir/tracing.py tests/paiir/lowering/test_shape_analysis.py tests/paiir/lowering/test_dims_prop.py tests/paiir/pipeline/test_compile.py`
    - result: `All checks passed!`
  - `./.venv/bin/pytest tests/paiir/lowering/test_dims_prop.py tests/paiir/lowering/test_shape_analysis.py tests/paiir/pipeline/test_compile.py -q`
    - result: `105 passed`
    - warnings: existing expected warnings from unsupported `unsqueeze` / `repeat` bypass and AvgPool tau auto-optimization

# PAIIRGraph Shape-Simulation Fixes

# PAIIR API Surface Cleanup

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this in the current workspace because it is a direct follow-up to the in-flight `paiir` layering edits and should be integrated with those changes rather than split into a separate worktree mid-stream.

## Ownership

- Owner: Codex
- Branch: `bugfix-paiirgraph-sim-fixes`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files:
  - `paibox/paiir/**`
  - `paibox/backendv2/**`
  - `tests/paiir/**`
  - `tests/backendv2/**`
  - `docs/paiir_backend_guide.md`
  - `tasks/todo.md`
  - `tasks/lessons.md`
- Blocked Files:
  - unrelated onboard fixtures / generated outputs
  - unrelated frontend or backend modules outside direct import fallout
- Dependencies:
  - keep `AvgPoolDeployMetadata` semantically under `pipeline/avgpool`
  - prefer minimal edits and preserve current compile/runtime behavior
  - keep public user-facing compile entrypoints stable unless clearly internal-only
- Verification:
  - focused import smoke tests for `paibox.paiir`, `paibox.paiir.pipeline`, `paibox.paiir.pipeline.avgpool`
  - focused pytest for compile / avgpool / backend import fallout

## Plan

- [x] Restore `AvgPoolDeployMetadata` to `pipeline/avgpool` with typing-only decoupling from IR runtime imports.
- [x] Audit repo usage of `paibox.paiir` package exports and define a smaller top-level API surface.
- [x] Remove unnecessary re-exports from package `__init__` files and update internal/backend call sites that should import from narrower modules.
- [x] Run focused verification and record the resulting public API boundary.

## Review

- Final minimal-change design keeps `AvgPoolDeployMetadata` in `paibox/paiir/pipeline/avgpool/metadata.py`, which matches its functional ownership as AvgPool-specific compile-pass metadata.
- `paibox/paiir/ir/op_node.py` no longer imports that type at runtime. It now uses a `TYPE_CHECKING`-only import plus a quoted annotation, so IR does not depend on pipeline initialization while static typing remains intact.
- The earlier `paibox/paiir/ir/compile_metadata.py` experiment was removed; this section supersedes that intermediate layering attempt.
- Top-level `paibox.paiir` is now intentionally frontend-facing. It keeps:
  - neuron / LUT building blocks
  - `PAIIRGraph`
  - `register_neuron`
  - `torch_to_paiir`
  - `compile_to_paiir`
  - `CompileConfig`
- Removed from `paibox.paiir` package exports:
  - backend/internal IR node classes such as `OfflineCoreOp`, `SequentialOp`, `AccumulateOp`, `ConcatOp`, `Standalone*Op`
  - graph plumbing / expression-layer internals such as `Edge`, `InputNode`, `OutputNode`, `GeneralAddOp`, `PotentialAddOp`
  - compile-time parameter / inference helpers such as `LutData`, `OfflineCoreParams`, `NeuronParams`, `OnlineCoreParams`, `DataFormat`, `infer_*`, `merge_data_formats`
- `paibox.paiir.pipeline` is reduced to `CompileConfig` and `compile_to_paiir`.
- `paibox.paiir.pipeline.avgpool` keeps deploy-scheme and calibration entrypoints, while compensation helpers and `AvgPoolDeployMetadata` now require narrower imports from `compensation.py` and `metadata.py`.
- Updated internal/backend call sites to use narrower imports:
  - `paibox/backendv2/**`
  - `tests/backendv2/**`
  - AvgPool tests
  - backend guide snippets
- Verification:
  - `ruff check paibox/paiir/__init__.py paibox/paiir/ir/op_node.py paibox/paiir/pipeline/__init__.py paibox/paiir/pipeline/avgpool/__init__.py paibox/paiir/pipeline/avgpool/metadata.py paibox/paiir/pipeline/avgpool/fusion.py paibox/backendv2/op_node.py paibox/backendv2/routing.py paibox/backendv2/mapper.py paibox/backendv2/core_config.py tests/paiir/ir/test_op_node.py tests/paiir/pipeline/test_compile.py tests/paiir/pipeline/avgpool/test_compensation.py tests/paiir/pipeline/avgpool/test_deploy_scheme.py tests/backendv2/test_routing_v2_raw_weights.py docs/paiir_backend_guide.md`
    - result: `All checks passed!`
  - `./.venv/bin/python - <<'PY' ... import smoke ...`
    - result: `import smoke ok`
  - `./.venv/bin/python - <<'PY' ... avgpool import split ...`
    - result: `avgpool import split ok`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q`
    - result: `71 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/avgpool/test_deploy_scheme.py tests/paiir/ir/test_op_node.py tests/backendv2/test_routing_v2_raw_weights.py -q`
    - result: `50 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/avgpool/test_compensation.py tests/paiir/pipeline/avgpool/test_deploy_scheme.py tests/paiir/ir/test_op_node.py tests/backendv2/test_routing_v2_raw_weights.py -q`
    - result: `7 failed, 82 passed`
    - note: the failures are in `test_compensation.py` and are unrelated to this API-surface change; they come from pre-existing `LutData` frozen-dataclass mutation in compensation helpers.

# PAIIR Layering Cleanup

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Do not create a dedicated `git worktree` for this task because the target `paiir` files already contain in-flight local changes that this refactor must preserve and integrate with directly.

## Ownership

- Owner: Codex
- Branch: `bugfix-paiirgraph-sim-fixes`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files:
  - `paibox/paiir/ir/**`
  - `paibox/paiir/pipeline/**`
  - `paibox/paiir/__init__.py`
  - `tests/paiir/pipeline/test_compile.py`
  - `tasks/todo.md`
- Blocked Files:
  - unrelated `backendv2/**`
  - unrelated onboard fixtures / generated outputs
- Dependencies:
  - preserve existing public imports from `paibox.paiir` and `paibox.paiir.pipeline.avgpool`
  - keep compatibility with current `compile_to_paiir` / `torch_to_paiir` call paths
- Verification:
  - import smoke tests for `paibox.paiir`, `paibox.paiir.pipeline`, `paibox.paiir.pipeline.avgpool`
  - focused pytest coverage for compile / avgpool paths

## Plan

- [x] Audit `paiir` file responsibilities and pin the intended layer order from file function and call direction.
- [x] Remove IR-to-pipeline reverse imports by relocating shared compile-time metadata to a lower layer.
- [x] Replace package-level lazy imports with explicit re-exports where the cycle is no longer needed.
- [x] Run focused verification and record the review conclusion, including whether lazy import is still justified anywhere.

## Review

- Intended layer order is now explicit:
  - foundation: `paibox/paiir/ir/**` and `paibox/paiir/nn/**`
  - front-end lowering: `paibox/paiir/lowering/**`
  - compile pipeline: `paibox/paiir/pipeline/**`
  - package aggregation only in `paibox/paiir/__init__.py`
- The concrete layer violation was `paibox/paiir/ir/op_node.py -> paibox/paiir/pipeline/avgpool/metadata.py`.
- Resolved by introducing `paibox/paiir/ir/compile_metadata.py` and moving `AvgPoolDeployMetadata` there, because it is compile-time state attached to IR nodes rather than pipeline implementation detail.
- `paibox/paiir/pipeline/avgpool/metadata.py` now acts only as a compatibility re-export, so existing import paths keep working.
- With the reverse import removed, `paibox/paiir/pipeline/__init__.py` and `paibox/paiir/pipeline/avgpool/__init__.py` no longer need package-level lazy import to avoid circular initialization and were restored to explicit re-exports.
- Conclusion on lazy import:
  - it was compensating for a misplaced cross-layer type, not expressing a healthy boundary
  - after the type moves downward, eager package exports are simpler and make dependency direction obvious
  - lazy import should remain an optional startup-cost optimization only, not the primary fix for package layering
- Verification:
  - `ruff check paibox/paiir/ir/compile_metadata.py paibox/paiir/ir/__init__.py paibox/paiir/ir/op_node.py paibox/paiir/pipeline/__init__.py paibox/paiir/pipeline/avgpool/__init__.py paibox/paiir/pipeline/avgpool/fusion.py paibox/paiir/pipeline/avgpool/metadata.py tests/paiir/pipeline/test_compile.py`
    - result: `All checks passed!`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q`
    - result: `71 passed`
  - `./.venv/bin/pytest tests/paiir/pipeline/avgpool/test_deploy_scheme.py -q`
    - result: `7 passed`
  - `./.venv/bin/python - <<'PY' ... import smoke ...`
    - result: `import smoke ok`

# Converter Reshape Lowering Cleanup

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep the change scoped to `converter.py` plus minimal verification only; no extra worktree is needed for this small refactor-and-design pass.

## Plan

- [x] Extract the duplicated flatten-lowering logic in `_build_reshape_like_ir_node()` into a dedicated helper without changing behavior.
- [x] Run a minimal reshape/flatten regression to confirm the refactor is behavior-preserving.
- [x] Record the refactor result and the follow-up design plan for `ReshapeOp` / shape-slice generalization.

## Review

- Refactored [converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/converter.py) so module/function/method `flatten` lowering now all reuse a single `_build_flatten_ir_node(...)` helper instead of duplicating the same `start_dim/end_dim -> ReshapeOp` logic in multiple branches.
- This was a structure-only cleanup. No lowering semantics changed.
- Minimal verification:
  - `./.venv/bin/python -m pytest tests/paiir/pipeline/test_graph_simulation.py -q -k 'flatten_linear_transition or view_and_view_as_routing_before_linear'`
    - result: `2 passed`
  - `ruff check paibox/paiir/lowering/converter.py`
    - result: `All checks passed`

# weight_skew -> dev PR Draft

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Do not create a dedicated `git worktree` because this task only inspects remote refs and drafts PR text; it does not modify product code or require parallel write isolation.

## Ownership

- Owner: Codex
- Branch: `bugfix-paiirgraph-sim-fixes`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files:
  - `tasks/todo.md`
- Blocked Files:
  - all source files unless inspection shows an unavoidable documentation-only follow-up
- Dependencies:
  - compare the latest remote `weight_skew` against the latest remote `dev`
  - base the PR text on actual commit/file delta rather than local assumptions
- Verification:
  - `git fetch origin dev weight_skew`
  - `git log --oneline origin/dev..origin/weight_skew`
  - `git diff --stat origin/dev...origin/weight_skew`

## Plan

- [x] Fetch the latest remote `dev` and `weight_skew` refs.
- [x] Review commit and file-level deltas between `origin/dev` and `origin/weight_skew`.
- [x] Draft copyable Markdown PR title and description that matches the actual branch scope.

## Review

- Remote comparison result:
  - `origin/weight_skew` is ahead of `origin/dev` by 1 commit and not behind it.
  - unique commit: `e90ecbb feat: use weight skew`
  - changed files: 10, concentrated in `paibox/backendv2/**` plus one `paibox/paiir/__init__.py` export update
- Main implementation themes in the diff:
  - added `paibox/backendv2/get_weight.py` to expand graph paths into dense raw-weight matrices and centralize dense-vs-sparse selection
  - introduced `ReorderNode` / `ReorderGroup` flow so `ReshapeOp` can participate in backend routing and destination lookup
  - updated routing/core allocation to reuse base weights with per-neuron `weight_skew` offsets instead of storing each shifted weight independently
  - threaded `add_potential` / `input_width` handling into frontend-core config, weight packing, and register generation
  - added SRAM accounting helpers for core placement summaries
- PR-writing decision:
  - title should emphasize backendv2 weight-skew support as the primary user-facing change
  - description should explicitly mention the supporting reshape/reorder routing refactor, because it is part of the same functional delivery
  - note that no tests are changed in this branch diff; verification should rely on the branch author's local/CI results if needed

# Converter Shape-Aux Generalization

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep `ReshapeOp` unchanged and limit the implementation to `converter.py` plus focused tests.

## Plan

- [x] Replace the current global shape-aux propagation heuristic with a reshape-sink-driven collector.
- [x] Broaden shape-aux coverage for common arithmetic nodes used in reshape size expressions, while keeping them ignored during `PAIIRGraph` wiring.
- [x] Add regression coverage for reshape/view shapes that depend on `add/sub/floordiv` rather than only `mul`.
- [x] Run focused verification and record results.

## Review

- Updated [converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/converter.py) so shape-aux recognition is now reshape-sink-driven instead of a global fixed-point over a tiny hardcoded subset.
- The new flow is:
  - identify reshape-like sinks (`flatten/view/reshape/view_as` and `torch.reshape` / `torch.flatten`)
  - collect only their non-data FX arguments as shape seeds
  - recursively walk upstream through shape-only helper nodes and mark them as `aux_bypass_nodes`
  - keep those helpers out of lowering and out of PAIIR data-flow wiring
- Broadened supported shape helper ops to include common arithmetic forms seen in shape expressions:
  - `getitem`
  - `add/sub`
  - `mul`
  - `floordiv`
  - `truediv`
  - `mod/remainder`
- Kept `ReshapeOp` unchanged, per the requested simplicity constraint.
- Added regression coverage in [test_graph_simulation.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_graph_simulation.py) for a `view(...)` shape expression that depends on `size + add/sub + mul + floordiv` before a `Linear`.
- Focused verification:
  - `./.venv/bin/python -m pytest tests/paiir/pipeline/test_graph_simulation.py -q -k 'flatten_linear_transition or view_and_view_as_routing_before_linear or view_shape_arithmetic_before_linear'`
    - result: `3 passed`
  - `ruff check paibox/paiir/lowering/converter.py tests/paiir/pipeline/test_graph_simulation.py`
    - result: `All checks passed`

# Converter Helper Documentation

# DualCNN Conv-BN-ReLU Fusion Investigation

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Do not create a dedicated `git worktree` for this task because the relevant `paiir` pipeline files and `tests/user/**` artifacts already have in-flight local changes that this investigation must compare against directly.

## Ownership

- Owner: Codex
- Branch: `bugfix-paiirgraph-sim-fixes`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files:
  - `tests/user/dualcnn.py`
  - `tests/user/**`
  - `tests/paiir/pipeline/**`
  - `tasks/todo.md`
- Blocked Files:
  - unrelated backend / onboard / docs files
  - unrelated generated artifacts outside the dualcnn debug scope
- Dependencies:
  - preserve the current contract that `torch_to_paiir()` returns an unfused atomic graph
  - preserve the current contract that `compile_to_paiir()` returns a fused compiled graph
  - keep the user-facing DualCNN helper on a single final compile path
- Verification:
  - focused pytest for any new dualcnn / fusion regression
  - rerun the dualcnn inspection script if needed

## Plan

- [x] Compare the dualcnn frontend and compiled summaries against the lowering and fusion pipeline behavior.
- [x] Add explicit dualcnn-stage diagnostics and/or regression assertions for Conv-BN-ReLU expectations.
- [x] Run focused verification and record the conclusion.

## Review

- Root cause:
  - `torch_to_paiir()` returns the atomic frontend graph by design, so `Conv/Linear` and `ReLU` still appear as `StandaloneCompOp -> StandaloneActOp`.
  - `BatchNorm1d/2d` is correctly bypassed during lowering and does not appear as a PAIIR node.
  - `compile_to_paiir()` then runs `fuse_to_offline_cores()`, which correctly turns those pairs into `SequentialOp`.
- Follow-up after user correction:
  - simplified `tests/user/dualcnn.py` to a single final `compile_to_paiir()` path
  - removed the extra `torch_to_paiir()` inspection pass from the helper and its regression
- DualCNN-specific verification outcome on the final compiled graph:
  - `BatchNorm` does not remain as a PAIIR node
  - `Conv2d=4`, `Conv1d=3`, `Linear=2` appear as `SequentialOp`
  - no `StandaloneActOp` remains after compilation
- Changes made:
  - `tests/user/dualcnn.py` now compiles once, writes `compiled_paiir_summary.log`, and asserts only the final compiled structure
  - `tests/user/test_dualcnn_compile.py` now validates only the final compiled graph
- Verification:
  - `ruff check tests/user/dualcnn.py tests/user/test_dualcnn_compile.py`
    - result: `All checks passed!`
  - `./.venv/bin/pytest tests/user/test_dualcnn_compile.py -q`
    - result: `1 passed`
  - `./.venv/bin/python tests/user/dualcnn.py`
    - result: structure report generated under `tests/user/debug/dualcnn_branch_pair_compile/compile_structure_report.log`
    - result: stale `frontend_paiir_summary.log` removed; output directory now contains only the final compiled summary plus the compiled-graph structure report

# PAIlib dev Branch Push

## Workspace Decision

- [ ] Continue from the current workspace as the coordination root, but operate on the existing sibling repo `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIlib`.
- [ ] Do not create a dedicated `git worktree` because the user explicitly asked to submit the current `dev` branch changes in place, and the change scope is a single in-flight release-style update.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIlib`
- Allowed Files:
  - `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIlib/.pre-commit-config.yaml`
  - `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIlib/pyproject.toml`
  - `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIlib/uv.lock`
  - `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIlib/frameParser/**` only if confirmed as part of this release
  - `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tasks/todo.md`
- Blocked Files:
  - `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIlib/test.py` unless the user later confirms it belongs in the push
  - unrelated local caches, build artifacts, and files under `PAIBox`
- Dependencies:
  - preserve the user's existing `dev` branch state in `PAIlib`
  - avoid accidentally committing local scratch files
  - push to `origin/dev`
- Verification:
  - inspect staged diff
  - run focused validation in `PAIlib` appropriate to the touched release metadata
  - confirm branch is ahead of `origin/dev` before push

## Plan

- [ ] Audit `PAIlib` working tree and define the safe commit boundary.
- [ ] Run focused verification for the selected changes.
- [ ] Commit the selected changes on `dev`.
- [ ] Push the new commit to `origin/dev`.

## Review

- Pending.

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this pass scoped to docstrings/comments in `converter.py` plus minimal lint verification.

## Plan

- [x] Add necessary docstrings/comments to the recently added reshape/shape-aux helpers.
- [x] Run minimal lint verification on the touched file.
- [x] Summarize the `_apply_shape_aux_rule` flow with direct code references.

## Review

- Added concise docstrings to the new reshape/shape-aux helpers in [converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/converter.py), including:
  - reshape sink classification
  - shape-seed extraction
  - shape-aux candidate collection
  - flatten lowering
  - reshape-like lowering
  - `_apply_shape_aux_rule(...)` itself
- Minimal verification:
  - `ruff check paibox/paiir/lowering/converter.py`
    - result: `All checks passed`

# Converter Module Classification Cleanup

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this cleanup scoped to naming/classification in `converter.py` plus focused verification.

## Plan

- [x] Separate tracer leaf-module classification from true lowering-time bypass modules.
- [x] Rename the helper/constant set so `nn.Flatten` is no longer described as a bypassed module.
- [x] Run focused verification covering both BN bypass and Flatten preservation.

## Review

- Updated [converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/converter.py) to separate two previously conflated concepts:
  - `TRACE_LEAF_MODULE_TYPES`: modules that should stay leaf during FX tracing so lowering can inspect them later
  - `LOWERING_BYPASS_MODULE_TYPES`: modules that should truly disappear from PAIIR data flow during lowering
- `nn.Flatten` now lives only in `TRACE_LEAF_MODULE_TYPES`, which matches its real behavior on this branch:
  - trace it as a leaf
  - lower it into `ReshapeOp`
  - do not treat it as a true bypass module
- `BatchNorm1d/2d` remain true lowering-time bypass modules.
- Renamed `_is_bypass_module(...)` to `_is_lowering_bypass_module(...)` to make the call site semantics explicit.
- Focused verification:
  - `./.venv/bin/python -m pytest tests/paiir/pipeline/test_passes.py -q -k 'conv_bn_relu or flatten_transition'`
    - result: `2 passed`
  - `ruff check paibox/paiir/lowering/converter.py`
    - result: `All checks passed`

## Workspace Decision

- [x] Continue in the current workspace because the user explicitly allowed writing on the current branch for this follow-up bugfix.
- [x] Rename branch from `bugfix-maxpool-signal-fix` to `bugfix-paiirgraph-sim-fixes` so the name still matches the now-broader simulation-fix scope.
- [x] Keep the write scope inside `paibox/paiir/**`, `tests/paiir/**`, `tests/user/**`, and `tasks/todo.md`.

# StandaloneActOp Signal-Domain Investigation

## Workspace Decision

- [x] Continue in the current workspace.
- [x] Keep this as a read-only investigation plus task-note update; no dedicated `git worktree` is needed because the goal is to answer a code-understanding question and avoid functional edits.

## Ownership

- Owner: Codex
- Branch: `bugfix-paiirgraph-sim-fixes`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files:
  - `paibox/paiir/**`
  - `tasks/todo.md`
- Blocked Files:
  - any functional code or test edits unless the investigation proves a bug that the user asks to fix
- Dependencies:
  - preserve current in-flight repository state
  - answer from actual code paths, not assumptions
- Verification:
  - source inspection across IR node definitions, lowering, and signal-domain propagation code

## Plan

- [x] Inspect `StandaloneActOp` structure and its signal-domain related fields.
- [x] Trace how `paiir` conversion constructs a standalone activation node when it appears alone in the graph.
- [x] Trace where input/output signal domains are inferred or propagated after conversion.
- [x] Record the conclusion with exact code references and any ambiguity boundary.

## Review

- `StandaloneActOp` itself does not store an `input_domain`; the IR base node only has `output_domain`.
- `torch_to_paiir(...)` creates explicit `InputNode` and `OutputNode` around the lowered op, so an activation-only model becomes `InputNode -> StandaloneActOp -> OutputNode`.
- Domain filling does not happen during `torch_to_paiir(...)`; it happens later in `propagate_signal_domain(...)`, which is called by `compile_to_paiir(...)`.
- In that pass:
  - every `InputNode` is currently assigned `SignalDomain.VALUE`
  - every `StandaloneActOp` falls under the generic `OfflineCoreOp` rule, so its `output_domain` is inferred from `node.neuron_params.output_type`
  - `CoreNeuronV25.to_neuron_params()` returns `NeuronParams(**kwargs)`, and `NeuronParams.output_type` defaults to `OutputType.VALUE`, so normal standalone activation nodes infer `output_domain=VALUE`
  - `OutputNode` then copies its predecessor domain, so it also becomes `VALUE`
- Practical conclusion:
  - for a normal activation-only converted graph, the effective input domain is inferable as predecessor `InputNode.output_domain == VALUE`
  - the op/output domain is inferable as `VALUE`
  - if someone manually constructs a graph containing only a disconnected `StandaloneActOp`, `propagate_signal_domain(...)` can still fill the node's own `output_domain`, but there is no explicit input-domain field to infer, and `validate_graph(...)` rejects the graph because it has no input/output nodes
- Local verification:
  - `OnlyIF` and `OnlyReLU` toy models both lowered to `InputNode -> StandaloneActOp -> OutputNode`
  - after `propagate_signal_domain(...)`, all three nodes resolved to `SignalDomain.VALUE`

# PAIIR Param Context Investigation

## Workspace Decision

- [x] Continue in the current workspace.
- [x] Keep this as a read-only investigation plus task-note update; no dedicated `git worktree` is needed because the goal is to answer parameter-propagation behavior from existing code.

## Ownership

- Owner: Codex
- Branch: `bugfix-paiirgraph-sim-fixes`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files:
  - `paibox/paiir/**`
  - `tasks/todo.md`
- Blocked Files:
  - functional code or test edits unless the user later asks for a fix
- Dependencies:
  - answer from actual pass behavior and parameter ownership
  - distinguish semantic domain annotations from backend format params
- Verification:
  - source inspection across calc params, op properties, compilation passes, and targeted local repros when needed

## Plan

- [x] Inspect `OfflineCoreParams` / `NeuronParams` definitions and ownership boundaries.
- [x] Trace which compilation passes mutate `core_params` from graph context.
- [x] Trace whether `neuron_params` is stored or dynamically derived from `act`.
- [x] Conclude whether graph context rewrites input/output types, with `StandaloneActOp` as the concrete example.

## Review

- `OfflineCoreParams` mixes:
  - semantic mode fields owned by the operator itself (`snn_mode`, `pooling_mode`, `add_potential`)
  - compile-time fields that are meant to be filled from graph context (`input_sign/width`, `output_sign/width`, `weight_sign/width`, `tick_*`)
- `override_compile_state_from(...)` explicitly preserves semantic fields and copies only non-semantic compile state, which confirms the intended ownership split.
- `NeuronParams` is not stored as mutable compile state on the node. For activation-bearing ops it is recomputed on every property access from the live `act` module via `act.to_neuron_params(...)`.
- Because `CoreNeuronV25.to_neuron_params()` does not pass any graph-context-derived `output_type`, activation-bearing ops inherit `NeuronParams.output_type=VALUE` from the dataclass default. Compute-only / add-pass-through ops explicitly return `NeuronParams(output_type=POTENTIAL)`.
- Generic graph passes do not rewrite `neuron_params.output_type` based on predecessor domains.
- What graph context does rewrite:
  - `propagate_signal_domain(...)` fills per-node semantic `output_domain`
  - `propagate_data_format(...)` fills `core_params.input_*` from predecessor output formats, `core_params.output_*` mostly from the node’s own activation semantics, and `core_params.weight_*` from weight ranges
  - `assign_tick_params(...)` fills `core_params.tick_*` from topology depth / ANN-vs-SNN policy
- Important nuance:
  - `core_params.output_*` is sometimes graph-sensitive for pass-through-like ops such as standalone MaxPool, where output format is inherited from predecessors
  - but `StandaloneActOp.neuron_params.output_type` is not predecessor-sensitive; if it consumes POTENTIAL and applies an activation, the predecessor domain stays represented by the predecessor `output_domain` and the node’s `core_params.input_*`, while the activation node still exports `output_type=VALUE`
- Local repro confirmed the intended split:
  - in a manual `StandaloneCompOp(Linear) -> StandaloneActOp(IF)` chain, after passes the comp node resolved to `output_domain=POTENTIAL`
  - the act node resolved to `output_domain=VALUE`
  - `act.neuron_params.output_type` stayed `VALUE`
  - `act.core_params.input_sign/input_width` were updated to match the predecessor’s output format

# propagate_data_format Documentation Touch-Up

## Workspace Decision

- [x] Continue in the current workspace.
- [x] Keep the change scoped to `paibox/paiir/pipeline/passes.py` plus task-note update; no dedicated `git worktree` is needed for this small documentation-only edit.

## Plan

- [x] Review the current `propagate_data_format(...)` implementation and identify the key steps worth documenting.
- [x] Expand the function docstring to describe responsibilities, pass ordering, and `input_formats` behavior.
- [x] Add concise inline comments for the important propagation stages.
- [x] Run focused lint verification on the touched file and record the result.

## Review

- Expanded `propagate_data_format(...)` in [passes.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/passes.py) with a fuller docstring that now explains:
  - which `OfflineCoreParams` fields the pass fills
  - why the algorithm uses two topological passes
  - how explicit `input_formats` overrides interact with default input inference
- Added concise inline comments around the key implementation stages:
  - unknown input-override warning behavior
  - the purpose of the `resolved` cache
  - pass-1 output/weight-format seeding
  - routing-node deferral to pass 2
  - pass-2 routing propagation and final input-format backfill
- Focused verification:
  - `ruff check paibox/paiir/pipeline/passes.py`
    - result: `All checks passed!`

## Task Block

- Owner: `codex`
- Branch: `bugfix-paiirgraph-sim-fixes`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files:
  - `paibox/paiir/lowering/**`
  - `paibox/paiir/pipeline/**`
  - `tests/paiir/**`
  - `tests/user/**`
  - `tasks/todo.md`
- Blocked Files:
  - `paibox/backend/**`
  - `paibox/backendv2/**`
- Dependencies:
  - preserve the already-present MaxPool signal-domain/data-format fixes on this branch
  - handle only bug 1 and bug 2 first: reshape/view lowering plus compiled-graph runtime-shape validation
- Verification:
  - focused `pytest` on touched PAIIR pass/simulation/user-example tests
  - `ruff check` on touched files

## Plan

- [x] Inspect current `converter.py` lowering around `flatten/view/reshape/view_as` and verify how the `DualCNN` FX graph expresses `view(x.size(0), -1)`.
- [x] Materialize reshape-like routing nodes for simulation instead of bypassing them, and make `size` shape-only so strict compilation does not fail on these nodes.
- [x] Strengthen `validate_compiled_graph()` so routing-node shape contracts catch bad executable graphs before runtime.
- [x] Update and extend tests for `flatten`, `view/view_as`, and compiled-graph concat-shape validation, then rerun focused verification.

## Review

- Updated [converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/converter.py) so:
  - `nn.Flatten`, `Tensor.flatten`, `Tensor.view`, `Tensor.reshape`, `Tensor.view_as`, and `torch.reshape` now lower to executable `ReshapeOp` nodes instead of being silently bypassed.
  - `size()` is treated as a shape-only bypass input to those reshape-like nodes, so `strict=True` no longer trips over `view(x.size(0), -1)` patterns.
  - reshape-like nodes wire only their real data predecessor into the `PAIIRGraph`, so shape-helper nodes no longer leak into runtime data flow.
- Updated [passes.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/passes.py) so `validate_compiled_graph()` now enforces executable routing contracts:
  - `ConcatOp` predecessor output shapes must match its declared `input_shapes`
  - `ConcatOp` metadata must describe a real concatenation result
  - `ReshapeOp` predecessor shape must match its declared input shape and preserve element count
- Updated tests:
  - [test_graph_simulation.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_graph_simulation.py)
    - unskipped and validated `SNNFlattenTransition`
    - added a `view(size(0), -1) + view_as(...) + Linear` simulation regression
  - [test_passes.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_passes.py)
    - now expects `flatten` to survive as a `ReshapeOp`
    - added a compiled-graph validation regression for concat predecessor shape mismatch
  - [test_paiir_frontend_example_graphs.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/test_paiir_frontend_example_graphs.py)
    - updated the reshape example to reflect that both explicit `reshape()` and `flatten()` now materialize routing nodes
- Verification:
  - `./.venv/bin/python -m pytest tests/paiir/pipeline/test_passes.py -q`
    - result: `72 passed`
  - `./.venv/bin/python -m pytest tests/user/test_paiir_frontend_example_graphs.py -q`
    - result: `4 passed`
  - `./.venv/bin/python -m pytest tests/paiir/pipeline/test_graph_simulation.py -q`
    - result: `40 passed, 1 failed, 2 skipped`
    - remaining failure is the known split-core `AvgPool+LIF` issue, which is outside this bugfix slice
  - `ruff check paibox/paiir/lowering/converter.py paibox/paiir/pipeline/passes.py tests/paiir/pipeline/test_passes.py tests/paiir/pipeline/test_graph_simulation.py tests/user/test_paiir_frontend_example_graphs.py`
    - result: `All checks passed`
- Residual note:
  - `DualCNN` no longer fails on `size/view_as` lowering or the old `ConcatOp` rank mismatch path.
  - It still hits a separate runtime issue on the current branch: `MaxPool1d` execution for `Byte` tensors in the 1D branch (`"max_pool1d_impl" not implemented for 'Byte'`). That is independent of the shape-lowering/validation fixes completed here.

# MaxPool Signal-Domain Fix

## Workspace Decision

- [x] Use a dedicated `git worktree` because this is a non-trivial PAIIR semantic fix that will touch core IR propagation rules and tests.
- [x] Branch: `bugfix-maxpool-signal-fix`
- [x] Worktree: `/tmp/PAIBox-codex-maxpool-signal-fix`
- [x] Base branch: local `dev`

## Task Block

- Owner: `codex`
- Branch: `bugfix-maxpool-signal-fix`
- Worktree: `/tmp/PAIBox-codex-maxpool-signal-fix`
- Allowed Files:
  - `paibox/paiir/ir/**`
  - `paibox/paiir/pipeline/**`
  - `tests/paiir/**`
  - `tasks/todo.md`
- Blocked Files:
  - `paibox/backend/**`
  - `paibox/backendv2/**`
- Dependencies:
  - keep the fix inside `paiir`
  - preserve existing AvgPool split/shared activation-special handling
- Verification:
  - focused `pytest` on touched PAIIR pass/data-format/simulation tests

## Plan

- [x] Inspect the minimal PAIIR code paths that currently force standalone pool outputs to `POTENTIAL` and `SIGNED/WIDTH_8BIT`.
- [x] Implement standalone pool output-domain propagation so value-domain pooling stays `VALUE` instead of defaulting to `POTENTIAL`.
- [x] Implement standalone pool output-format propagation so MaxPool preserves the predecessor value format instead of widening generically.
- [x] Add focused tests for `MaxPool` with `uint8`, `int8`, and `1bit` inputs, and verify no unrelated PAIIR pool behavior regresses.
- [x] Run focused verification and record results.

## Review

- Updated `paibox/paiir/pipeline/passes.py` so standalone `MaxPool1d/2d` nodes preserve predecessor signal-domain and predecessor data format instead of falling through the generic standalone-comp defaults.
- Kept the change scoped to `MaxPool`; `AvgPool` keeps its existing generic/shared-core/split-core handling because its output coding rules are not the same as max pooling.
- Updated `paibox/paiir/ir/op_node.py` so MaxPool simulation preserves integer input dtype through the pooling op, while spike-neuron activation inputs still get promoted into a safe membrane integer domain before IF/LIF execution.
- Added focused regression tests covering:
  - standalone MaxPool signal-domain preservation for both value-domain and potential-domain predecessors
  - MaxPool output-format preservation through a `MaxPool -> Conv2d -> ReLU` chain for `uint8`, `int8`, and `1bit` inputs
  - standalone MaxPool simulation preserving `uint8` / `int8` output dtype
- Verification:
  - `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/.venv/bin/pytest tests/paiir/pipeline/test_passes.py -q`
  - `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/.venv/bin/pytest tests/paiir/pipeline/test_data_format.py -q`
  - `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/.venv/bin/pytest tests/paiir/pipeline/test_graph_simulation.py -q -k 'maxpool_lif_snn or standalone_maxpool_preserves_integer_dtype'`
  - `/usr/bin/ruff check paibox/paiir/ir/op_node.py paibox/paiir/pipeline/passes.py tests/paiir/pipeline/test_passes.py tests/paiir/pipeline/test_data_format.py tests/paiir/pipeline/test_graph_simulation.py`

# DualCNN Last-Conv Int8 Dump

## Workspace Decision

- [x] Stay in the current workspace.
- [x] Scope is limited to `tests/user/dualcnn.py` plus task tracking in `tasks/todo.md`; no public API or broader integration change is involved.

## Plan

- [x] Inspect whether the repo already has a local convention for manual/static `int8` tensor quantization that should be mirrored here.
- [x] Update `tests/user/dualcnn.py` so `main()` statically quantizes `conv2d_last` and `conv1d_last` weights and bias tensors from `fp32` to `int8`.
- [x] Print the final quantized tensor `dtype` and values for both layers.
- [x] Execute the script as focused verification and record the result.

## Review

- Updated [dualcnn.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/dualcnn.py) so `main()` now performs per-tensor symmetric static quantization for `conv2d_last` and `conv1d_last` weights and bias with `scale = absmax / 127`, then rewrites those tensors back onto the two conv modules as `torch.int8` parameters.
- Enabled full tensor printing with `torch.set_printoptions(threshold=100_000, linewidth=160)` so the script prints the complete quantized values instead of summarized ellipses.
- Focused verification:
  - `./.venv/bin/python tests/user/dualcnn.py > /tmp/dualcnn_int8_dump.txt`
    - result: exit code `0`
    - output includes `Weight dtype: torch.int8` and `Bias dtype: torch.int8` for both `conv2d_last` and `conv1d_last`
    - full dump length: `41244` lines
  - `./.venv/bin/python -m py_compile tests/user/dualcnn.py`
    - result: exit code `0`
- `ruff check tests/user/dualcnn.py`
  - result: `All checks passed!`

# Shape-Helper Detection Refactor Plan

## Workspace Decision

- [x] Stay in the current workspace.
- [x] This step is design/planning only; no implementation files need to change yet beyond `tasks/todo.md`.

## Plan

- [ ] Identify the current coupling points between reshape lowering, shape-helper detection, and graph wiring in `converter.py`.
- [ ] Define a modular replacement for `_is_shape_aux_candidate` based on graph relations, user closure, and shape-expression semantics.
- [ ] Propose a staged migration plan that improves applicability without expanding `ReshapeOp` itself.

# Shape Analysis Module Extraction

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this step focused on extracting reshape sink / shape helper analysis out of `converter.py`, with only the minimum test updates needed to preserve current behavior.

## Plan

- [x] Introduce a dedicated `shape_analysis` module that owns reshape-sink detection and shape-helper collection.
- [x] Make `converter.py` consume analysis results (`reshape_sinks` + `aux_bypass_nodes`) instead of hosting the analysis logic inline.
- [x] Use Torch-propagated FX metadata as the first source for sink output shapes and shape-like scalar/container detection.
- [x] Add focused tests for the new module and rerun the existing targeted reshape/compile regressions.

## Review

- Added [shape_analysis.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/shape_analysis.py) as a dedicated local analysis module. It now owns:
  - reshape-like sink classification
  - extraction of the real tensor data input vs shape-only seed nodes
  - sink-driven shape-helper collection
  - a first-pass `meta["val"]` / `tensor_meta.shape` check so Torch-propagated metadata is used before target-name heuristics
- Updated [converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/converter.py) so:
  - the shape-helper logic is no longer implemented inline in the main lowering flow
  - `_LoweringContext` now stores `reshape_sinks`
  - `_apply_shape_aux_rule(...)` delegates to `analyze_shape_helpers(...)`
  - reshape lowering consumes `ReshapeSinkInfo` instead of reparsing FX nodes locally
- Kept `ReshapeOp` unchanged. This step only moved analysis out of the converter and kept the same thin routing-node construction model.
- Added [test_shape_analysis.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/lowering/test_shape_analysis.py) with direct unit coverage for:
  - `nn.Flatten` sink detection
  - `view(...)` shape arithmetic analysis
- Updated [test_compile.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_compile.py) to use the renamed leaf-module constant and to keep the “shape-only args do not become data predecessors” assertion at the lowering level without accidentally depending on a separate unsupported `unsqueeze/repeat` compile-path issue.
- Focused verification:
  - `./.venv/bin/python -m pytest tests/paiir/lowering/test_shape_analysis.py -q`
    - result: `2 passed`
  - `./.venv/bin/python -m pytest tests/paiir/pipeline/test_compile.py -q -k 'shape_only_reshape_args_do_not_become_data_predecessors or analysis_prebuilds_functional_conv_and_marks_shape_aux_nodes'`
    - result: `2 passed`
  - `./.venv/bin/python -m pytest tests/paiir/pipeline/test_graph_simulation.py -q -k 'flatten_linear_transition or view_and_view_as_routing_before_linear or view_shape_arithmetic_before_linear'`
    - result: `3 passed`
  - `./.venv/bin/python -m pytest tests/user/test_paiir_frontend_example_graphs.py -q -k reshape_op_example_graph`
    - result: `1 passed`
  - `ruff check paibox/paiir/lowering/converter.py paibox/paiir/lowering/shape_analysis.py tests/paiir/pipeline/test_compile.py tests/paiir/lowering/test_shape_analysis.py`
    - result: `All checks passed`

# Shape Analysis User Closure

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this step focused on tightening `shape_analysis.py` without expanding `ReshapeOp` or touching deployment/back-end code.

## Plan

- [x] Add user-closure pruning to shape-helper collection so nodes shared with non-shape consumers are not silently marked auxiliary.
- [x] Add direct unit coverage for a reshape shape node reused by a non-shape consumer.
- [x] Run focused verification on shape-analysis, compile, and reshape simulation tests.

## Review

- Updated [shape_analysis.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/shape_analysis.py) so shape-helper collection is now conservative in a second dimension beyond backward-slice reachability:
  - after collecting candidate shape helpers from reshape seeds
  - it now prunes any candidate whose value is also consumed by a user outside the candidate set and outside reshape sinks
- This new user-closure step prevents shared nodes from being silently marked auxiliary when they also feed a non-shape path such as a real output or downstream computation.
- Added direct unit coverage in [test_shape_analysis.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/lowering/test_shape_analysis.py):
  - a `size(0)` node reused both by `view(...)` and by a normal tuple output is now confirmed to stay out of `aux_nodes`
- Focused verification:
  - `./.venv/bin/python -m pytest tests/paiir/lowering/test_shape_analysis.py -q`
    - result: `3 passed`
  - `./.venv/bin/python -m pytest tests/paiir/pipeline/test_compile.py -q -k 'shape_only_reshape_args_do_not_become_data_predecessors or analysis_prebuilds_functional_conv_and_marks_shape_aux_nodes'`
    - result: `2 passed`
  - `./.venv/bin/python -m pytest tests/paiir/pipeline/test_graph_simulation.py -q -k 'flatten_linear_transition or view_and_view_as_routing_before_linear or view_shape_arithmetic_before_linear'`
    - result: `3 passed`
  - `ruff check paibox/paiir/lowering/shape_analysis.py tests/paiir/lowering/test_shape_analysis.py`
    - result: `All checks passed`

# Shape Analysis Interface Cleanup

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this step focused on API cleanup between `shape_analysis.py` and `converter.py`; no semantic widening is intended.

## Plan

- [x] Replace the tuple return of `analyze_shape_helpers(...)` with an explicit analysis-result object.
- [x] Make `converter.py` consume that result object instead of directly depending on raw sink/aux containers.
- [x] Run focused verification on shape-analysis, compile, and reshape simulation regressions.

## Review

- Introduced `ShapeAnalysisResult` in [shape_analysis.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/shape_analysis.py) so the shape-analysis API now returns one explicit object instead of a raw `(reshape_sinks, aux_nodes)` tuple.
- `ShapeAnalysisResult` exposes:
  - `reshape_sinks`
  - `aux_nodes`
  - `sink_for(node)`
  - `is_aux(node)`
- Updated [converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/converter.py) so:
  - `_LoweringContext` stores `shape_analysis`
  - `_apply_shape_aux_rule(...)` saves the analysis result and merges only `analysis.aux_nodes` into `ctx.aux_bypass_nodes`
  - reshape lowering now uses `_get_reshape_sink_info(...)` instead of directly indexing a raw `reshape_sinks` dict
- Updated tests to consume the new interface rather than tuple unpacking or direct context dict access where possible:
  - [test_shape_analysis.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/lowering/test_shape_analysis.py)
  - [test_compile.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_compile.py)
- Focused verification:
  - `./.venv/bin/python -m pytest tests/paiir/lowering/test_shape_analysis.py -q`
    - result: `4 passed`
  - `./.venv/bin/python -m pytest tests/paiir/pipeline/test_compile.py -q -k 'shape_only_reshape_args_do_not_become_data_predecessors or analysis_prebuilds_functional_conv_and_marks_shape_aux_nodes'`
    - result: `2 passed`
  - `./.venv/bin/python -m pytest tests/paiir/pipeline/test_graph_simulation.py -q -k 'flatten_linear_transition or view_and_view_as_routing_before_linear or view_shape_arithmetic_before_linear'`
    - result: `3 passed`
  - `ruff check paibox/paiir/lowering/converter.py paibox/paiir/lowering/shape_analysis.py tests/paiir/lowering/test_shape_analysis.py tests/paiir/pipeline/test_compile.py`
    - result: `All checks passed`

# Chained Shape-Transform Tests

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this step scoped to reshape/flatten test coverage only; no production behavior change is intended.

## Plan

- [x] Add shape-analysis tests for chained size transforms such as `flatten -> reshape`.
- [x] Add a graph-simulation regression for a chained `flatten -> reshape(with size arithmetic) -> flatten -> Linear` path.
- [x] Run focused verification and record results.

# MaxPool1d Integer Simulation Fix

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this step scoped to PAIIR simulation behavior (`op_node.py`) plus focused tests and local repros.

## Plan

- [x] Fix `MaxPool1d` simulation for integer inputs so graph execution does not depend on unsupported PyTorch `Byte/Char` kernels.
- [x] Add focused tests for standalone integer `MaxPool1d` and the `DualCNN`-style 1D branch behavior.
- [x] Re-run the relevant graph simulation and local repro checks.

## Review

- Root cause confirmation:
  - the `DualCNN` failure was caused by `_run_comp(...)`, not `_prepare_act_input(...)`
  - `_run_comp(...)` passed integer `uint8` / `int8` tensors directly into `nn.MaxPool1d`
  - current PyTorch CPU kernels reject that path with:
    - `NotImplementedError: "max_pool1d_impl" not implemented for 'Byte'`
    - `NotImplementedError: "max_pool1d_impl" not implemented for 'Char'`
- Applied a minimal fix in [op_node.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/op_node.py):
  - `MaxPool1d` now keeps floating inputs unchanged
  - for integer inputs only, it runs pooling in `float32` and casts the exact max values back to the original integer dtype
  - `MaxPool2d` behavior remains unchanged
- Added focused regression coverage in [test_graph_simulation.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_graph_simulation.py):
  - existing `MaxPool2d` integer-dtype preservation test still passes
  - new `test_standalone_maxpool1d_preserves_integer_dtype(...)` covers `uint8` and `int8`
- Local repro on the real `DualCNN` graph now runs through all scheduled steps:
  - `compile_to_paiir(DualCNN(), x2d, x1d, strict=True)` succeeds
  - `graph.step(x2d, x1d)` succeeds through `max_tick_start = 12`
  - final output shapes: `(1, 5)` and `(1, 2)`
- Focused verification:
  - `./.venv/bin/python -m pytest tests/paiir/pipeline/test_graph_simulation.py -q -k 'standalone_maxpool1d_preserves_integer_dtype or standalone_maxpool_preserves_integer_dtype'`
    - result: `4 passed`
  - local `DualCNN` smoke:
    - `compile_to_paiir(DualCNN(), x2d, x1d, strict=True)` + 12 `graph.step(...)`
    - result: all steps passed
  - `ruff check paibox/paiir/ir/op_node.py tests/paiir/pipeline/test_graph_simulation.py`
    - result: `All checks passed`

## Review

- Added chain-oriented shape-analysis coverage in [test_shape_analysis.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/lowering/test_shape_analysis.py):
  - `flatten -> reshape` with fixed reshape sizes records two ordered reshape sinks
  - `flatten -> reshape(x.size(...), ..., x.size(...) // 2)` correctly treats the post-flatten `size(...)` nodes as shape-only auxiliaries
- Added a graph-simulation regression in [test_graph_simulation.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_graph_simulation.py):
  - chained `flatten -> reshape(with size arithmetic) -> flatten -> Linear`
  - verifies the compiled `PAIIRGraph` materializes three `ReshapeOp` nodes and matches PyTorch output exactly
- Focused verification:
  - `./.venv/bin/python -m pytest tests/paiir/lowering/test_shape_analysis.py -q -k 'flatten_then_fixed_reshape_records_two_sinks or flatten_then_reshape_shape_arithmetic_marks_flatten_size_users_aux'`
    - result: `2 passed`
  - `./.venv/bin/python -m pytest tests/paiir/pipeline/test_graph_simulation.py -q -k 'flatten_then_reshape_chain_before_linear'`
    - result: `1 passed`
  - `ruff check tests/paiir/lowering/test_shape_analysis.py tests/paiir/pipeline/test_graph_simulation.py`
    - result: `All checks passed`

# Torch Compile Shape Handling Research

## Workspace Decision

- [x] Stay in the current workspace.
- [x] This is read-only local research against the installed `torch` sources plus planning notes in `tasks/todo.md`.

## Plan

- [ ] Inspect local `torch` compile/FX/inductor code for reshape/view simplification and symbolic-shape handling.
- [ ] Identify the module boundaries Torch uses for shape reasoning vs graph rewriting.
- [ ] Summarize concrete ideas worth borrowing for PAIIR's shape-helper detection.

# DualCNN Last-Conv Pair PAIIR + backendv2 Compile Helper

## Workspace Decision

- [x] Stay in the current workspace.
- [x] Scope started as a local helper in `tests/user/dualcnn.py`; during verification it exposed a package-level import cycle that required the smallest shared unblocker in `paibox/paiir/pipeline/**` so the script could actually import PAIIR/backendv2 entry points.
- [x] User clarified the compile unit must be one combined `2`-input / `2`-output network, not two separately compiled layers.

## Plan

- [x] Confirm the smallest working `compile_to_paiir(...) -> backendv2.Mapper().compile(...)` flow for a combined `Conv2d + Conv1d` wrapper model with `2` inputs and `2` outputs.
- [x] Replace the current per-layer helper in `tests/user/dualcnn.py` with a combined-network helper that compiles `conv2d_last` and `conv1d_last` together from `main()`.
- [x] Keep the function output explicit enough to show the combined graph inputs, outputs, and backendv2 routing-group result.
- [x] Run focused verification and record the result.
- [x] Correct the default paired-compile sample shapes to the real in-network shapes from `tests/user/dualcnn.log`.
- [x] Add a backendv2 preflight estimate so the script reports the real-size backend memory barrier instead of silently compiling with placeholder reduced shapes.

## Review

- Updated [pipeline/**init**.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/__init__.py) and [avgpool/**init**.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/avgpool/__init__.py) to use lazy exports via `__getattr__`, which removes the package import cycle triggered when `op_node.py` imports `pipeline.avgpool.metadata`.
- Updated [dualcnn.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/dualcnn.py) so the compile helper now builds one paired wrapper module `forward(x2d, x1d) -> (conv2d_last(x2d), conv1d_last(x1d))` instead of compiling the two layers separately.
- The new helper `compile_last_conv_pair_with_paiir_and_backendv2(...)`:
  - compiles the paired wrapper through `compile_to_paiir(...)` with two sample inputs
  - uses the true in-network input shapes from [dualcnn.log](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/dualcnn.log): `conv2d_last -> (1, 64, 10, 32)` and `conv1d_last -> (1, 64, 64)`
  - computes a backendv2 dense-routing preflight estimate before attempting the backend stage
  - only runs `backendv2.Mapper().compile(...)` when that estimate stays under the configured safety limit, or when explicitly forced
  - writes `paiir_summary.log` and `backendv2.log` next to that output directory
- Focused verification:
  - minimal smoke test: a combined `Conv2d + Conv1d` wrapper compiles successfully through `compile_to_paiir(...)`
    - result: graph contains `2` input nodes and `2` output nodes
    - result: `backendv2.Mapper().compile(...)` accepts the combined graph and produces `2` routing groups
  - `./.venv/bin/python -m py_compile tests/user/dualcnn.py`
    - result: exit code `0`
  - `ruff check tests/user/dualcnn.py tasks/lessons.md`
    - result: `All checks passed!`
  - `timeout 120s ./.venv/bin/python tests/user/dualcnn.py > /tmp/dualcnn_pair_compile.log`
    - result: exit code `0`
    - output confirms corrected logged shapes `2D sample shape: (1, 64, 10, 32)` and `1D sample shape: (1, 64, 64)`
    - output confirms paired compile path with `PAIIR inputs: ['InputNode_0', 'InputNode_1']`
    - output confirms paired compile path with `PAIIR outputs: ['OutputNode_0', 'OutputNode_1']`
    - output reports the real backendv2 dense-routing estimate:
      - `conv2d_last`: `20480 x 40960 -> 3.125 GiB`
      - `conv1d_last`: `4096 x 8192 -> 0.125 GiB`
      - total: `3.250 GiB`
    - backend stage is skipped with a clear log because the estimate exceeds the `1.000 GiB` safety limit
    - artifacts written under `tests/user/debug/dualcnn_last_conv_pair_compile/`
  - forced actual-size backend attempt:
    - `timeout 120s ./.venv/bin/python -u - <<'PY' ... force_large_backendv2=True ... PY`
    - result: exit code `124` from `timeout`
    - observed stdout before the backend stage: `starting forced real-size backend compile`
    - no `backendv2.log` was written under `tests/user/debug/dualcnn_last_conv_pair_compile_force/`, while `paiir_summary.log` was written
    - interpretation: the run reached PAIIR graph generation and then remained stuck inside the real-size backendv2 compile long enough to hit the timeout; this test did not produce an observed OOM kill

# AvgPool Window-Helper Typing Tightening

## Workspace Decision

- [x] Stay in the current workspace because this is a small PAIIR typing cleanup scoped to `paibox/paiir/pipeline/avgpool/**`.

## Plan

- [x] Audit `_get_pool_window_size(...)` call sites to confirm whether `SumPool*` inputs still occur.
- [x] Tighten the helper signature to the minimal explicit AvgPool union and remove no-longer-needed imports/assertions.
- [x] Run focused verification on the touched avgpool helper surface.

# PAIIR API Surface Audit

## Workspace Decision

- [x] Stay in the current workspace.
- [x] This task is a source audit and API design review; expected write scope is limited to `tasks/todo.md` unless the audit reveals a small unblocker worth fixing immediately.

## Plan

- [x] Inspect the `paibox/paiir` package structure and each relevant `__init__.py` export surface.
- [x] Check the real import path for `compile_to_paiir` and confirm whether any package-level import issues remain.
- [x] Evaluate whether `paibox/paiir/pipeline/compile.py` belongs in `pipeline/` or should move to the package root.
- [x] Record concrete recommendations and residual risks.

## Review

- Package/API surface by layer:
  - [paibox/paiir/**init**.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/__init__.py) is the public convenience facade. It eagerly re-exports IR types from `ir/`, compile/data-format APIs from `pipeline/`, and lowering entrypoints from `lowering/`.
  - [paibox/paiir/ir/**init**.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/__init__.py) is a cohesive IR subpackage facade. Its eager export pattern is defensible because the subpackage contents are tightly related and already mutually coupled.
  - [paibox/paiir/lowering/**init**.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/__init__.py) is a thin frontend-lowering facade and looks structurally fine.
  - [paibox/paiir/nn/**init**.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/nn/__init__.py) is minimal and fine.
  - [paibox/paiir/pipeline/**init**.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/__init__.py) is now a lazy facade via `__getattr__`, which is the right shape for a package that otherwise risks import cycles between compile/passes/data-format and IR modules.
  - [paibox/paiir/pipeline/avgpool/**init**.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/avgpool/__init__.py) also uses lazy exports; this is appropriate because its submodules participate in the same cycle-prone compile-time surface.
- Import audit:
  - `from paibox.paiir import compile_to_paiir` succeeds.
  - `from paibox.paiir.pipeline import compile_to_paiir` succeeds.
  - related smoke imports from `ir`, `lowering`, and `pipeline.avgpool` also succeed.
  - So `compile_to_paiir` import is currently functional; there is no remaining reproducible import failure in the audited paths.
- Design assessment:
  - [pipeline/compile.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/compile.py) is conceptually in the right place. Its job is to orchestrate compile-time passes, so it belongs to `pipeline/`, not the package root.
  - Moving the implementation file to `paibox/paiir/compile.py` would blur layering by mixing the root package facade with pass orchestration internals.
  - The better improvement is to keep the implementation in `pipeline/compile.py` and, if desired, make [paibox/paiir/**init**.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/__init__.py) a lazy facade too, for consistency with `pipeline/` and to reduce import-time coupling.
- Recommendation:
  - keep `compile_to_paiir` implemented in `pipeline/compile.py`
  - keep re-exporting it from `paibox.paiir` as the primary user-facing import path
  - consider a follow-up refactor to make the top-level `paibox.paiir` facade lazy instead of eagerly importing a broad symbol surface
  - avoid moving implementation files upward solely for import convenience; use re-exports or a tiny shim module instead

# DualCNN Recheck

## Workspace Decision

- [x] Stay in the current workspace.
- [x] This step is verification-only for `tests/user/dualcnn.py` after the user's local fixes.

## Plan

- [x] Re-inspect the current `dualcnn.py` helper path.
- [x] Re-run focused checks (`ruff`, `py_compile`, import, script execution).
- [x] Record the current status and remaining risk.

## Review

- Current status:
  - `tests/user/dualcnn.py` imports successfully.
  - `ruff check tests/user/dualcnn.py` passes.
  - `./.venv/bin/python -m py_compile tests/user/dualcnn.py` passes.
  - `timeout 60s ./.venv/bin/python tests/user/dualcnn.py` exits with code `0`.
- Residual risk:
  - the paired compile helper still does not execute the real backendv2 compile for the logged true shapes by default.
  - In [dualcnn.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/dualcnn.py), the guard around `backend_skipped` intentionally stops before `Mapper().compile(graph)` when the dense routing estimate exceeds the configured limit.
  - With the current logged shapes, the script reports:
    - `conv2d`: `20480 x 40960 -> 3.125 GiB`
    - `conv1d`: `4096 x 8192 -> 0.125 GiB`
    - total: `3.250 GiB`
  - So the current script is now healthy as a diagnostic/preflight tool, but it is not yet an end-to-end backend compile of the real-size paired network.

# DualCNN Conv+ReLU Pair Refinement

## Workspace Decision

- [x] Stay in the current workspace.
- [x] Scope is limited to `tests/user/dualcnn.py` plus task tracking in `tasks/todo.md`.

## Plan

- [x] Replace the current deploy wrapper so it represents `conv2d_last + relu` and `conv1d_last + relu`, not conv-only branches.
- [x] Keep the paired `2`-input / `2`-output compile path and existing real-shape/backend preflight behavior intact.
- [x] Run focused verification and record the result.

## Review

- Updated [dualcnn.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/dualcnn.py) so the paired deploy wrapper is now `conv2d_last + relu` on the 2D branch and `conv1d_last + relu` on the 1D branch.
- In `main()`, the wrapper now pulls:
  - `conv2d_last = model.cnn2d[-5]`
  - `conv2d_relu = model.cnn2d[-3]`
  - `conv1d_last = model.cnn1d[-5]`
  - `conv1d_relu = model.cnn1d[-3]`
- Focused verification:
  - `ruff check tests/user/dualcnn.py`
    - result: `All checks passed!`
  - `./.venv/bin/python -m py_compile tests/user/dualcnn.py`
    - result: exit code `0`
  - `timeout 60s ./.venv/bin/python tests/user/dualcnn.py > /tmp/dualcnn_conv_relu.log`
    - result: exit code `0`
    - output confirms paired compile path still produces `2` inputs and `2` outputs
- [paiir_summary.log](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/debug/dualcnn_last_conv_pair_compile/paiir_summary.log)
  - result: both branches now lower as `SequentialOp (Conv2d -> ANNNodeV25)` and `SequentialOp (Conv1d -> ANNNodeV25)`, which confirms the added `ReLU` is represented in the compiled paired subnetwork

# DualCNN Full-Network PAIIR Compile Helper

## Workspace Decision

- [x] Stay in the current workspace.
- [x] Scope is limited to `tests/user/dualcnn.py` plus task tracking in `tasks/todo.md`.

## Plan

- [x] Add a full-network DualCNN PAIIR compile helper for graph inspection.
- [x] Emit both frontend-lowered and compiled PAIIR summaries so the whole-network graph can be checked more easily.
- [x] Run focused verification and record the result.

## Review

- Updated [dualcnn.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/user/dualcnn.py) with a whole-network helper, which compiles the full `DualCNN` at the real input sizes:
  - `x2d`: `(1, 1, 80, 256)`
  - `x1d`: `(1, 4, 256)`
- The helper now emits two graph artifacts under `tests/user/debug/dualcnn_branch_pair_compile/`:
  - `frontend_paiir_summary.log` from `torch_to_paiir(...)`
  - `compiled_paiir_summary.log` from `compile_to_paiir(...)`
- Focused verification:
  - `ruff check tests/user/dualcnn.py`
    - result: `All checks passed!`
  - `./.venv/bin/python -m py_compile tests/user/dualcnn.py`
    - result: exit code `0`
  - whole-model smoke checks:
    - `compile_to_paiir(model, x2d, x1d)` succeeded with `26` nodes and `25` edges
    - `torch_to_paiir(model, x2d, x1d)` succeeded with `35` nodes and `34` edges
  - `timeout 90s ./.venv/bin/python tests/user/dualcnn.py > /tmp/dualcnn_full_helper.log`
    - result: exit code `0`
    - output confirms both full-network summary files were written

## Review

- Audited all current `_get_pool_window_size(...)` call sites under `paibox/paiir`: each call is dominated by `_is_avgpool(...)`, so the runtime caller set is `nn.AvgPool1d | nn.AvgPool2d` only.
- Tightened [utils.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/avgpool/utils.py) accordingly:
  - removed the unused `SumPool1d` / `SumPool2d` imports
  - changed `_get_pool_window_size(...)` from `nn.Module` + runtime `assert` to the explicit union `nn.AvgPool1d | nn.AvgPool2d`
  - simplified the `kernel_size` normalization branch to `AvgPool2d` vs `AvgPool1d`
- Focused verification:
  - `/usr/bin/ruff check paibox/paiir/pipeline/avgpool/utils.py paibox/paiir/pipeline/avgpool/fusion.py paibox/paiir/pipeline/avgpool/pass_ops.py`
    - result: passed
  - `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k avgpool`
    - result: blocked by an unrelated collection-time import error in `tests/paiir/pipeline/test_compile.py` (`converter.py` export mismatch for `BYPASS_MODULE_TYPES`)
  - `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/.venv/bin/pytest tests/paiir/pipeline/avgpool/test_compensation.py tests/paiir/pipeline/avgpool/test_deploy_scheme.py tests/paiir/pipeline/avgpool/test_transient.py -q`
    - result: failed on pre-existing `FrozenInstanceError` mutations in `avgpool/compensation.py` tests that do not exercise `_get_pool_window_size(...)` typing

# PAIIR Reshape Review

## Workspace Decision

- [x] Stay in the current workspace.
- [x] This task is a read-mostly code review of existing `paiir` reshape lowering and tests, so a dedicated `git worktree` is unnecessary.

## Ownership

- Owner: Codex
- Branch: `bugfix-paiirgraph-sim-fixes`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files:
  - `paibox/paiir/**`
  - `tests/paiir/**`
  - `tasks/todo.md`
- Blocked Files:
  - unrelated backend/onboard/docs changes already present in the dirty worktree
- Dependencies:
  - treat this as review-first; do not change reshape behavior unless a minimal review artifact is required
  - evaluate both lowering correctness and simulation/compile test coverage
- Verification:
  - focused pytest covering reshape/view/flatten lowering, dims propagation, compile wiring, and graph simulation

## Plan

- [x] Trace `ReshapeOp` generation from shape analysis through `converter.py` into IR execution.
- [x] Audit existing `paiir` tests for `reshape/view/flatten` across lowering, compile, pass, and simulation layers.
- [x] Run focused tests and record whether handling is correct and sufficiently covered.

## Review

- Generation path is structurally coherent:
  - `shape_analysis.py` recognizes `nn.Flatten`, method `flatten/reshape/view/view_as`, and function `torch.flatten/torch.reshape` as reshape-like sinks.
  - `converter.py` lowers flatten-like sinks to `ReshapeOp(shape_fn=_make_flatten_shape_fn(...))` and other reshape-like sinks to fixed-shape `ReshapeOp` using FX-propagated `output_shape`.
  - shape-only helper nodes are excluded from data-flow wiring, and compile wiring tests confirm reshape-size expressions do not become real graph predecessors.
- Confirmed via focused runtime checks that `compile_to_paiir(...)` materializes `ReshapeOp` for both `nn.Flatten` and method `reshape(...)`.
- Main correctness gap:
  - `DimsProp` does not treat `view_as`, `torch.flatten`, or `torch.reshape` as reshape-like, so after a transpose/permute it incorrectly preserves the old axis order instead of resetting to identity.
  - Reproducer observed `output dims == (0, 2, 1)` for all three cases rather than the expected reshaped identity ordering.
- Main completeness gap:
  - non-flatten reshape lowering depends on FX shape metadata; when `torch_to_paiir(...)` is called without `sample_inputs`, `_build_reshape_op(...)` returns `None`, so `view/reshape/view_as` become unsupported and are bypassed under `strict=False`.
  - `flatten` is better covered because it carries a runtime `shape_fn`; other reshape-like forms do not.
- Existing tests are meaningful but incomplete:
  - covered: shape analysis for `nn.Flatten`, dims reset for method `flatten/view/reshape`, compile wiring for shape-only reshape args, fused flatten path, graph simulation for `flatten`, `view`, `view_as`, and shape arithmetic feeding `view`.
  - missing: dims tests for `view_as` and function-form `torch.flatten` / `torch.reshape`, compile/simulation tests for module `nn.Flatten`, direct IR-level `ReshapeOp` unit tests, and tests for the no-`sample_inputs` reshape path.
- Focused verification:
  - `./.venv/bin/pytest tests/paiir/lowering/test_shape_analysis.py tests/paiir/lowering/test_dims_prop.py tests/paiir/pipeline/test_compile.py tests/paiir/pipeline/test_passes.py tests/paiir/pipeline/test_graph_simulation.py -q -k 'reshape or flatten or view'`
    - result: `12 passed, 205 deselected`
  - ad hoc compile smoke:
    - `FlattenModule` compiled with `['ReshapeOp_0']`
    - `ReshapeMethod` compiled with `['ReshapeOp_1']`
  - ad hoc dims reproducer:
    - `view_as`, `torch.flatten`, `torch.reshape` after `transpose` all retained `(0, 2, 1)` dims, confirming the `DimsProp` mismatch
  - ad hoc no-shape-meta reproducer:
  - `torch_to_paiir(ReshapeNoMeta(), strict=False)` produced only `InputNode_0` and `OutputNode_0` and warned `view` was unsupported

# BackendV2 Export Path Investigation

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this in the current workspace because the issue is a narrow cross-file fix in `backendv2` and the user-side helper under `tests/user`.

## Plan

- [x] Confirm whether the current `DualCNN` path actually calls `Mapper.compile()`.
- [x] Trace `backendv2.Mapper.compile()` export behavior and real output directory.
- [x] Fix the export path mismatch and verify real files are generated where the helper reports.

## Review

- Root cause was twofold:
  - the then-current full-network helper only ran `compile_to_paiir()` and never entered `backendv2.Mapper.compile()`
  - `paibox/backendv2/mapper.py` hardcoded export output to `./output`, while `tests/user/dualcnn.py` reported the path as `output_root / "output"`
- Additional backendv2 blocker:
  - `paibox/backendv2/get_weight.py` still imported IR node types from the narrowed `paibox.paiir` top-level API, which broke `Mapper` import after the API-surface cleanup
- Changes made:
  - `paibox/backendv2/get_weight.py` now imports IR node classes from `paibox.paiir.ir`
  - `paibox/backendv2/mapper.py` now accepts `output_path` in `Mapper.compile(...)` and exports there instead of hardcoding `./output`
  - `tests/user/dualcnn.py` now passes `output_root / "output"` into `mapper.compile(...)`, so the printed path matches the real files
- Verification:
  - `ruff check paibox/backendv2/get_weight.py paibox/backendv2/mapper.py tests/user/dualcnn.py`
    - result: `All checks passed!`
  - executed `compile_last_conv_pair_with_paiir_and_backendv2(...)` with the real saved DualCNN weights
    - result: backendv2 export files were generated under `tests/user/debug/dualcnn_last_conv_pair_compile/output`
    - generated files:
      - `frame_type1.txt`
      - `frame_type2.txt`
      - `frame_type3.txt`
      - `frame_type1.h`
      - `frame_type2.h`
      - `frame_type3.h`
  - follow-up per user request:
    - reverted the `Mapper.compile(..., output_path=...)` interface change and restored backendv2 export to the legacy `./output` path
    - kept the separate `backendv2/get_weight.py -> paibox.paiir.ir` import narrowing, because it fixes backendv2 imports independently of the `Mapper` export-path change
    - verified `from paibox.backendv2.mapper import Mapper` succeeds
    - verified `tests/user/dualcnn.py` full-network final PAIIR compile still succeeds unchanged

# DualCNN Branch Backend Compile

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this in the current workspace because the work is a focused update to the existing user helper and does not justify splitting away from the in-flight PAIIR/backendv2 changes already present here.

## Plan

- [x] Confirm why full-network backendv2 compile fails on the current compiled DualCNN graph.
- [x] Replace the full-network helper with a two-input/two-output branch-only network that excludes `ConcatOp`.
- [x] Redirect backendv2 stdout into log files and reduce low-value helper prints.
- [x] Run the real-weight branch compile and record whether backendv2 executes or is safely blocked by preflight size limits.

## Review

- The original full compiled DualCNN graph failed backendv2 node lowering at `ConcatOp`, which is not implemented in `paibox/backendv2/op_node.py`.
- `compile_dualcnn_branch_pair_with_paiir_and_backendv2(...)` now compiles a branch-only subnetwork:
  - input 0 -> full `cnn2d` branch -> flatten -> output 0
  - input 1 -> full `cnn1d` branch -> flatten -> output 1
  - this removes the unsupported concat/head portion while preserving the two-branch structure the user requested
- Helper output was simplified:
  - backendv2 stdout is now redirected into `backendv2.log`
  - terminal output only prints the output root, artifact paths, and backend status
- Real-weight / real-shape branch compile result:
  - compiled PAIIR branch graph was generated successfully with 20 nodes / 18 edges
  - backendv2 did not run, but it was blocked intentionally by the new preflight guard rather than failing mid-compile
  - the current backendv2 dense expansion model would require about `90.906 GiB` of int32 routing buffers for the branch network
  - dominant cost is the 2D branch early conv stack:
    - `cnn2d[0]`: `25.000 GiB`
    - `cnn2d[5]`: `50.000 GiB`
    - `cnn2d[10]`: `12.500 GiB`
    - `cnn2d[15]`: `3.125 GiB`
- Artifacts:
  - `tests/user/debug/dualcnn_branch_pair_compile/compiled_paiir_summary.log`
  - `tests/user/debug/dualcnn_branch_pair_compile/compile_structure_report.log`
  - `tests/user/debug/dualcnn_branch_pair_compile/backendv2.log`
- Verification:
  - `ruff check tests/user/dualcnn.py`
    - result: `All checks passed!`
  - `./.venv/bin/python tests/user/dualcnn.py`
    - result: branch-only compiled graph emitted successfully
    - result: backendv2 preflight logged and compile safely skipped due the `1.000 GiB` guard

# DualCNN Last-Conv Pair Real-Shape Compile

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] No dedicated worktree is needed because this is an execution/verification task against the existing helper, not a parallel write-heavy change.

## Plan

- [x] Clean the previous `dualcnn_last_conv_pair_compile` output.
- [x] Run `compile_last_conv_pair(...)` with the real logged feature-map shapes and saved weights.
- [x] Verify generated backend artifacts and record their paths.

## Review

- Executed `compile_last_conv_pair(...)` with the real saved DualCNN weights and the logged in-network last-layer feature-map sizes:
  - 2D input shape: `(1, 64, 10, 32)`
  - 1D input shape: `(1, 64, 64)`
- The helper quantized the real weights before compilation:
  - `conv2d_last` weight scale: `0.00144090`
  - `conv1d_last` weight scale: `0.00117372`
- PAIIR compile result:
  - 2-input / 2-output graph
  - `6` nodes / `4` edges
  - both branches compiled to `SequentialOp`
- backendv2 result:
  - preflight estimate: `3.250 GiB` total dense int32 buffers
  - backendv2 compile was skipped by the current `1.000 GiB` safety guard
- Generated artifacts:
  - `tests/user/debug/dualcnn_last_conv_pair_compile/paiir_summary.log`
  - `tests/user/debug/dualcnn_last_conv_pair_compile/backendv2.log`
  - no `output/frame_type*.{txt,h}` files were generated in this real-shape run because backendv2 did not execute past preflight

# DualCNN Last-Conv Pair Half-Channel Trial

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] This is an execution-only trial using an ad hoc reduced-channel copy of the last conv pair, so no dedicated worktree is needed.

## Plan

- [ ] Build reduced-channel `conv2d_last` / `conv1d_last` copies by slicing the real saved weights.
- [ ] Run `compile_last_conv_pair(...)` with half-channel real-shape inputs.
- [ ] Record whether backendv2 still fails and capture timing / artifact paths.

## Review

- Built a reduced-channel trial by slicing the real saved last-layer weights:
  - `conv2d_last`: `64 -> 128` reduced to `32 -> 64`
  - `conv1d_last`: `64 -> 128` reduced to `32 -> 64`
- Kept the real spatial/sequence sizes:
  - 2D input shape: `(1, 32, 10, 32)`
  - 1D input shape: `(1, 32, 64)`
- Result:
  - PAIIR compile succeeded
  - backendv2 still failed at `LCN_EX`, but the computed value dropped from `9` to `8`
  - failure: `ValueError: 8 is not a valid LCN_EX`
- Timing:
  - `paiir_compile_seconds: 0.312s`
  - `backendv2_compile_seconds: 0.050s`
  - `total_seconds: 0.362s`
- Artifacts:
  - `tests/user/debug/dualcnn_last_conv_pair_half_channels/paiir_summary.log`
  - `tests/user/debug/dualcnn_last_conv_pair_half_channels/backendv2.log`

# DualCNN Last-Conv Pair Quarter-Channel Trial

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] This is another execution-only parameter trial using sliced copies of the real saved weights.

## Plan

- [x] Build a quarter-channel `conv2d_last` / `conv1d_last` pair from the real saved weights.
- [x] Run `compile_last_conv_pair(...)` with quarter-channel real-shape inputs.
- [x] Record backendv2 result, timing, and artifact paths.

## Review

- Built a reduced-channel trial by slicing the real saved last-layer weights:
  - `conv2d_last`: `64 -> 128` reduced to `16 -> 32`
  - `conv1d_last`: `64 -> 128` reduced to `16 -> 32`
- Kept the real spatial/sequence sizes:
  - 2D input shape: `(1, 16, 10, 32)`
  - 1D input shape: `(1, 16, 64)`
- Result:
  - PAIIR compile succeeded
  - backendv2 compile succeeded
  - `2` routing groups compiled
- Timing:
  - `paiir_compile_seconds: 0.285s`
  - `backendv2_compile_seconds: 36.303s`
  - `total_seconds: 36.589s`
- Artifacts generated:
  - `tests/user/debug/dualcnn_last_conv_pair_quarter_channels/paiir_summary.log`
  - `tests/user/debug/dualcnn_last_conv_pair_quarter_channels/backendv2.log`
  - `tests/user/debug/dualcnn_last_conv_pair_quarter_channels/output/frame_type1.txt`
  - `tests/user/debug/dualcnn_last_conv_pair_quarter_channels/output/frame_type2.txt`
  - `tests/user/debug/dualcnn_last_conv_pair_quarter_channels/output/frame_type3.txt`
  - `tests/user/debug/dualcnn_last_conv_pair_quarter_channels/output/frame_type1.h`
  - `tests/user/debug/dualcnn_last_conv_pair_quarter_channels/output/frame_type2.h`
  - `tests/user/debug/dualcnn_last_conv_pair_quarter_channels/output/frame_type3.h`
  - follow-up integration:
    - `tests/user/dualcnn.py` now contains two direct user-facing test entry functions:
      - `test_full_network(...)`
      - `test_last_conv_pair(...)`
    - reduced last-conv-pair testing is now controlled by four fixed top-level channel constants:
      - `DUALCNN_TEST_CONV2D_IN_CHANNELS`
      - `DUALCNN_TEST_CONV2D_OUT_CHANNELS`
      - `DUALCNN_TEST_CONV1D_IN_CHANNELS`
      - `DUALCNN_TEST_CONV1D_OUT_CHANNELS`
    - `main()` now defaults to the last-conv-pair test path using `tests/user/debug/dualcnn_last_conv_pair_test`
    - verification:
      - `ruff check tests/user/dualcnn.py`
      - `./.venv/bin/python tests/user/dualcnn.py`

# DualCNN Last-Conv1d Quantized Test

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] This is a small additive test helper in the same user script.

## Plan

- [x] Add a dedicated `conv1d_last + relu` compile test using quantized actual weights.
- [x] Give it its own output directory.
- [x] Run it once and verify generated artifacts.

## Review

- Added `test_last_conv1d_quantized(...)` to `tests/user/dualcnn.py`.
- The function now fuses the original `BatchNorm1d` into `conv1d_last` first, then quantizes the fused conv weights into int8-domain values, compiles to PAIIR, and runs backendv2.
- Verification result:
  - backendv2 compile succeeded
  - `1` routing group compiled
  - timing:
    - `paiir_compile_seconds: 0.242s`
    - `backendv2_compile_seconds: 22.486s`
    - `total_seconds: 22.729s`
- Artifacts:
  - `tests/user/debug/dualcnn_last_conv1d_test/paiir_summary.log`
  - `tests/user/debug/dualcnn_last_conv1d_test/backendv2.log`
  - `tests/user/debug/dualcnn_last_conv1d_test/output/frame_type1.txt`
  - `tests/user/debug/dualcnn_last_conv1d_test/output/frame_type2.txt`
  - `tests/user/debug/dualcnn_last_conv1d_test/output/frame_type3.txt`
  - `tests/user/debug/dualcnn_last_conv1d_test/output/frame_type1.h`
  - `tests/user/debug/dualcnn_last_conv1d_test/output/frame_type2.h`
  - `tests/user/debug/dualcnn_last_conv1d_test/output/frame_type3.h`

# DualCNN Last-Conv1d Quantized Test

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] This is a small extension to the existing user test script and does not warrant a separate worktree.

## Plan

- [ ] Add a dedicated `conv1d_last + relu` compile test using quantized actual weights.
- [ ] Give it an isolated output directory.
- [ ] Run the test once and record the result.

## Review

- In progress.
  - latest simplification:
    - `test_last_conv_pair(...)` now only checks whether the chosen conv sizes can be compiled
    - it no longer copies real weights from the original model and no longer quantizes the pair

# Weekly Summary and PPT Outline (2026-03-23 to 2026-03-28)

## Workspace Decision

- [x] Stay in the current workspace.
- [x] This task is documentation-only summarization based on local session history and task notes, so no dedicated `git worktree` is needed.

## Ownership

- Owner: Codex
- Branch: `bugfix-paiirgraph-sim-fixes`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files:
  - `tasks/todo.md`
- Blocked Files:
  - product code and tests
- Dependencies:
  - summarize only from locally accessible PAIBox records for this week
  - merge session history with `tasks/todo.md` review notes
- Verification:
  - inspect `.codex/sessions/2026/03/23..28/**`
  - inspect `tasks/todo.md`

## Plan

- [x] Locate this week's accessible PAIBox session history and task records.
- [x] Group the work into major technical themes and extract concrete outcomes.
- [x] Draft a weekly summary suitable for a written report.
- [x] Draft a PPT outline suitable for a weekly presentation.

## Review

- Reviewed local Codex session history for `2026-03-23` through `2026-03-28` under `/home/kafcoppelia/.codex/sessions/2026/03/**` with `cwd=/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`.
- Cross-checked the session findings against the accumulated review notes in this file.
- Consolidated this week's work into these main themes:
  - PAIIR graph/tracing and package-layer cleanup
  - reshape/view/flatten lowering and shape-analysis refactor
  - signal-domain / data-format / MaxPool behavior clarification and fixes
  - DualCNN full-network / branch / last-conv compilation experiments
  - backendv2 feasibility boundaries and user-model validation cleanup
- Prepared a Chinese weekly work summary plus a PPT outline centered on accomplishments, evidence, risks, and next-step plans.

# Lockfile Refresh

## Workspace Decision

- [x] Continue in the current workspace on `bugfix-paiirgraph-sim-fixes`.
- [x] Keep this step focused on refreshing `poetry.lock` and `uv.lock` to match the current `pyproject.toml` dependency declarations.

## Plan

- [x] Regenerate `poetry.lock` from the current `pyproject.toml`.
- [x] Regenerate `uv.lock` from the current `pyproject.toml`.
- [x] Confirm both lockfiles reflect the declared dependency updates.

## Review

- Refreshed dependency declarations in [pyproject.toml](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/pyproject.toml):
  - added runtime dependency `rich>=14.0.0`
  - bumped dev dependency `paicorelib` from `>=2.0.0a2` to `>=2.0.0a3`
- Regenerated both lockfiles from the updated project metadata:
  - `poetry.lock` via `poetry lock`
  - `uv.lock` via `uv lock`
- Verification:
  - `timeout 900s env POETRY_CACHE_DIR=/tmp/pypoetry-cache POETRY_VIRTUALENVS_PATH=/tmp/pypoetry-virtualenvs poetry lock`
    - result: lock resolved and wrote `poetry.lock`
  - `env UV_CACHE_DIR=/tmp/uv-cache uv lock`
    - result: resolved 71 packages and wrote `uv.lock`

# PAIIR SumPool Expression Tightening

## Workspace Decision

- [x] Continue in the current workspace on `feat-paiir-lowering-cleanup`.
- [x] Do not create a dedicated `git worktree` because this fix is tightly coupled to the in-flight local `paiir` / `avgpool` changes already present in this workspace, and it needs to integrate with them directly.

## Ownership

- Owner: Codex
- Branch: `feat-paiir-lowering-cleanup`
- Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files:
  - `paibox/paiir/nn/**`
  - `paibox/paiir/pipeline/avgpool/**`
  - `paibox/backendv2/**`
  - `tests/paiir/pipeline/**`
  - `tests/backendv2/**`
  - `tasks/todo.md`
- Blocked Files:
  - unrelated existing dirty files outside the SumPool / AvgPool / backendv2 scope
- Dependencies:
  - preserve split-core AvgPool sum-domain semantics in PAIIR
  - keep backend-visible pool metadata sufficient and unambiguous
  - keep graph simulation correct with quantized weights and integer inputs
- Verification:
  - focused `pytest` for `tests/paiir/pipeline/**` and `tests/backendv2/**`
  - focused `ruff check` on touched files

## Plan

- [ ] Add regression coverage proving split-core SumPool is backend-expandable via raw-weight extraction.
- [ ] Refactor `SumPool1d/2d` so PAIIR keeps explicit sum-pool semantics while the runtime representation remains backend-friendly and simulation-safe.
- [ ] Re-run focused compile / simulation / backend tests and record the results.

## Review

- In progress.
  - current design direction:
    - user explicitly chose to keep the current dedicated `SumPool1d/2d` form
    - do not change any `backendv2/**` product code for this line of work
    - follow-up work is narrowed to unsupported AvgPool parameter guards in `paiir/**`

# AvgPool Unsupported Parameter Guards

## Workspace Decision

- [x] Continue in the current workspace on `feat-paiir-lowering-cleanup`.
- [x] Keep this task entirely outside `backendv2/**`; only touch `paiir/**`, focused tests, and task notes.

## Plan

- [x] Revert the abandoned `divisor_override`-oriented exploratory test edits so the branch matches the chosen `SumPool` direction again.
- [x] Add regression coverage that rejects `count_include_pad=False` when `padding>0` while still allowing `padding=0`.
- [x] Implement the minimal lowering-time guard in `paiir/**` and record a concrete evaluation of the work needed to support `divisor_override` later.

## Review

- Reverted the abandoned exploratory direction so the active diff stays outside `backendv2/**` and `tests/backendv2/**`.
- Added lowering-layer coverage in `tests/paiir/lowering/test_converter.py` for:
  - strict rejection of `AvgPool2d(count_include_pad=False, padding=1)`
  - non-strict warning/bypass for the same case
  - explicit allowance for `AvgPool2d(count_include_pad=False, padding=0)`
- Added compile-wrapper coverage in `tests/paiir/pipeline/test_compile.py` for:
  - `compile_to_paiir(..., strict=True)` rejecting `count_include_pad=False` with non-zero padding
  - successful compilation when `count_include_pad=False` but `padding=0`
- Implemented the guard in `paibox/paiir/lowering/converter.py`:
  - `AvgPool1d/2d` now becomes an unsupported op when `count_include_pad=False` and any padding component is non-zero
  - the guard reuses the existing strict/non-strict unsupported-op flow rather than introducing a separate validation mechanism
- Verification:
  - `ruff check paibox/paiir/lowering/converter.py tests/paiir/lowering/test_converter.py tests/paiir/pipeline/test_compile.py`
    - result: `All checks passed!`
  - `./.venv/bin/pytest tests/paiir/lowering/test_converter.py -q -k 'count_include_pad or strict_mode'`
    - result: `5 passed, 1 deselected`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k 'count_include_pad or strict_raises_for_count_include_pad_false_with_padding or padding_free_count_include_pad_false_still_compiles'`
    - result: `2 passed, 82 deselected`
- `divisor_override` support evaluation:
  - it is still not handled today and remains silently outside the current semantic model
  - supporting it cleanly should be treated as an AvgPool deployment follow-up in `paiir/**`, not as a `SumPool` runtime tweak
  - the main change is to split the current single `window_size` concept into:
    - window extent used for raw sum range / exact-sum feasibility
    - effective divisor used for AvgPool-domain compensation and threshold scaling
  - the minimum affected areas are:
    - `paibox/paiir/pipeline/avgpool/compensation.py`
    - `paibox/paiir/pipeline/avgpool/fusion.py`
    - `paibox/paiir/pipeline/avgpool/deploy_scheme.py`
    - `paibox/paiir/pipeline/avgpool/calibration.py`
    - `paibox/paiir/pipeline/avgpool/pass_ops.py`
    - focused compile / simulation tests under `tests/paiir/pipeline/**`
  - by contrast, `count_include_pad=False` with non-zero padding remains a larger task because the effective divisor varies at borders rather than staying globally constant

# AvgPool divisor_override Support

## Workspace Decision

- [x] Continue in the current workspace on `feat-paiir-lowering-cleanup`.
- [x] Keep this work strictly inside `paibox/paiir/**` and `tests/paiir/**`; do not modify `backendv2/**` or `tests/backendv2/**`.

## Plan

- [x] Thread `divisor_override` through AvgPool deployment logic by separating real sum-window extent from AvgPool effective divisor.
- [x] Add focused regression coverage for split-core IF deployment and AvgPool deployment scoring with `divisor_override`.
- [x] Re-run focused validation and confirm the earlier `count_include_pad` guard still holds.

## Review

- Added `_get_avgpool_divisor(...)` in `paibox/paiir/pipeline/avgpool/utils.py`:
  - `window_size` remains the real pooling-window element count
  - `avg_divisor` now represents the effective AvgPool divisor (`divisor_override` when present, otherwise `window_size`)
- Updated AvgPool deployment logic in:
  - `paibox/paiir/pipeline/avgpool/fusion.py`
  - `paibox/paiir/pipeline/avgpool/deploy_scheme.py`
  - `paibox/paiir/pipeline/avgpool/calibration.py`
  - `paibox/paiir/pipeline/avgpool/pass_ops.py`
- The new behavior is:
  - split-core exact-sum feasibility and sampled sum-domain probe ranges still use the real pooling-window size
  - shared-core compensation, split-core LUT scaling, split-core Core-2 voltage scaling, candidate scoring, and calibration reference reconstruction now use the effective AvgPool divisor
- Also fixed an existing local issue in `paibox/paiir/pipeline/avgpool/compensation.py`:
  - `LutData` is frozen, so compensation helpers now return new `LutData` instances instead of assigning back into frozen fields
- Added focused regressions in:
  - `tests/paiir/pipeline/avgpool/test_deploy_scheme.py`
    - split-core IF with `AvgPool2d(..., divisor_override=1)` now proves Core-1 LUT thresholds scale by divisor `1` rather than kernel area `4`
  - `tests/paiir/pipeline/avgpool/test_compensation.py`
    - candidate scoring now proves `avg_divisor` changes the scored deployment outcome even when the real sum-window size stays fixed
  - `tests/paiir/pipeline/test_graph_simulation.py`
    - `AvgPool2d(..., divisor_override=1)` on the IF split-core path matches the PyTorch / SpikingJelly reference in simulation
- Verification:
  - `ruff check paibox/paiir/pipeline/avgpool/utils.py paibox/paiir/pipeline/avgpool/fusion.py paibox/paiir/pipeline/avgpool/deploy_scheme.py paibox/paiir/pipeline/avgpool/calibration.py paibox/paiir/pipeline/avgpool/pass_ops.py paibox/paiir/pipeline/avgpool/compensation.py tests/paiir/pipeline/avgpool/test_deploy_scheme.py tests/paiir/pipeline/avgpool/test_compensation.py tests/paiir/pipeline/test_graph_simulation.py`
    - result: `All checks passed!`
  - `./.venv/bin/pytest tests/paiir/pipeline/avgpool/test_deploy_scheme.py tests/paiir/pipeline/avgpool/test_compensation.py -q`
    - result: `48 passed`
  - `env COVERAGE_FILE=/tmp/paiir_divisor_override.coverage ./.venv/bin/pytest tests/paiir/pipeline/test_graph_simulation.py -q -k 'avgpool_if_snn and not avgpool_lif_split_core_snn or avgpool_if_snn_with_divisor_override'`
    - result: `3 passed`
  - `./.venv/bin/pytest tests/paiir/lowering/test_converter.py -q -k 'count_include_pad or strict_mode'`
    - result: `5 passed, 1 deselected`
  - `./.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k 'count_include_pad or strict_raises_for_count_include_pad_false_with_padding or padding_free_count_include_pad_false_still_compiles'`
    - result: `2 passed, 82 deselected`

# AvgPool Helper Cleanup

## Workspace Decision

- [x] Continue in the current workspace on `feat-paiir-lowering-cleanup`.
- [x] Keep this cleanup limited to `paibox/paiir/pipeline/avgpool/**` and `tests/paiir/**`; do not touch `backendv2/**` or `tests/backendv2/**`.

## Plan

- [x] Narrow `_is_avgpool(...)` so call sites can treat the checked module as `nn.AvgPool1d | nn.AvgPool2d`.
- [x] Replace back-to-back `_get_pool_window_size(...)` + `_get_avgpool_divisor(...)` fetches with a single helper where both semantics are needed.
- [x] Re-run focused checks and keep the existing divisor-override behavior unchanged.

## Review

- Added `TypeGuard` narrowing in `paibox/paiir/pipeline/avgpool/utils.py`:
  - `_is_avgpool(...)` now narrows to `nn.AvgPool1d | nn.AvgPool2d`
- Added `AvgPoolSemantics(window_size, avg_divisor)` plus `_get_avgpool_semantics(...)`:
  - clarifies that `window_size` is the real sum-domain window extent
  - clarifies that `avg_divisor` is the effective AvgPool divisor after `divisor_override`
- Updated the call sites that genuinely need both values:
  - `paibox/paiir/pipeline/avgpool/fusion.py`
  - `paibox/paiir/pipeline/avgpool/pass_ops.py`
- Left the sites that only need the effective divisor on `_get_avgpool_divisor(...)`, so the split between “sum extent” and “division semantics” stays explicit rather than over-normalized.
- Added a focused regression in `tests/paiir/pipeline/avgpool/test_compensation.py` proving `AvgPool2d(2, divisor_override=1)` yields:
  - `window_size == 4`
  - `avg_divisor == 1`
- Verification:
  - `ruff check paibox/paiir/pipeline/avgpool/utils.py paibox/paiir/pipeline/avgpool/fusion.py paibox/paiir/pipeline/avgpool/pass_ops.py tests/paiir/pipeline/avgpool/test_compensation.py`
    - result: `All checks passed!`
- `./.venv/bin/pytest tests/paiir/pipeline/avgpool/test_compensation.py -q -k 'avgpool_semantics or avg_divisor_changes_candidate_scoring'`
  - result: `2 passed, 39 deselected`
  - `./.venv/bin/pytest tests/paiir/pipeline/avgpool/test_deploy_scheme.py -q -k 'divisor_override_controls_split_core_if_lut_scaling'`
    - result: `1 passed, 7 deselected`

# PR Summary: feat-paiir-lowering-cleanup -> dev

## Workspace Decision

- [x] Stay in the current workspace.
- [x] This task is documentation-only (PR title and summary) based on the branch diff.

## Plan

- [x] Fetch `origin/dev` and `origin/feat-paiir-lowering-cleanup` to ensure the diff is current.
- [x] Inspect the commit list and file-level diff to identify the dominant change themes.
- [x] Draft a concise PR title and summary consistent with existing PR style.
- [x] Record the final summary in this task review section.

## Review

- Verified commit range: `origin/dev..origin/feat-paiir-lowering-cleanup`.
- Captured diff stats and commit themes to drive the PR summary.
- Proposed PR title:
  - `Feat(paiir): clean up lowering flow and AvgPool handling`
- Proposed short PR summary themes:
  - improve reshape-like / conv-family lowering and tighten routing validation
  - support `AvgPool divisor_override` while explicitly guarding unsupported `count_include_pad=False` with padding
  - narrow package exports and expand compile / simulation / pass coverage

# Backendv2 Committed Change Rollback

## Workspace Decision

- [x] Stay in the current workspace on `feat-paiir-lowering-cleanup`.
- [x] Do not create a dedicated `git worktree` because this request is targeted branch-local cleanup against already committed history, and it needs to preserve the current in-place working state with minimal Git disturbance.

## Ownership

- [x] Owner: Codex
- [x] Branch: `feat-paiir-lowering-cleanup`
- [x] Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- [x] Allowed Files: `paibox/backendv2/core_config.py`, `paibox/backendv2/get_weight.py`, `paibox/backendv2/op_node.py`, `tasks/todo.md`
- [x] Blocked Files: all other product files and tests unless focused verification requires read-only inspection
- [x] Dependencies: compare current branch against `origin/dev` to isolate committed `backendv2/**` changes before editing
- [x] Verification: confirm `backendv2/**` diff vs `origin/dev` keeps only import-path changes, then run focused backendv2 tests

## Plan

- [x] Inspect the committed `backendv2/**` diff on the current branch and classify import-path updates versus implementation/body changes.
- [x] Confirm that only `acd7d52` contains `backendv2` body changes; later `backendv2` commits only adjust import paths.
- [x] Restore `paibox/backendv2/op_node.py` to committed `HEAD`, stash current local work safely, and non-interactively rebase from `origin/dev` with `acd7d52` marked for edit.
- [x] Amend `acd7d52` so `paibox/backendv2/op_node.py` keeps only the narrowed import paths and drops the backend reorder/body changes, then continue the rebase.
- [x] Reapply the stash, verify rewritten history plus final `backendv2/**` diff, and record the new commit ids.

## Review

- Classified the current-branch committed `backendv2/**` changes against `origin/dev` as:
  - import-only changes in `paibox/backendv2/core_config.py` and `paibox/backendv2/get_weight.py`
  - mixed import + implementation changes in `paibox/backendv2/op_node.py`
- The earlier working-tree-only rollback was intentionally superseded after the user clarified that the request is to rewrite existing commits rather than leave an uncommitted diff.
- Commit classification for the history rewrite:
  - `acd7d52` contains the only `backendv2` body change and must be amended
  - `4bd9670` and `f75dc76` only change import paths and should remain semantically intact
- Rebase / rewrite execution:
  - stashed local work with `git stash push -u -m 'codex-backendv2-history-rewrite-safety'`
  - first rebase attempt stopped at the wrong earliest commit and was immediately aborted
  - restarted rebase from `origin/dev` with only `acd7d52` marked `edit`
  - amended that commit in place so `paibox/backendv2/op_node.py` now keeps only the narrowed `paibox.paiir.ir.*` import changes
  - continued the rebase successfully through the remaining commits
- Rewritten branch-local commit ids are now:
  - `41af274` unchanged
  - `3e7ae32` rewritten from `acd7d52`
  - `eee8253` rewritten from `4bd9670`
  - `5825a9d` rewritten from `f75dc76`
  - `217cb63` rewritten from `539b8b6`
  - `913eb3f` rewritten from `1ecfd9e`
  - `b9a57dd` rewritten from `b708988`
  - `34b8859` rewritten from `3e74916`
- Verification:
  - `git show --word-diff=color 3e7ae32^..3e7ae32 -- paibox/backendv2/op_node.py`
    - result: the rewritten commit now changes only import paths in `op_node.py`
  - `git diff origin/dev -- paibox/backendv2`
    - result: `backendv2/**` net diff versus `origin/dev` retains only import-path changes in `core_config.py`, `get_weight.py`, and `op_node.py`
  - `git log --oneline --reverse origin/dev..HEAD -- paibox/backendv2`
    - result: `backendv2` is still touched by exactly three branch commits, but only for import-path updates after the rewrite
- Stash restore notes:
  - `git stash pop stash@{0}` restored the tracked modifications and the untracked worktree contents
  - `.codex` already existed, so Git kept `stash@{0}` instead of dropping it automatically as a safety fallback
  - branch status now shows local history divergence from `origin/feat-paiir-lowering-cleanup` (`ahead 7, behind 7`), which is expected after rewriting existing commits and would require a force-push later if the remote should match

# PAIIRGraph Mutator API Cleanup

## Workspace Decision

- [x] Use a dedicated worktree because this change touches core graph mutation APIs and should not be developed directly inside the shared dirty workspace.
- [x] Verification worktree: `/tmp/PAIBox-kafcoppelia-graph-api-cleanup` on `codex/kafcoppelia/graph-api-cleanup`
- [x] Final integration: apply the verified minimal patch back to the current workspace files the user is actively editing.

## Ownership

- [x] Owner: Codex
- [x] Allowed Files: `paibox/paiir/ir/graph.py`, `tests/paiir/ir/test_graph.py`, `tasks/todo.md`
- [x] Blocked Files: unrelated in-flight PAIIR/lowering/backend files remain untouched
- [x] Dependencies: preserve existing layout-pass call patterns while tightening graph API semantics
- [x] Verification: focused `pytest` on graph API tests plus the two layout rewrite modules inside the isolated worktree

## Plan

- [x] Convert `replace_all_uses_with` and `remove_node_and_reconnect` into pure mutator-style APIs with `None` returns.
- [x] Tighten `remove_node_and_reconnect` so an explicit `source_name` must be an actual predecessor.
- [x] Update focused tests to assert graph behavior rather than integer return counts, then rerun the affected test targets.

## Review

- Updated [graph.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/graph.py):
  - `replace_all_uses_with(...)` now returns `None` and documents that `delete_old` is an explicit opt-in.
  - `remove_node_and_reconnect(...)` now returns `None` and rejects explicit `source_name` values that are not real predecessors of the removed node.
- Updated [test_graph.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/ir/test_graph.py):
  - graph mutation tests now assert resulting structure instead of integer return counts
  - added a regression test for non-predecessor source rejection
- Focused verification ran in `/tmp/PAIBox-kafcoppelia-graph-api-cleanup`:
  - `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/.venv/bin/pytest tests/paiir/ir/test_graph.py tests/paiir/pipeline/test_layout_chain_canonicalization.py tests/paiir/pipeline/test_layout_cross_node_elision.py -q`
  - result: `20 passed`
- Current-workspace confirmation:
  - `./.venv/bin/pytest tests/paiir/ir/test_graph.py tests/paiir/pipeline/test_layout_chain_canonicalization.py tests/paiir/pipeline/test_layout_cross_node_elision.py -q`
  - result: `20 passed`

# PAIIRGraph Validation API Review

## Workspace Decision

- [x] Stay in the current workspace on `feat/layout-two-pass-canonicalization`.
- [x] Keep this task review-only because the request is about API/design direction rather than an immediate code change.

## Ownership

- [x] Owner: Codex
- [x] Allowed Files: `tasks/todo.md`
- [x] Blocked Files: product code remains read-only unless the user asks to implement the validator
- [x] Dependencies: compare `torch.fx.Graph.lint` against current `PAIIRGraph` invariants and existing verification hooks
- [x] Verification: ensure the recommendation is grounded in current source definitions and call patterns

## Plan

- [x] Inspect `torch.fx.Graph.lint` and current `PAIIRGraph` validation helpers/invariants.
- [x] Identify which checks are graph-structural versus simulation-/deployment-specific in PAIIR.
- [x] Recommend an appropriate `lint` / `validate` API surface for `PAIIRGraph`.

## Review

- `torch.fx.Graph.lint()` in [torch/fx/graph.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/.venv/lib/python3.13/site-packages/torch/fx/graph.py#L1777) is intentionally narrow:
  - node ownership belongs to this graph
  - topological order is valid
  - names are unique
  - referenced module targets exist when an owning module is present
- `PAIIRGraph` already has two heavier validation layers plus simulation readiness checks:
  - mid-pipeline cleanup/validation in [passes.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/passes.py#L298)
  - final compiled-graph validation in [passes.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/passes.py#L374)
  - runtime/simulation checks in [graph.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/graph.py#L349)
- Recommendation:
  - Yes, add a `PAIIRGraph.lint()`-style helper, but keep it graph-structural only.
  - Do not move all of `validate_graph` / `validate_compiled_graph` into `PAIIRGraph`; those functions encode pipeline-stage semantics and some cleanup behavior.
  - Best split:
    - `PAIIRGraph.lint()` for structural invariants
    - `validate_graph()` for mid-pipeline cleanup + early semantic checks
    - `validate_compiled_graph()` for final compile invariants
    - `verify_before_sim()` for simulation readiness
- PAIIR-specific checks that belong in a new graph-level lint:
  - every `graph.nodes` key matches `node.name`
  - every edge endpoint exists in `graph.nodes`
  - the graph is acyclic / topologically sortable
  - `InputNode` has no predecessors, `OutputNode` has no successors
  - no exact duplicate edges
  - incoming `dst_port` values for the same destination are unique, because simulation assumes predecessor order matches port order in [graph.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/graph.py#L506)
  - optional fixed-arity checks for obviously single-input nodes such as `OutputNode` and `ReshapeOp`, if you want `lint()` to catch malformed routing graphs earlier
- Checks that should stay outside graph-level lint:
  - shape/dims/domain metadata completeness
  - deployability subset restrictions
  - LUT/data-format/tick-parameter consistency
  - cleanup/removal of disconnected nodes

# PAIIRGraph Lint Refactor

## Workspace Decision

- [x] Use a dedicated worktree because this change touches core IR validation plus pipeline validators across multiple files.
- [x] Keep the shared workspace as the user-facing integration workspace and do implementation/verification in an isolated branch first.

## Ownership

- [x] Owner: Codex
- [x] Branch: `codex/kafcoppelia/graph-lint`
- [x] Worktree: `/tmp/PAIBox-kafcoppelia-graph-lint`
- [x] Allowed Files: `paibox/paiir/ir/graph.py`, `paibox/paiir/pipeline/passes.py`, focused graph/pipeline tests, `tasks/todo.md`
- [x] Blocked Files: unrelated lowering/backend/onboard/user files remain untouched
- [x] Dependencies: carry the current in-flight PAIIR files into the isolated worktree before editing so the refactor is based on the active local code, not stale `HEAD`
- [x] Verification: focused pytest for graph tests and validation-pass tests in the isolated worktree, then confirm the same targets in the current workspace after integrating the patch

## Plan

- [x] Create the isolated worktree and sync the relevant in-flight PAIIR files into it.
- [x] Implement `PAIIRGraph.lint()` plus shared validation helpers, then refactor `validate_graph`, `validate_compiled_graph`, and `verify_before_sim` to use the new layering.
- [x] Add/update focused tests for structural lint, disconnected-node detection, and validator responsibilities, then run targeted pytest verification.

## Review

- Added structural validation primitives to [graph.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/ir/graph.py):
  - `PAIIRGraph.lint(allow_disconnected=False)` for graph/container consistency
  - `nodes_on_input_output_paths()` and `disconnected_nodes()` for shared reachability logic
  - `verify_before_sim()` now reuses `lint()` and preserves its public `RuntimeError` surface by aggregating lint failures into the existing simulation-readiness report
- Refactored validator layering in [passes.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/pipeline/passes.py):
  - `validate_graph()` now does:
    - hard structural lint with `allow_disconnected=True`
    - disconnected-node cleanup via `graph.disconnected_nodes()`
    - post-cleanup empty-graph checks with the existing “after cleanup” wording
    - final structural lint
    - remaining mid-pipeline semantic checks (shape presence, potential-add contract, LUT mode consistency)
  - `validate_compiled_graph()` now starts from `graph.lint()` and focuses on final compile metadata/contracts instead of duplicating structure checks
  - removed the duplicated reachability helper from `passes.py`
- Added focused tests:
  - [test_graph.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/ir/test_graph.py)
    - `lint()` rejects missing-edge endpoints
    - `lint()` rejects disconnected nodes
    - `verify_before_sim()` reports structural lint failures
  - [test_passes.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_passes.py)
    - updated the compiled-graph disconnected-path test so final validation owns that error directly, instead of relying on `validate_graph()` to leave the dead branch in place
- Focused verification in the isolated worktree:
  - `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/.venv/bin/pytest tests/paiir/ir/test_graph.py tests/paiir/pipeline/test_passes.py -q`
  - result: `87 passed`
  - `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/.venv/bin/pytest tests/paiir/pipeline/test_layout_chain_canonicalization.py tests/paiir/pipeline/test_layout_cross_node_elision.py -q`
  - result: `8 passed`
- Current-workspace confirmation:
  - `./.venv/bin/pytest tests/paiir/ir/test_graph.py tests/paiir/pipeline/test_passes.py tests/paiir/pipeline/test_layout_chain_canonicalization.py tests/paiir/pipeline/test_layout_cross_node_elision.py -q`
  - result: `95 passed`
- Baseline note:
  - `tests/paiir/pipeline/test_compile.py` currently has four pre-existing failures in both the isolated worktree and the current workspace:
    - `test_function_unsqueeze_before_linear_compiles`
    - `test_tuple_repeat_all_ones_before_linear_compiles`
    - `test_method_squeeze_before_linear_compiles`
    - `test_function_squeeze_before_linear_compiles`
  - all four fail because the current branch now produces `1` `ReshapeOp` instead of the older expected `2`, which is unrelated to this lint/validator refactor.

# PAIIR Test Import Hoisting Cleanup

## Workspace Decision

- [x] Used a dedicated worktree because this cleanup spans multiple `tests/paiir/**` files in a dirty shared workspace and should not be edited in place first.
- [x] Branch: `feat-paiir-test-import-hoist`
- [x] Worktree: `/tmp/PAIBox-kafcoppelia-paiir-test-import-hoist`
- [x] Scope note: initial AST scan found no function-local imports under `paibox/paiir/**`; the cleanup only touched `tests/paiir/**`.

## Ownership

- [x] Owner: Codex
- [x] Allowed Files: `tests/paiir/ir/test_core_neuron.py`, `tests/paiir/lowering/test_converter.py`, `tests/paiir/pipeline/test_graph_simulation.py`, `tests/paiir/pipeline/test_passes.py`, `tasks/todo.md`
- [x] Blocked Files: all other implementation and test files unless the scan discovers another real function-local import in scope
- [x] Dependencies: preserve current branch behavior; only hoist imports that are not intentionally deferred for circular-import, optional-dependency, or side-effect reasons
- [x] Verification: use AST re-scan plus targeted `pytest`; `ruff` was attempted but is not installed in this environment

## Plan

- [x] Confirm every function-local import under `paibox/paiir/**` and `tests/paiir/**`, and classify whether it should stay local or move to module scope.
- [x] Create the isolated worktree from the current branch tip, copy in any active local versions of the touched test files if needed, and hoist the eligible imports.
- [x] Run focused lint/tests on the touched files, then record the exact results in the review section.

## Review

- Function-local import audit result:
  - `paibox/paiir/**`: no function-local `import` / `from ... import ...` occurrences were found
  - `tests/paiir/**`: all discovered function-local imports were in tests and were eligible for hoisting
- Hoisted the test-only imports to module scope in:
  - [test_core_neuron.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/ir/test_core_neuron.py)
  - [test_converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/lowering/test_converter.py)
  - [test_graph_simulation.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_graph_simulation.py)
  - [test_passes.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/pipeline/test_passes.py)
- Imports kept local: none in the final `paibox/paiir/**` + `tests/paiir/**` scan
- Verification:
  - AST re-scan over `paibox/paiir` and `tests/paiir`
  - result: `NO_FUNCTION_LOCAL_IMPORTS_FOUND`
  - `./.venv/bin/pytest tests/paiir/ir/test_core_neuron.py tests/paiir/lowering/test_converter.py tests/paiir/pipeline/test_graph_simulation.py tests/paiir/pipeline/test_passes.py -q`
  - result: `164 passed, 2 skipped`
  - `./.venv/bin/python -m ruff check tests/paiir/ir/test_core_neuron.py tests/paiir/lowering/test_converter.py tests/paiir/pipeline/test_graph_simulation.py tests/paiir/pipeline/test_passes.py`
  - result: unavailable in this environment (`No module named ruff`)

# ModelSCNN Quantized Bundle PAIIR Chip-Native Approximation

## Workspace Decision

- [x] Continue in the current workspace on `feat/layout-two-pass-canonicalization`.
- [x] Do not create a dedicated `git worktree` for this task because the target `tests/onboard/modelscnn/**` models and `.pt` bundles are untracked local workspace assets that would not exist in a fresh worktree without extra copying.

## Ownership

- [x] Owner: Codex
- [x] Branch: `feat/layout-two-pass-canonicalization`
- [x] Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- [x] Allowed Files: `tasks/todo.md`, `tests/onboard/modelscnn/**`
- [x] Blocked Files: unrelated `paibox/**`, `tests/paiir/**`, and other in-flight user edits outside `tests/onboard/modelscnn/**`
- [x] Dependencies: reuse the existing `paiir` -> `backendv2` compile path and the built-in legacy SpikingJelly compatibility already added to converter
- [x] Verification: run the new targeted pytest file and confirm both MNIST and CIFAR10 bundles compile through PAIIR and backendv2 under a chip-native integer approximation path

## Plan

- [x] Inspect `modelscnn.py`, `quantized_modelscnn.py`, and the two `.pt` bundles to determine whether deployment should use the quantized runtime modules directly or rebuild a chip-native integer approximation.
- [x] Add a focused onboard test that reconstructs a single-step LeNet5 from each quantized bundle using deployable integer-valued parameters, compiles it with `compile_to_paiir`, and then runs `Mapper.compile`.
- [x] Run focused verification for the new test and record the outcomes here.

## Interface Notes

- Freeze the deployment path to standard PyTorch ops with chip-native integer-valued parameters:
  - load bundle metadata from the `.pt`
  - rebuild `SpikingLeNet5` from `modelscnn.py`
  - copy raw `int8` weight values from the quantized bundle into the standard `Conv2d` / `Linear` modules
  - integerize bias by rounding before deployment
  - neutralize BatchNorm to identity because the quantized export already folded BN into Conv
  - treat the first layer input as spike input (`UNSIGNED, WIDTH_1BIT`) rather than image-style quantized input
  - wrap the network in a single-step forward so lowering sees the deployable graph and avoids the training-time `out / self.T` reduction
- Integer-domain facts confirmed from the `.pt` bundles:
  - all saved Conv/Linear weights are `torch.qint8`
  - every saved weight tensor exposes `int_repr().dtype == torch.int8`
  - saved weight zero-points are all `0` per output channel, so weights themselves can stay on a signed-int8 path
  - per-channel weight scales are available in the bundle but are intentionally ignored in this simpler chip-native approximation
- Do not compile `QuantizedSpikingLeNet5` directly:
  - `QuantizedLIFNode` is custom runtime logic outside the current built-in lowering support
  - the standard model structure is the path that preserves parseable operators for `paiir`
  - the current approximation does not attempt to replay PyTorch quantized requantize semantics on chip

## Review

- Added [test_modelscnn_paiir_deploy.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/onboard/modelscnn/test_modelscnn_paiir_deploy.py), which:
  - rebuilds the standard `SpikingLeNet5` topology from [modelscnn.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/onboard/modelscnn/modelscnn.py)
  - copies raw `int8` Conv/Linear weights from each quantized bundle into the standard modules
  - integerizes Conv/Linear bias by rounding
  - turns BatchNorm into identity because the quantized export already folded BN into Conv
  - wraps the network into a single-step forward to avoid tracing the training-time `out / self.T` reduction that currently lowers to unsupported `truediv`
  - compiles with spike input format (`UNSIGNED, WIDTH_1BIT`)
  - runs `compile_to_paiir(...)` and then `Mapper.compile(...)` for both CIFAR10 and MNIST bundles
- Scope correction:
  - this is a chip-native integer approximation path
  - it is deployable in the sense that it uses only chip-friendly integer-valued parameters
  - it is not an exact replay of the saved PyTorch quantized runtime semantics because per-channel scale driven requantization is ignored
- Added integer-domain metadata coverage in the same test file:
  - confirms every Conv/Linear bundle weight is `torch.qint8` with `torch.int8` `int_repr()`
  - writes per-bundle quantization assessment logs that record:
    - per-channel weight scales / zero-points
    - integerized bias ranges used by the approximation
    - that input qparams are ignored because deployment assumes spike input at the first layer
  - records the main exactness gap for true PyTorch-quantized equivalence:
    - per-channel scale driven requantization is not replayed in the chip-native approximation
- Runtime artifacts written by the test:
  - [cifar10_paiir_summary.log](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/onboard/modelscnn/debug/paiir_modelscnn_deploy/cifar10_paiir_summary.log)
  - [cifar10_backendv2.log](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/onboard/modelscnn/debug/paiir_modelscnn_deploy/cifar10_backendv2.log)
  - [cifar10_quantization_assessment.log](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/onboard/modelscnn/debug/paiir_modelscnn_deploy/cifar10_quantization_assessment.log)
  - [mnist_paiir_summary.log](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/onboard/modelscnn/debug/paiir_modelscnn_deploy/mnist_paiir_summary.log)
  - [mnist_backendv2.log](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/onboard/modelscnn/debug/paiir_modelscnn_deploy/mnist_backendv2.log)
  - [mnist_quantization_assessment.log](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/onboard/modelscnn/debug/paiir_modelscnn_deploy/mnist_quantization_assessment.log)
- Verification:
  - direct repo config path:
    - `./.venv/bin/pytest tests/onboard/modelscnn/test_modelscnn_paiir_deploy.py -q`
    - result: blocked by unrelated local syntax residue in [pyproject.toml](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/pyproject.toml#L85) (`==`)
  - focused fallback used for real verification:
    - `printf '[pytest]\n' > /tmp/pytest-modelscnn.ini`
    - `./.venv/bin/pytest -c /tmp/pytest-modelscnn.ini --rootdir=/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox tests/onboard/modelscnn/test_modelscnn_paiir_deploy.py -q`
    - result: `4 passed` in `68.30s`

# PAIRV Removed Items Documentation Sync

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Keep this task scoped to `../PAIRV/README.md` plus `tasks/**`.
- [x] Do not create a dedicated `git worktree` because this is a small documentation-only sync after user deletions.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `../PAIRV/README.md`, `tasks/**`
- Blocked Files: build logic and product code
- Dependencies: current filesystem state after removal of `SoC/evalsoc/runmode.mk` and `application/freertos/smpdemo`
- Verification: ensure the main docs/help surface no longer references the removed items

## Plan

- [x] Record the user correction in `tasks/lessons.md`.
- [x] Patch `README.md` to remove references to `runmode.mk`, run-mode support, and `application/freertos/smpdemo`.
- [x] Verify no documentation/help references remain for the removed items.

## Review

- Updated [README.md](/home/kafcoppelia/WORK/PAIRV/README.md) to match the current tree:
  - removed `application/freertos/smpdemo` from the active application list
  - removed `SoC/evalsoc/runmode.mk` from the important file list
  - removed the old “run-mode control” language from the project/SoC description
  - replaced the stray absolute-path `Makefile` reference with plain inline code
- Verification:
  - `find ../PAIRV/SoC/evalsoc -maxdepth 2 -type f | sort`
    - result: confirms `runmode.mk` is gone
  - `find ../PAIRV/application/freertos -maxdepth 2 -type d | sort`
    - result: confirms only `demo` remains under `application/freertos`
  - `rg -n "smpdemo|runmode\.mk|Run Mode|run modes|RUNMODE|application/freertos/smpdemo|SoC/evalsoc/runmode\.mk|/home/" ../PAIRV/README.md ../PAIRV/Build/Makefile.rules ../PAIRV/Makefile`
    - result: no remaining documentation/help references to the removed items

# PAIRV setup.sh CI Hook Review And Commit Split Plan

## Workspace Decision

- [x] Continue in the current workspace on `dev`.
- [x] Keep this task scoped to `../PAIRV/setup.sh`, the reference `/home/kafcoppelia/WORK/BOARDS/nuclei-sdk/.ci/**`, the focused PAIRV build/documentation files, and `tasks/**`.
- [x] Do not create a dedicated `git worktree` because this phase is analysis plus commit grouping, not new implementation.

## Ownership

- Owner: Codex
- Branch: `dev`
- Worktree: current workspace `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- Allowed Files: `../PAIRV/setup.sh`, `../PAIRV/.editorconfig`, `../PAIRV/.gitignore`, `../PAIRV/Makefile`, `../PAIRV/README.md`, `../PAIRV/application/**`, `../PAIRV/Build/**`, `../PAIRV/NMSIS/**`, `../PAIRV/OS/**`, `../PAIRV/SoC/**`, `tasks/**`
- Blocked Files: unrelated paths outside the requested review scope
- Dependencies: current PAIRV tree and the upstream/reference `.ci` scripts in `/home/kafcoppelia/WORK/BOARDS/nuclei-sdk/.ci/`
- Verification: base the recommendation on actual script contents, current tree contents, and current diff grouping

## Plan

- [x] Inspect `setup.sh` and the referenced `.ci` scripts from the reference nuclei-sdk tree.
- [x] Decide whether the `.ci/build_sdk.sh` and `.ci/build_applications.sh` sourcing is still useful for PAIRV.
- [x] Review the current focused diff and propose a clean multi-commit split without creating commits yet.

## Review

- `setup.sh` assessment:
  - current [setup.sh](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIRV/setup.sh) previously sourced:
    - `.ci/build_sdk.sh`
    - `.ci/build_applications.sh`
  - current PAIRV tree does not contain a `.ci/` directory at all, so those lines were guaranteed no-ops
  - in the reference `/home/kafcoppelia/WORK/BOARDS/nuclei-sdk/.ci/` tree:
    - `build_sdk.sh` only defines a convenience shell function `nsdk_build()`
    - `build_applications.sh` only defines a convenience shell function `nsdk_build_directory()`
  - those scripts are upstream CI/developer helpers, not a requirement for PAIRV’s current CLI environment setup
  - action taken:
    - removed the two dead `.ci` source lines from [setup.sh](/home/kafcoppelia/WORK/PAIRV/setup.sh)
    - verified with `bash -n ../PAIRV/setup.sh`
- Commit split recommendation for the current scoped changes:
  - `1.` Root build entry and application help
    - `Makefile`
    - `Build/Makefile.rules`
    - purpose: fix root dispatch, update help text, remove `tags/ctags`, align examples
  - `2.` Documentation and environment cleanup
    - `README.md`
    - `setup.sh`
    - optional: `.editorconfig`, `.gitignore` if you want repo hygiene in the same slice
    - purpose: rewrite docs to current active flow, remove dead `.ci` sourcing, keep environment/setup docs coherent
  - `3.` Application-specific policy and sample-tree cleanup
    - `application/baremetal/nice/Makefile`
    - removal of `SoC/evalsoc/runmode.mk`
    - removal of `application/freertos/smpdemo/**`
    - purpose: encode NICE runtime constraint and reflect the deliberate removal of obsolete sample/build pieces
- Important git-state caveat:
  - `git -C ../PAIRV status --short -- .editorconfig .gitignore Makefile README.md setup.sh application Build NMSIS OS SoC` currently shows many of these paths as untracked
  - so any real commit split should use explicit path staging, otherwise it is easy to accidentally sweep in the whole imported tree
  - I have not created commits yet
  - observed warnings:
    - PyTorch `TypedStorage` deprecation warning while loading the saved bundles
    - deprecated warning from legacy `clock_driven.LIFNode` lowering compatibility inside converter
    - Numba pending deprecation warnings from `paibox/backendv2/get_weight.py` during backend compile

# Legacy SpikingJelly clock_driven LIF Lowering Compatibility

## Workspace Decision

- [x] Continue in the current workspace on `feat/layout-two-pass-canonicalization`.
- [x] Keep this task in the shared workspace because the scope is a small, cohesive lowering/test compatibility change touching one implementation file and one focused test file.

## Ownership

- [x] Owner: Codex
- [x] Branch: `feat/layout-two-pass-canonicalization`
- [x] Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- [x] Allowed Files: `paibox/paiir/lowering/converter.py`, `tests/paiir/lowering/test_converter.py`, `tasks/todo.md`
- [x] Blocked Files: unrelated `tests/onboard/**`, backend files, and other in-flight edits
- [x] Dependencies: reuse the existing PAIIR warning hierarchy and existing activation-based lowering semantics
- [x] Verification: add a focused lowering test that proves legacy `clock_driven.LIFNode` is lowered and emits a deprecation warning

## Plan

- [x] Inspect current SpikingJelly neuron mapping and available warning types.
- [x] Extend `converter.py` to recognize legacy `spikingjelly.clock_driven.neuron.LIFNode`.
- [x] Emit a deprecation warning during lowering that explicitly recommends `spikingjelly.activation_based.neuron.LIFNode`.
- [x] Add and run a targeted test for the new compatibility path.
- [x] Extend the same compatibility layer to legacy `spikingjelly.clock_driven.neuron.IFNode`.
- [x] Make the legacy `clock_driven` import path optional so upstream removal does not break normal converter import.

## Review

- Updated [converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/paibox/paiir/lowering/converter.py) to support legacy `spikingjelly.clock_driven.neuron.LIFNode` when that module exists in the installed SpikingJelly version.
- The compatibility path now covers both legacy `clock_driven.IFNode` and `clock_driven.LIFNode`.
- The legacy warnings now describe the SpikingJelly usage itself as deprecated and point users to the corresponding `activation_based` type.
- The converter no longer hard-imports `spikingjelly.clock_driven` at module import time; it probes `spikingjelly.clock_driven.neuron` optionally and only registers the legacy mappings when that module still exists.
- Added focused tests in [test_converter.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/tests/paiir/lowering/test_converter.py):
  - legacy `clock_driven.IFNode` lowers successfully and emits the deprecation warning
  - legacy `clock_driven.LIFNode` lowers successfully and emits the deprecation warning
  - `activation_based.LIFNode` lowers without emitting that deprecation warning
- Verification:
  - `./.venv/bin/python -m py_compile paibox/paiir/lowering/converter.py tests/paiir/lowering/test_converter.py`
    - result: success
  - `printf '[pytest]\n' > /tmp/pytest-paiir-converter.ini`
  - `./.venv/bin/pytest -c /tmp/pytest-paiir-converter.ini --rootdir=/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox tests/paiir/lowering/test_converter.py -q`
    - result: `9 passed`

# Backendv2 Route Feasibility Inspection

## Workspace Decision

- [x] Continue in the current workspace on the current branch.
- [x] Keep this task in the shared workspace because it is a read-only inspection of existing backendv2 routing logic with no production edits.

## Ownership

- [x] Owner: Codex
- [x] Branch: current branch in `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- [x] Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox`
- [x] Allowed Files: `paibox/backendv2/route_solver.py`, `paibox/backendv2/mapper.py`, `paibox/backendv2/routing.py`, `tasks/todo.md`
- [x] Blocked Files: all production files outside the inspected routing scope
- [x] Dependencies: current OR-Tools placement model in `route_solver.py`, mapper call sites, and backendv2 routing-group data flow
- [x] Verification: trace the code path from mapper into the solver and routing-group construction, then answer with file/line references whether hard pairwise feasibility constraints exist

## Plan

- [x] Read the backendv2 placement and routing-feasibility code paths in `route_solver.py`, `mapper.py`, and `routing.py`.
- [x] Identify whether OR-Tools encodes hard pairwise routing-feasibility constraints or only a distance-based objective.
- [x] Determine whether the requested simultaneous pair set constraints are expressible in the current model or require new constraints, then record the conclusion below.

## Review

- `Mapper.routing()` topologically orders routing groups, derives `next_rg_group`, and passes that adjacency to `route_solve(...)` as `next_area_id`; no other feasibility artifact is passed into the solver.
- In `route_solver.py`, the CP-SAT model constrains only:
  - each requested area chooses exactly one placement
  - placements do not overlap on hive cells
  - center coordinates match the selected placement
  - per-edge absolute row/column deltas contribute to `total_distance`
- The solver does not add any hard constraint about a specific source-destination pair being simultaneously routeable with another pair, does not reserve NoC links, and does not model edge conflicts or path capacity.
- In `routing.py`, `set_detail_dest()` calls `find_coordxy_shortest_path(...)` only after placement is already fixed, using each source core and the destination routing group's base coordinate to derive an address offset for that connection. That path computation is not fed back into OR-Tools as a feasibility constraint.
- Conclusion:
  - current OR-Tools placement is a distance-minimizing placement heuristic over routing-group centers, plus non-overlap
  - the requested simultaneous pair set constraints (`rg1-rg2`, `rg3-rg4`, `rg1-rg3`, `rg2-rg4` all routeable together) are not hard-expressible in the current model as implemented
  - they would require new constraints and likely a richer routing-resource model or a post-placement feasibility checker coupled back into placement

# TensorLayout Refactor On Latest Dev

## Workspace Decision

- [x] Use a dedicated worktree because this is a broad interface refactor touching IR, lowering, passes, tests, and docs.
- [x] Work from `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-kafcoppelia-tensor-layout-refactor` on branch `feat-paiir-tensor-layout-refactor`, created from latest `origin/dev`.

## Ownership

- [x] Owner: Codex
- [x] Branch: `feat-paiir-tensor-layout-refactor`
- [x] Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-kafcoppelia-tensor-layout-refactor`
- [x] Allowed Files: `paibox/paiir/ir/**`, `paibox/paiir/lowering/**`, `paibox/paiir/pipeline/**`, `tests/paiir/**`, `docs/paiir_backend_guide.md`, `docs/paiir_compile_pass_design.md`, `docs/paiir_architecture_slides.md`, `tasks/todo.md`
- [x] Blocked Files: `paibox/backendv2/**`, `tests/backendv2/**`, unrelated docs/tests outside the listed scope
- [x] Dependencies: latest `origin/dev`, existing `Edge.src_port/dst_port` semantics, current FX lowering metadata flow, current graph simulation flow
- [x] Verification: targeted `tests/paiir/ir/**` and `tests/paiir/pipeline/**`, plus focused compile-time smoke where needed

## Plan

- [x] Introduce `TensorLayout` and replace stored OpNode shape/dims fields with `input_layouts/output_layouts`.
- [x] Update graph helpers and simulation to consume layouts while preserving `src_port/dst_port`.
- [x] Update lowering and split lowering to populate layouts.
- [x] Rewrite affected passes and layout passes to use layouts.
- [x] Update `tests/paiir/**` to construct and assert layouts.
- [x] Update the three PAIIR docs to document `TensorLayout` and backendv2 follow-up requirements.
- [x] Run targeted verification and record the result.

## Review

- Added `TensorLayout` as an immutable `shape + dims` carrier in `paibox/paiir/ir/op_node.py`, exported via `paibox.paiir.ir`.
- Replaced stored `OpNode` metadata fields with:
  - `input_layouts: tuple[TensorLayout, ...]`
  - `output_layouts: tuple[TensorLayout, ...]`
  - `num_inputs` / `num_outputs`
- Preserved `Edge.src_port` and `Edge.dst_port`; they remain the source-output index and destination-input slot respectively.
- Updated routing/runtime behavior:
  - `ReshapeOp.forward()` now reads `input_layouts[0].dims`
  - graph helpers can read concrete edge layouts through `graph.get_edge_output_layout(...)`
  - split branch selection still uses `src_port`
- Updated lowering and passes:
  - FX metadata copy now fills layouts rather than four separate shape/dims fields
  - split lowering materializes one input layout plus per-branch output layouts
  - fusion / layout passes / validation now consume `TensorLayout`
- Updated targeted tests in `tests/paiir/ir/**` and `tests/paiir/pipeline/**` to construct and assert layouts directly.
- Updated docs:
  - refreshed `docs/paiir_backend_guide.md`
  - added `docs/paiir_compile_pass_design.md`
  - added `docs/paiir_architecture_slides.md`
- Verification:
  - `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/.venv/bin/python -m py_compile paibox/paiir/ir/op_node.py paibox/paiir/ir/add_ops.py paibox/paiir/ir/graph.py paibox/paiir/lowering/fx_utils.py paibox/paiir/lowering/converter.py paibox/paiir/lowering/split_lowering.py paibox/paiir/pipeline/fusion_utils.py paibox/paiir/pipeline/layout_chain_canonicalization.py paibox/paiir/pipeline/layout_cross_node_elision.py paibox/paiir/pipeline/avgpool/fusion.py paibox/paiir/pipeline/passes.py`
  - `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/.venv/bin/pytest tests/paiir/ir tests/paiir/pipeline -q`
  - result: `528 passed, 2 skipped`

## Follow-up

- [x] Improve `graph.summary()` so multi-output nodes print per-output layouts and edge port semantics clearly.
- [x] Make `InputNode` / `OutputNode` carry explicit `TensorLayout` instead of only `shape`.
- [x] Generalize graph runtime tuple-output handling from `SplitOp`-only to a node-output-count-based protocol.
- [x] Add a compile/smoke example for `Split -> Concat -> Reshape` and write a stable summary log artifact for manual inspection.

### Follow-up Review

- `PAIIRGraph.format_summary()` now returns a stable summary string and `summary()` prints it.
- This was later simplified per review: only `summary(verbose=...)` remains as the public summary API.
- The summary now prints:
  - explicit boundary layouts for `InputNode` / `OutputNode`
  - per-input and per-output layouts for `OpNode`
  - full edge port annotations (`src_port` / `dst_port`)
- `InputNode` and `OutputNode` now store an explicit `layout: TensorLayout`; `shape`/`dims` are projections from that boundary layout.
- `_resolve_edge_tensor()` no longer hard-codes `SplitOp` as the only tuple-output runtime node; it now accepts any `OpNode` whose metadata declares `num_outputs > 1`.
- Added tests covering:
  - explicit boundary layout lookup
  - multi-output summary formatting
  - generic multi-output routing runtime
  - `torch_to_paiir` smoke for `Split -> Concat -> Reshape` summary rendering
- Generated example summary log for direct inspection:
  - `debug/paiir_split_concat_reshape.summary.log`

# Standalone AvgPool Binary Majority Policy

## Workspace Decision

- [x] Use a dedicated worktree because this is a non-trivial compiler feature touching compile config, passes, AvgPool lowering strategy, backend weight expansion, and tests.
- [x] Work from `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority` on branch `feat-paiir-avgpool-binary-majority`, created from `feat-paiir-tensor-layout-refactor`.

## Ownership

- [x] Owner: Codex
- [x] Branch: `feat-paiir-avgpool-binary-majority`
- [x] Worktree: `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority`
- [x] Allowed Files: `paibox/paiir/pipeline/**`, `paibox/paiir/ir/**`, `tests/paiir/**`, `tests/user/test_dvsgesture.py`, `tasks/**`
- [x] Blocked Files: `paibox/backendv2/**`, unrelated frontend/backend modules, docs unless the implementation requires user-facing API notes
- [x] Dependencies: tensor-layout refactor branch state, current AvgPool fusion logic, `CoreNeuronV25` SNN semantics, backendv2 pool weight expansion
- [x] Verification: targeted `py_compile`, focused `tests/paiir/pipeline/**`, and `tests/user/test_dvsgesture.py`

## Interface Notes

- Standalone `AvgPool` must not reuse the standalone `MaxPool` “transparent format/domain” rule.
- Standalone `AvgPool` is now handled automatically by the compiler; there is no public mode selector in `compile_to_paiir(...)` or `CompileConfig`.
- Automatic standalone AvgPool handling must:
  - distinguish upstream producer mode (`SNN` vs `ANN`)
  - distinguish resolved propagated input format
  - reject direct `InputNode -> AvgPool` and mixed upstream producer modes explicitly
- Preserve source-aligned DVSGesture assumptions used in the user test: spike input, `channels=8`, `Conv2d(..., bias=False)`, and actual source `IFNode + surrogate.ATan()` instantiation.

## Plan

- [x] Audit current standalone AvgPool path and finalize the `binary_majority` implementation shape plus compile-time rewrite-mode API under the no-`backendv2` constraint.
- [x] Implement the standalone AvgPool rewrite-mode framework and `binary_majority` rewrite as a pure frontend/compiler change.
- [x] Add focused regression coverage for graph rewriting, signal/data-format narrowing, and user-facing DVSGesture behavior.
- [x] Run targeted verification and record the results below.

## Review

- Added [standalone_rewrite.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/paiir/pipeline/avgpool/standalone_rewrite.py) as the standalone AvgPool auto-rewrite entry.
- Wiring changes:
  - [compile.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/paiir/pipeline/compile.py) now always runs standalone AvgPool auto-rewrite after the first mid-compile analysis round
  - public API no longer exposes a standalone AvgPool mode selector
  - introduced a structured revisit architecture:
    - [rewrite_phase.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/paiir/pipeline/rewrite_phase.py) now owns the generic “analyze -> rewrite -> re-analyze until fixed point” behavior
    - [compile.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/paiir/pipeline/compile.py) now only supplies compile-specific pieces:
      - `_run_mid_compile_analyses(...)`
      - `_post_fusion_rewrite_passes(...)`
  - this keeps `compile_to_paiir(...)` extensible for future analysis-dependent rewrites without prematurely coupling the whole pipeline to the more generic `pass_manager`
  - standalone AvgPool rewriting now runs after the first graph/domain/data-format analysis, and re-runs those analyses only if the graph changed
  - introduced [graph_utils.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/paibox/paiir/pipeline/graph_utils.py) to hold reusable effective-upstream traversal helpers and node predicates shared by `passes.py` and `standalone_rewrite.py`
- Policy behavior:
  - standalone `AvgPool1d/2d` is scanned after fusion and after the first graph/domain/data-format analysis
  - automatic dispatch now depends on both:
    - the resolved input format of the standalone AvgPool
    - the effective upstream producer mode traced through routing-only nodes and standalone MaxPool
  - direct `InputNode -> AvgPool` is rejected explicitly because no upstream producer mode exists
  - mixed upstream producer modes are rejected explicitly
  - `SNN + UNSIGNED/WIDTH_1BIT` input rewrites standalone AvgPool into `SequentialOp(SumPool, IFNodeV25)`
  - the IF node is configured as:
    - `v_threshold = 1`
    - `v_reset = 0`
    - `thres_neg_mode = FLOOR`
    - `thres_neg = 0`
    - `leak_v = -(majority_threshold - 1)`
  - this yields per-step-independent binary majority behavior and naturally narrows output format to `UNSIGNED/WIDTH_1BIT`
  - `ANN + signed/unsigned WIDTH_8BIT VALUE` input rewrites standalone AvgPool into `SequentialOp(SumPool, ANNNodeV25(LutCustom))`
  - the ANN rewrite uses an exact sum-domain LUT:
    - input to the LUT is the exact pooling-window sum
    - output is `torch.round(sum / divisor)` with the same tie rule as `torch.round`
    - `divisor_override` is respected through `_get_avgpool_divisor(...)`
  - this path guarantees **IR-level exactness** for integer VALUE inputs but does not claim backendv2 compile support
- Design boundary:
  - left `paibox/backendv2/**` untouched per user instruction
  - because backendv2 is frozen, this pass is implemented as a pure frontend/compiler feature
- Regression coverage:
- added [test_standalone_avgpool_rewrite.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/tests/paiir/pipeline/test_standalone_avgpool_rewrite.py)
  - new tests verify:
    - direct `InputNode -> AvgPool` now fails explicitly
    - spike-predecessor standalone AvgPool rewrites to `SumPool2d + IFNodeV25`
    - the rewritten spike path exactly matches manual binary-majority output
    - unsigned ANN producer -> standalone AvgPool rewrites to `SumPool2d + ANNNodeV25`
    - signed ANN producer -> standalone AvgPool rewrites to `SumPool2d + ANNNodeV25`
    - both ANN rewrites match `torch.round(avgpool(x.float()))` exactly on integer VALUE tensors
    - mixed upstream producer modes fail explicitly
- added [test_rewrite_phase.py](/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox-codex-avgpool-binary-majority/tests/paiir/pipeline/test_rewrite_phase.py)
  - verifies analysis replay occurs after a rewrite changes the graph
  - verifies the phase raises if rewrite rounds do not converge
- Verification:
  - `/home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/.venv/bin/python` one-off `py_compile.compile(..., cfile=/tmp/*.pyc, doraise=True)` for all changed Python files
  - `env COVERAGE_FILE=/tmp/standalone_avgpool_rewrite.coverage /home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/.venv/bin/pytest tests/paiir/pipeline/test_standalone_avgpool_rewrite.py -q`
  - result: `6 passed in 4.64s`
  - `env COVERAGE_FILE=/tmp/rewrite_phase.coverage /home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/.venv/bin/pytest tests/paiir/pipeline/test_rewrite_phase.py tests/paiir/pipeline/test_standalone_avgpool_rewrite.py -q`
  - result: `8 passed in 3.07s`
  - `env COVERAGE_FILE=/tmp/avgpool_compile.coverage /home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/.venv/bin/pytest tests/paiir/pipeline/test_compile.py -q -k 'avgpool or maxpool'`
  - result: `48 passed, 38 deselected in 7.46s`
  - `env COVERAGE_FILE=/tmp/avgpool_passes.coverage /home/kafcoppelia/WORK/PAIBox_Workgroup/PAIBox/.venv/bin/pytest tests/paiir/pipeline/test_passes.py -q -k 'maxpool or avgpool'`
  - result: `3 passed, 74 deselected in 4.02s`
