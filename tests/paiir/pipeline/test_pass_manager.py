from paibox.paiir.ir.graph import PAIIRGraph
from paibox.paiir.pipeline.pass_manager import (
    CompileInfo,
    CompilePass,
    CompilePassManager,
    CompilePassResult,
)


class TestCompilePassManager:
    def test_reruns_invalidated_requirements(self):
        graph = PAIIRGraph("g")
        manager = CompilePassManager(graph)
        calls: list[str] = []

        def run_validate(g, _ctx):
            calls.append("validate")
            return CompilePassResult(
                analysis_updates={CompileInfo.VALIDATED: calls.count("validate")}
            )

        validate_pass = CompilePass(
            name="validate_graph",
            func=run_validate,
            provides=frozenset({CompileInfo.VALIDATED}),
        )
        manager.register_provider(validate_pass)

        def run_compensation(g, ctx):
            calls.append("compensate")
            assert CompileInfo.VALIDATED in ctx.available_info
            assert ctx.analysis_data[CompileInfo.VALIDATED] == 1
            return CompilePassResult(invalidated_info={CompileInfo.VALIDATED})

        compensation_pass = CompilePass(
            name="compensate",
            func=run_compensation,
            requires=frozenset({CompileInfo.VALIDATED}),
        )

        def run_consumer(g, ctx):
            calls.append("consumer")
            assert ctx.analysis_data[CompileInfo.VALIDATED] == 2

        consumer_pass = CompilePass(
            name="consumer",
            func=run_consumer,
            requires=frozenset({CompileInfo.VALIDATED}),
        )

        manager.run_pass(compensation_pass)
        manager.run_pass(consumer_pass)

        assert calls == ["validate", "compensate", "validate", "consumer"]

    def test_invalidation_cascades_to_dependent_analyses(self):
        graph = PAIIRGraph("g")
        manager = CompilePassManager(graph)
        counts = {"validate": 0, "format": 0, "tick": 0}

        def run_validate(g, _ctx):
            counts["validate"] += 1

        validate_pass = CompilePass(
            name="validate_graph",
            func=run_validate,
            provides=frozenset({CompileInfo.VALIDATED}),
        )

        def run_format(g, _ctx):
            counts["format"] += 1

        format_pass = CompilePass(
            name="propagate_data_format",
            func=run_format,
            requires=frozenset({CompileInfo.VALIDATED}),
            provides=frozenset({CompileInfo.DATA_FORMAT}),
        )

        def run_tick(g, _ctx):
            counts["tick"] += 1

        tick_pass = CompilePass(
            name="assign_tick_params",
            func=run_tick,
            requires=frozenset({CompileInfo.DATA_FORMAT}),
            provides=frozenset({CompileInfo.TICK_PARAMS}),
        )

        for pass_spec in (validate_pass, format_pass, tick_pass):
            manager.register_provider(pass_spec)

        manager.run_pass(
            CompilePass(
                name="tick_consumer",
                func=lambda g, _ctx: None,
                requires=frozenset({CompileInfo.TICK_PARAMS}),
            )
        )
        assert counts == {"validate": 1, "format": 1, "tick": 1}

        manager.run_pass(
            CompilePass(
                name="topology_compensation",
                func=lambda g, _ctx: CompilePassResult(
                    invalidated_info={CompileInfo.VALIDATED}
                ),
                requires=frozenset({CompileInfo.DATA_FORMAT}),
            )
        )

        assert manager.context.available_info == set()

        manager.run_pass(
            CompilePass(
                name="tick_consumer_again",
                func=lambda g, _ctx: None,
                requires=frozenset({CompileInfo.TICK_PARAMS}),
            )
        )
        assert counts == {"validate": 2, "format": 2, "tick": 2}

    def test_graph_replacement_updates_manager(self):
        old_graph = PAIIRGraph("old")
        new_graph = PAIIRGraph("new")
        manager = CompilePassManager(old_graph)

        manager.run_pass(
            CompilePass(
                name="replace_graph",
                func=lambda g, _ctx: new_graph,
            )
        )

        assert manager.graph is new_graph
