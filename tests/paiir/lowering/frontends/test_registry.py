from collections.abc import Callable

import pytest
import torch
from spikingjelly.activation_based import layer, neuron
from torch import nn

from paibox.paiir.ir.op_node import OpNode
from paibox.paiir.lowering import converter
from paibox.paiir.lowering.frontends import registry
from paibox.paiir.lowering.frontends import snntorch as snntorch_frontend
from paibox.paiir.lowering.support import (
    MISSING,
    AttributeView,
    SourceOpSchema,
    attr,
    eq,
    one_of,
    reader,
    scalar,
    source_op_schema,
)


class DummyModule(nn.Module):
    def forward(self, x):
        return x


class FakeFrontend:
    def __init__(self, name: str, mapper: Callable[[nn.Module], OpNode]) -> None:
        self.name = name
        self.erase_types: tuple[type[nn.Module], ...] = ()
        self.source_schemas: tuple[SourceOpSchema, ...] = ()
        self._mapper = mapper

    def detect(self, model: nn.Module) -> bool:
        return any(self.owns_module(module) for module in model.modules())

    def prepare(self, model: nn.Module) -> nn.Module:
        return model

    def build_module_map(self):
        return {DummyModule: self._mapper}

    def owns_module(self, module: nn.Module) -> bool:
        return isinstance(module, DummyModule)


def _frontend_mapper(_: nn.Module) -> OpNode:
    raise AssertionError("frontend mapper should not be called")


def _user_mapper(_: nn.Module) -> OpNode:
    raise AssertionError("user mapper should not be called")


def test_spikingjelly_frontend_detects_activation_based_modules():
    model = nn.Sequential(layer.Linear(2, 2), neuron.IFNode())

    names = {frontend.name for frontend in registry.resolve_frontends(model)}

    assert "spikingjelly" in names


def test_snntorch_frontend_detects_loaded_leaky_module():
    snn = pytest.importorskip("snntorch")
    model = nn.Sequential(snn.Leaky(beta=1.0, init_hidden=True, reset_delay=False))

    names = {frontend.name for frontend in registry.resolve_frontends(model)}

    assert "snntorch" in names


def test_snntorch_schema_matches_loaded_exact_leaky_type():
    snn = pytest.importorskip("snntorch")

    resolution = snntorch_frontend.source_schemas[0].resolve(
        snn.Leaky(beta=1.0, init_hidden=True, reset_delay=False)
    )

    assert resolution is not None


def test_frontend_module_map_conflict_raises():
    frontends = (
        FakeFrontend("frontend_a", _frontend_mapper),
        FakeFrontend("frontend_b", _frontend_mapper),
    )

    with pytest.raises(ValueError, match="frontend module map conflict"):
        registry.build_frontend_module_map(frontends)


def test_user_module_map_overrides_frontend_map(monkeypatch):
    monkeypatch.setattr(converter, "_USER_MODULE_MAP", {DummyModule: _user_mapper})

    full_map = converter._get_full_module_map({DummyModule: _frontend_mapper})

    assert full_map[DummyModule] is _user_mapper


def test_source_op_schema_builder_returns_dataclass_and_canonicalizes():
    schema = (
        source_op_schema(
            "fake", "Dummy", recognize=lambda m: isinstance(m, DummyModule)
        )
        .attribute("enabled", attr("enabled", default=True))
        .attribute("mode", attr("mode", default="linear"))
        .generic(eq("enabled", True))
        .case("linear")
        .when(eq("mode", "linear"))
        .canonicalize_to(lambda op: op.attributes.raw("mode"))
        .build()
    )

    resolution = schema.resolve(DummyModule())

    assert isinstance(schema, SourceOpSchema)
    assert resolution is not None
    assert resolution.supported
    assert isinstance(resolution.source_op.attributes, AttributeView)
    assert resolution.canonicalize() == "linear"


def test_attribute_reader_and_view_format_values():
    module = DummyModule()
    custom_reader = reader(lambda _: "custom-value")
    default_reader = attr("missing", default=3)
    param = nn.Parameter(torch.ones(2))
    view = AttributeView({"tensor": torch.ones(2), "param": param, "missing": MISSING})

    assert custom_reader(module) == "custom-value"
    assert default_reader(module) == 3
    assert view.display("tensor") == "Tensor(shape=(2,))"
    assert view.display("param") == "Parameter(shape=(2,))"
    assert view.display("missing") == "<missing>"


def test_source_op_schema_reports_generic_constraint_failure():
    schema = (
        source_op_schema(
            "fake", "Dummy", recognize=lambda m: isinstance(m, DummyModule)
        )
        .attribute("enabled", attr("enabled", default=False))
        .generic(eq("enabled", True, reason="must be enabled"))
        .case("enabled")
        .canonicalize_to(lambda op: op.module)
        .build()
    )

    resolution = schema.resolve(DummyModule())

    assert resolution is not None
    assert not resolution.supported
    assert resolution.error is not None
    assert resolution.error.field == "enabled"
    assert resolution.error.value is False
    assert resolution.error.reason == "must be enabled"


def test_source_op_schema_reports_no_matching_case():
    schema = (
        source_op_schema(
            "fake", "Dummy", recognize=lambda m: isinstance(m, DummyModule)
        )
        .attribute("mode", attr("mode", default="unknown"))
        .generic(one_of("mode", ("linear", "zero", "unknown")))
        .case("linear")
        .when(eq("mode", "linear"))
        .canonicalize_to(lambda op: op.module)
        .build()
    )

    resolution = schema.resolve(DummyModule())

    assert resolution is not None
    assert not resolution.supported
    assert resolution.error is not None
    assert resolution.error.case == "linear"
    assert resolution.error.field == "mode"
    assert resolution.error.value == "unknown"


def test_source_op_schema_reports_ambiguous_case():
    schema = (
        source_op_schema(
            "fake", "Dummy", recognize=lambda m: isinstance(m, DummyModule)
        )
        .attribute("mode", attr("mode", default="linear"))
        .case("first")
        .when(eq("mode", "linear"))
        .canonicalize_to(lambda op: op.module)
        .case("second")
        .when(eq("mode", "linear"))
        .canonicalize_to(lambda op: op.module)
        .build()
    )

    resolution = schema.resolve(DummyModule())

    assert resolution is not None
    assert not resolution.supported
    assert resolution.error is not None
    assert resolution.error.constraint == "ambiguous case"
    assert "first, second" in str(resolution.error.reason)


def test_source_op_schema_supports_scalar_predicates():
    schema = (
        source_op_schema(
            "fake", "Dummy", recognize=lambda m: isinstance(m, DummyModule)
        )
        .attribute("value", attr("value", default=torch.tensor(1.0)))
        .generic(scalar("value"))
        .generic(eq("value", 1.0))
        .case("scalar_one")
        .canonicalize_to(lambda op: op.attributes.raw("value"))
        .build()
    )

    resolution = schema.resolve(DummyModule())

    assert resolution is not None
    assert resolution.supported
