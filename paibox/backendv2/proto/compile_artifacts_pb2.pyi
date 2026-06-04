# ruff: noqa: I001
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor

class CoreOffset(_message.Message):
    __slots__ = ("xy", "x", "y")
    XY_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    xy: int
    x: int
    y: int
    def __init__(
        self, xy: _Optional[int] = ..., x: _Optional[int] = ..., y: _Optional[int] = ...
    ) -> None: ...

class CopyCount(_message.Message):
    __slots__ = ("xy", "x", "y")
    XY_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    xy: int
    x: int
    y: int
    def __init__(
        self, xy: _Optional[int] = ..., x: _Optional[int] = ..., y: _Optional[int] = ...
    ) -> None: ...

class TickParams(_message.Message):
    __slots__ = ("tick_start", "tick_duration", "tick_initial")
    TICK_START_FIELD_NUMBER: _ClassVar[int]
    TICK_DURATION_FIELD_NUMBER: _ClassVar[int]
    TICK_INITIAL_FIELD_NUMBER: _ClassVar[int]
    tick_start: int
    tick_duration: int
    tick_initial: int
    def __init__(
        self,
        tick_start: _Optional[int] = ...,
        tick_duration: _Optional[int] = ...,
        tick_initial: _Optional[int] = ...,
    ) -> None: ...

class DataType(_message.Message):
    __slots__ = ()

    class Code(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        NOT_SET: _ClassVar[DataType.Code]
        UINT1: _ClassVar[DataType.Code]
        INT1: _ClassVar[DataType.Code]
        UINT2: _ClassVar[DataType.Code]
        INT2: _ClassVar[DataType.Code]
        UINT4: _ClassVar[DataType.Code]
        INT4: _ClassVar[DataType.Code]
        UINT8: _ClassVar[DataType.Code]
        INT8: _ClassVar[DataType.Code]

    NOT_SET: DataType.Code
    UINT1: DataType.Code
    INT1: DataType.Code
    UINT2: DataType.Code
    INT2: DataType.Code
    UINT4: DataType.Code
    INT4: DataType.Code
    UINT8: DataType.Code
    INT8: DataType.Code
    def __init__(self) -> None: ...

class InputEntry(_message.Message):
    __slots__ = (
        "elem_idx",
        "core_offset",
        "copy_count",
        "tick_relative",
        "addr_axon",
        "target_lcn",
        "copy_id",
        "dtype",
    )
    ELEM_IDX_FIELD_NUMBER: _ClassVar[int]
    CORE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    COPY_COUNT_FIELD_NUMBER: _ClassVar[int]
    TICK_RELATIVE_FIELD_NUMBER: _ClassVar[int]
    ADDR_AXON_FIELD_NUMBER: _ClassVar[int]
    TARGET_LCN_FIELD_NUMBER: _ClassVar[int]
    COPY_ID_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    elem_idx: int
    core_offset: CoreOffset
    copy_count: CopyCount
    tick_relative: int
    addr_axon: int
    target_lcn: int
    copy_id: int
    dtype: DataType.Code
    def __init__(
        self,
        elem_idx: _Optional[int] = ...,
        core_offset: _Optional[_Union[CoreOffset, _Mapping]] = ...,
        copy_count: _Optional[_Union[CopyCount, _Mapping]] = ...,
        tick_relative: _Optional[int] = ...,
        addr_axon: _Optional[int] = ...,
        target_lcn: _Optional[int] = ...,
        copy_id: _Optional[int] = ...,
        dtype: _Optional[_Union[DataType.Code, str]] = ...,
    ) -> None: ...

class OutputEntry(_message.Message):
    __slots__ = ("elem_idx", "copy_id", "axon_bit_idx", "dtype")
    ELEM_IDX_FIELD_NUMBER: _ClassVar[int]
    COPY_ID_FIELD_NUMBER: _ClassVar[int]
    AXON_BIT_IDX_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    elem_idx: int
    copy_id: int
    axon_bit_idx: int
    dtype: DataType.Code
    def __init__(
        self,
        elem_idx: _Optional[int] = ...,
        copy_id: _Optional[int] = ...,
        axon_bit_idx: _Optional[int] = ...,
        dtype: _Optional[_Union[DataType.Code, str]] = ...,
    ) -> None: ...

class Shape(_message.Message):
    __slots__ = ("size",)
    SIZE_FIELD_NUMBER: _ClassVar[int]
    size: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, size: _Optional[_Iterable[int]] = ...) -> None: ...

class InputTensorMapping(_message.Message):
    __slots__ = ("name", "shape", "bit_width", "tick", "entries")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    BIT_WIDTH_FIELD_NUMBER: _ClassVar[int]
    TICK_FIELD_NUMBER: _ClassVar[int]
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    name: str
    shape: Shape
    bit_width: int
    tick: TickParams
    entries: _containers.RepeatedCompositeFieldContainer[InputEntry]
    def __init__(
        self,
        name: _Optional[str] = ...,
        shape: _Optional[_Union[Shape, _Mapping]] = ...,
        bit_width: _Optional[int] = ...,
        tick: _Optional[_Union[TickParams, _Mapping]] = ...,
        entries: _Optional[_Iterable[_Union[InputEntry, _Mapping]]] = ...,
    ) -> None: ...

class OutputTensorMapping(_message.Message):
    __slots__ = ("name", "shape", "kind", "bit_width", "tick", "entries")

    class OutputKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        DATA: _ClassVar[OutputTensorMapping.OutputKind]
        VOLTAGE: _ClassVar[OutputTensorMapping.OutputKind]

    DATA: OutputTensorMapping.OutputKind
    VOLTAGE: OutputTensorMapping.OutputKind
    NAME_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    BIT_WIDTH_FIELD_NUMBER: _ClassVar[int]
    TICK_FIELD_NUMBER: _ClassVar[int]
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    name: str
    shape: Shape
    kind: OutputTensorMapping.OutputKind
    bit_width: int
    tick: TickParams
    entries: _containers.RepeatedCompositeFieldContainer[OutputEntry]
    def __init__(
        self,
        name: _Optional[str] = ...,
        shape: _Optional[_Union[Shape, _Mapping]] = ...,
        kind: _Optional[_Union[OutputTensorMapping.OutputKind, str]] = ...,
        bit_width: _Optional[int] = ...,
        tick: _Optional[_Union[TickParams, _Mapping]] = ...,
        entries: _Optional[_Iterable[_Union[OutputEntry, _Mapping]]] = ...,
    ) -> None: ...

class InputTensorMappings(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[InputTensorMapping]
    def __init__(
        self, items: _Optional[_Iterable[_Union[InputTensorMapping, _Mapping]]] = ...
    ) -> None: ...

class OutputTensorMappings(_message.Message):
    __slots__ = ("target_lcn", "items")
    TARGET_LCN_FIELD_NUMBER: _ClassVar[int]
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    target_lcn: int
    items: _containers.RepeatedCompositeFieldContainer[OutputTensorMapping]
    def __init__(
        self,
        target_lcn: _Optional[int] = ...,
        items: _Optional[_Iterable[_Union[OutputTensorMapping, _Mapping]]] = ...,
    ) -> None: ...

class RuntimeParams(_message.Message):
    __slots__ = ("timesteps", "tick_depth", "sync_steps", "decode_mode")

    class DecodeMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STREAM: _ClassVar[RuntimeParams.DecodeMode]
        STEP: _ClassVar[RuntimeParams.DecodeMode]

    STREAM: RuntimeParams.DecodeMode
    STEP: RuntimeParams.DecodeMode
    TIMESTEPS_FIELD_NUMBER: _ClassVar[int]
    TICK_DEPTH_FIELD_NUMBER: _ClassVar[int]
    SYNC_STEPS_FIELD_NUMBER: _ClassVar[int]
    DECODE_MODE_FIELD_NUMBER: _ClassVar[int]
    timesteps: int
    tick_depth: int
    sync_steps: int
    decode_mode: RuntimeParams.DecodeMode
    def __init__(
        self,
        timesteps: _Optional[int] = ...,
        tick_depth: _Optional[int] = ...,
        sync_steps: _Optional[int] = ...,
        decode_mode: _Optional[_Union[RuntimeParams.DecodeMode, str]] = ...,
    ) -> None: ...

class CoreTick(_message.Message):
    __slots__ = ("core_offset", "tick", "nodes")
    CORE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    TICK_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    core_offset: CoreOffset
    tick: TickParams
    nodes: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        core_offset: _Optional[_Union[CoreOffset, _Mapping]] = ...,
        tick: _Optional[_Union[TickParams, _Mapping]] = ...,
        nodes: _Optional[_Iterable[str]] = ...,
    ) -> None: ...

class ThreadIOMapping(_message.Message):
    __slots__ = (
        "thread_id",
        "root_core_offset",
        "runtime",
        "input_mappings",
        "output_mappings",
        "core_ticks",
    )
    THREAD_ID_FIELD_NUMBER: _ClassVar[int]
    ROOT_CORE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_FIELD_NUMBER: _ClassVar[int]
    INPUT_MAPPINGS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_MAPPINGS_FIELD_NUMBER: _ClassVar[int]
    CORE_TICKS_FIELD_NUMBER: _ClassVar[int]
    thread_id: int
    root_core_offset: CoreOffset
    runtime: RuntimeParams
    input_mappings: InputTensorMappings
    output_mappings: OutputTensorMappings
    core_ticks: _containers.RepeatedCompositeFieldContainer[CoreTick]
    def __init__(
        self,
        thread_id: _Optional[int] = ...,
        root_core_offset: _Optional[_Union[CoreOffset, _Mapping]] = ...,
        runtime: _Optional[_Union[RuntimeParams, _Mapping]] = ...,
        input_mappings: _Optional[_Union[InputTensorMappings, _Mapping]] = ...,
        output_mappings: _Optional[_Union[OutputTensorMappings, _Mapping]] = ...,
        core_ticks: _Optional[_Iterable[_Union[CoreTick, _Mapping]]] = ...,
    ) -> None: ...

class IOMapping(_message.Message):
    __slots__ = ("threads",)
    THREADS_FIELD_NUMBER: _ClassVar[int]
    threads: _containers.RepeatedCompositeFieldContainer[ThreadIOMapping]
    def __init__(
        self, threads: _Optional[_Iterable[_Union[ThreadIOMapping, _Mapping]]] = ...
    ) -> None: ...

class ConfigFrames(_message.Message):
    __slots__ = ("words", "word_order")

    class WordOrder(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        HIGH_FIRST: _ClassVar[ConfigFrames.WordOrder]
        LOW_FIRST: _ClassVar[ConfigFrames.WordOrder]

    HIGH_FIRST: ConfigFrames.WordOrder
    LOW_FIRST: ConfigFrames.WordOrder
    WORDS_FIELD_NUMBER: _ClassVar[int]
    WORD_ORDER_FIELD_NUMBER: _ClassVar[int]
    words: _containers.RepeatedScalarFieldContainer[int]
    word_order: ConfigFrames.WordOrder
    def __init__(
        self,
        words: _Optional[_Iterable[int]] = ...,
        word_order: _Optional[_Union[ConfigFrames.WordOrder, str]] = ...,
    ) -> None: ...

class CompileArtifacts(_message.Message):
    __slots__ = ("schema_version", "io_mapping", "config_frames")
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    IO_MAPPING_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FRAMES_FIELD_NUMBER: _ClassVar[int]
    schema_version: int
    io_mapping: IOMapping
    config_frames: ConfigFrames
    def __init__(
        self,
        schema_version: _Optional[int] = ...,
        io_mapping: _Optional[_Union[IOMapping, _Mapping]] = ...,
        config_frames: _Optional[_Union[ConfigFrames, _Mapping]] = ...,
    ) -> None: ...
