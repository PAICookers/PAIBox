from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Shape(_message.Message):
    __slots__ = ("size",)
    SIZE_FIELD_NUMBER: _ClassVar[int]
    size: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, size: _Optional[_Iterable[int]] = ...) -> None: ...

class InputInfo(_message.Message):
    __slots__ = ("name", "shape")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    shape: Shape
    def __init__(self, name: _Optional[str] = ..., shape: _Optional[_Union[Shape, _Mapping]] = ...) -> None: ...

class OutputInfo(_message.Message):
    __slots__ = ("name", "shape")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    shape: Shape
    def __init__(self, name: _Optional[str] = ..., shape: _Optional[_Union[Shape, _Mapping]] = ...) -> None: ...

class InputEntry(_message.Message):
    __slots__ = ("idx", "copy_id", "bit_num", "tick_relative", "addr_axon", "addr_core_xy", "addr_core_x", "addr_core_y", "addr_copy_xy", "addr_copy_x", "addr_copy_y", "target_lcn")
    IDX_FIELD_NUMBER: _ClassVar[int]
    COPY_ID_FIELD_NUMBER: _ClassVar[int]
    BIT_NUM_FIELD_NUMBER: _ClassVar[int]
    TICK_RELATIVE_FIELD_NUMBER: _ClassVar[int]
    ADDR_AXON_FIELD_NUMBER: _ClassVar[int]
    ADDR_CORE_XY_FIELD_NUMBER: _ClassVar[int]
    ADDR_CORE_X_FIELD_NUMBER: _ClassVar[int]
    ADDR_CORE_Y_FIELD_NUMBER: _ClassVar[int]
    ADDR_COPY_XY_FIELD_NUMBER: _ClassVar[int]
    ADDR_COPY_X_FIELD_NUMBER: _ClassVar[int]
    ADDR_COPY_Y_FIELD_NUMBER: _ClassVar[int]
    TARGET_LCN_FIELD_NUMBER: _ClassVar[int]
    idx: int
    copy_id: int
    bit_num: int
    tick_relative: int
    addr_axon: int
    addr_core_xy: int
    addr_core_x: int
    addr_core_y: int
    addr_copy_xy: int
    addr_copy_x: int
    addr_copy_y: int
    target_lcn: int
    def __init__(self, idx: _Optional[int] = ..., copy_id: _Optional[int] = ..., bit_num: _Optional[int] = ..., tick_relative: _Optional[int] = ..., addr_axon: _Optional[int] = ..., addr_core_xy: _Optional[int] = ..., addr_core_x: _Optional[int] = ..., addr_core_y: _Optional[int] = ..., addr_copy_xy: _Optional[int] = ..., addr_copy_x: _Optional[int] = ..., addr_copy_y: _Optional[int] = ..., target_lcn: _Optional[int] = ...) -> None: ...

class OutputEntry(_message.Message):
    __slots__ = ("idx", "copy_id", "bit_num", "axon_addr")
    IDX_FIELD_NUMBER: _ClassVar[int]
    COPY_ID_FIELD_NUMBER: _ClassVar[int]
    BIT_NUM_FIELD_NUMBER: _ClassVar[int]
    AXON_ADDR_FIELD_NUMBER: _ClassVar[int]
    idx: int
    copy_id: int
    bit_num: int
    axon_addr: int
    def __init__(self, idx: _Optional[int] = ..., copy_id: _Optional[int] = ..., bit_num: _Optional[int] = ..., axon_addr: _Optional[int] = ...) -> None: ...

class Relative_Coord(_message.Message):
    __slots__ = ("xy", "x", "y")
    XY_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    xy: int
    x: int
    y: int
    def __init__(self, xy: _Optional[int] = ..., x: _Optional[int] = ..., y: _Optional[int] = ...) -> None: ...

class InputEntryList(_message.Message):
    __slots__ = ("entry",)
    ENTRY_FIELD_NUMBER: _ClassVar[int]
    entry: _containers.RepeatedCompositeFieldContainer[InputEntry]
    def __init__(self, entry: _Optional[_Iterable[_Union[InputEntry, _Mapping]]] = ...) -> None: ...

class OutputEntryList(_message.Message):
    __slots__ = ("entry",)
    ENTRY_FIELD_NUMBER: _ClassVar[int]
    entry: _containers.RepeatedCompositeFieldContainer[OutputEntry]
    def __init__(self, entry: _Optional[_Iterable[_Union[OutputEntry, _Mapping]]] = ...) -> None: ...

class InputInfoWithEntry(_message.Message):
    __slots__ = ("name", "shape", "input_entry_list")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    INPUT_ENTRY_LIST_FIELD_NUMBER: _ClassVar[int]
    name: str
    shape: Shape
    input_entry_list: InputEntryList
    def __init__(self, name: _Optional[str] = ..., shape: _Optional[_Union[Shape, _Mapping]] = ..., input_entry_list: _Optional[_Union[InputEntryList, _Mapping]] = ...) -> None: ...

class OutputInfoWithEntry(_message.Message):
    __slots__ = ("name", "shape", "output_entry_list")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_ENTRY_LIST_FIELD_NUMBER: _ClassVar[int]
    name: str
    shape: Shape
    output_entry_list: OutputEntryList
    def __init__(self, name: _Optional[str] = ..., shape: _Optional[_Union[Shape, _Mapping]] = ..., output_entry_list: _Optional[_Union[OutputEntryList, _Mapping]] = ...) -> None: ...

class InputInfoList(_message.Message):
    __slots__ = ("input_info",)
    INPUT_INFO_FIELD_NUMBER: _ClassVar[int]
    input_info: _containers.RepeatedCompositeFieldContainer[InputInfoWithEntry]
    def __init__(self, input_info: _Optional[_Iterable[_Union[InputInfoWithEntry, _Mapping]]] = ...) -> None: ...

class OutputInfoList(_message.Message):
    __slots__ = ("output_info",)
    OUTPUT_INFO_FIELD_NUMBER: _ClassVar[int]
    output_info: _containers.RepeatedCompositeFieldContainer[OutputInfoWithEntry]
    def __init__(self, output_info: _Optional[_Iterable[_Union[OutputInfoWithEntry, _Mapping]]] = ...) -> None: ...

class IOConfigSingleThread(_message.Message):
    __slots__ = ("thread_id", "root_core", "input_info_list", "output_info_list")
    THREAD_ID_FIELD_NUMBER: _ClassVar[int]
    ROOT_CORE_FIELD_NUMBER: _ClassVar[int]
    INPUT_INFO_LIST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_INFO_LIST_FIELD_NUMBER: _ClassVar[int]
    thread_id: int
    root_core: Relative_Coord
    input_info_list: InputInfoList
    output_info_list: OutputInfoList
    def __init__(self, thread_id: _Optional[int] = ..., root_core: _Optional[_Union[Relative_Coord, _Mapping]] = ..., input_info_list: _Optional[_Union[InputInfoList, _Mapping]] = ..., output_info_list: _Optional[_Union[OutputInfoList, _Mapping]] = ...) -> None: ...

class IOConfig(_message.Message):
    __slots__ = ("thread_io",)
    THREAD_IO_FIELD_NUMBER: _ClassVar[int]
    thread_io: _containers.RepeatedCompositeFieldContainer[IOConfigSingleThread]
    def __init__(self, thread_io: _Optional[_Iterable[_Union[IOConfigSingleThread, _Mapping]]] = ...) -> None: ...

class FrameList(_message.Message):
    __slots__ = ("frame",)
    FRAME_FIELD_NUMBER: _ClassVar[int]
    frame: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, frame: _Optional[_Iterable[int]] = ...) -> None: ...

class Config(_message.Message):
    __slots__ = ("version", "io_config", "frame_list")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    IO_CONFIG_FIELD_NUMBER: _ClassVar[int]
    FRAME_LIST_FIELD_NUMBER: _ClassVar[int]
    version: int
    io_config: IOConfig
    frame_list: FrameList
    def __init__(self, version: _Optional[int] = ..., io_config: _Optional[_Union[IOConfig, _Mapping]] = ..., frame_list: _Optional[_Union[FrameList, _Mapping]] = ...) -> None: ...
