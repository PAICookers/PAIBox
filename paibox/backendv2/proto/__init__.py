_PROTO_SCHEMA_VERSION = 2


def get_schema_version() -> int:
    return _PROTO_SCHEMA_VERSION


PROTO_SCHEMA_VERSION = _PROTO_SCHEMA_VERSION


__all__ = ["PROTO_SCHEMA_VERSION", "get_schema_version"]
