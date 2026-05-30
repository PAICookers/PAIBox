_PROTO_SCHEMA_VERSION = 1


def get_schema_version() -> int:
    return _PROTO_SCHEMA_VERSION


__all__ = ["get_schema_version"]
