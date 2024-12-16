# The backend will use orjson if available.
try:
    import orjson

    JSON_BACKEND = "orjson"

except ModuleNotFoundError:
    JSON_BACKEND = "json"

print(f"Use {JSON_BACKEND} for json encoding.")
