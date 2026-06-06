def get_osmgraphclip(*args, **kwargs):
    from .load import get_osmgraphclip as _fn
    return _fn(*args, **kwargs)


def get_osmgraphclip_loc_encoder(*args, **kwargs):
    from .load_lightweight import get_osmgraphclip_loc_encoder as _fn
    return _fn(*args, **kwargs)


__all__ = ["get_osmgraphclip", "get_osmgraphclip_loc_encoder"]
