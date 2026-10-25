from .nodes import Hi3DRefineMesh

NODE_CLASS_MAPPINGS = {
    "Hi3DRefineMesh": Hi3DRefineMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hi3DRefineMesh": "Hi3D Refine Mesh (Sparc3D)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
