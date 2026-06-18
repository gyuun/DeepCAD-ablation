from .autoencoder import CADTransformer, CADTransformerAblation


def build_model(cfg):
    model_type = getattr(cfg, "model_type", "baseline")
    if model_type == "baseline":
        return CADTransformer(cfg)
    if model_type == "ablation":
        return CADTransformerAblation(cfg)
    raise ValueError("Unknown model type: {}".format(model_type))
