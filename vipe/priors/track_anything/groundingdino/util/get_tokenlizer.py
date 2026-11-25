import os

from transformers import AutoConfig, AutoTokenizer, BertModel, RobertaModel

from vipe.utils.hub import load_local_first


def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )

    tokenizer = load_local_first(AutoTokenizer.from_pretrained, text_encoder_type)
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    """Build the text encoder architecture without loading pretrained weights.

    The GroundingDINO checkpoint stores every text-encoder parameter under
    ``bert.*`` and the detector loads it right after construction, so the
    Hub weights would be overwritten immediately. Only the config is fetched,
    which also avoids the pretraining-head (``cls.*``) "UNEXPECTED" load report.
    """
    if text_encoder_type == "bert-base-uncased" or (
        os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)
    ):
        return BertModel(load_local_first(AutoConfig.from_pretrained, text_encoder_type))
    if text_encoder_type == "roberta-base":
        return RobertaModel(load_local_first(AutoConfig.from_pretrained, text_encoder_type))

    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
