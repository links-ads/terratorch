import importlib
import logging
import sys
from collections.abc import Callable

import timm
import torch
from box import Box
from torch import nn

import terratorch.models.decoders as decoder_registry
from terratorch.models.backbones.clay_v15.model import ClayMAEBackbone
from terratorch.models.model import AuxiliaryHead, AuxiliaryHeadWithDecoderWithoutInstantiatedHead, Model, ModelFactory
from terratorch.models.pixel_wise_model import PixelWiseModel
from terratorch.models.scalar_output_model import ScalarOutputModel
from terratorch.models.utils import DecoderNotFoundError, extract_prefix_keys
from terratorch.registry import MODEL_FACTORY_REGISTRY

PIXEL_WISE_TASKS = ["segmentation", "regression"]
SCALAR_TASKS = ["classification", "scalar_regression"]
SUPPORTED_TASKS = PIXEL_WISE_TASKS + SCALAR_TASKS


@MODEL_FACTORY_REGISTRY.register
class Clay1_5ModelFactory(ModelFactory):

    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: str | nn.Module,
        in_channels: int,
        bands: list[int] = [],
        num_classes: int | None = None,
        pretrained: bool = True,  # noqa: FBT001, FBT002
        num_frames: int = 1,
        prepare_features_for_image_model: Callable | None = None,
        aux_decoders: list[AuxiliaryHead] | None = None,
        rescale: bool = True,  # noqa: FBT002, FBT001
        checkpoint_path: str = None,
        **kwargs,
    ) -> Model:
        # try:
        #     from claymodel.model import ClayMAE
        # except ImportError:
        #     message = "clay v1.5 not installed, please use pip install claimodel"
        #     logging.getLogger("terratorch").debug(message)
        #    raise Exception(message)
        backbone_kwargs, kwargs = extract_prefix_keys(kwargs, "backbone_")

        padding = backbone_kwargs.get("padding", "reflect")
        kwargs["metadata"] = Box(kwargs["metadata"])
        patch_size = kwargs.get("patch_size")
        # return ModelWrapper(batch_size, bands, platform, ClayMAE(**kwargs))
        encoder = ClayMAEBackbone(
            patch_size=patch_size,
            shuffle=backbone_kwargs.get("shuffle"),
            dim=backbone_kwargs.get("dim"),
            depth=backbone_kwargs.get("depth"),
            heads=backbone_kwargs.get("heads"),
            dim_head=backbone_kwargs.get("dim_head"),
            mlp_ratio=backbone_kwargs.get("mlp_ratio"),
        )
        # allow decoder to be a module passed directly
        decoder_cls = _get_decoder(decoder)
        decoder_kwargs, kwargs = extract_prefix_keys(kwargs, "decoder_")

        decoder: nn.Module = decoder_cls(backbone.feature_info.channels(),
                                         **decoder_kwargs)
        # decoder: nn.Module = decoder_cls([128, 256, 512, 1024], **decoder_kwargs)

        head_kwargs, kwargs = extract_prefix_keys(kwargs, "head_")
        if num_classes:
            head_kwargs["num_classes"] = num_classes
        if aux_decoders is None:
            return _build_appropriate_model(task,
                                            backbone,
                                            decoder,
                                            head_kwargs,
                                            prepare_features_for_image_model,
                                            patch_size=patch_size,
                                            padding=padding,
                                            rescale=rescale)

        to_be_aux_decoders: list[
            AuxiliaryHeadWithDecoderWithoutInstantiatedHead] = []

        for aux_decoder in aux_decoders:
            args = aux_decoder.decoder_args if aux_decoder.decoder_args else {}
            aux_decoder_cls: nn.Module = _get_decoder(aux_decoder.decoder)

            aux_decoder_kwargs, kwargs = extract_prefix_keys(args, "decoder_")
            aux_decoder_instance = aux_decoder_cls(
                backbone.feature_info.channels(), **aux_decoder_kwargs)
            # aux_decoder_instance = aux_decoder_cls([128, 256, 512, 1024], **decoder_kwargs)

            aux_head_kwargs, kwargs = extract_prefix_keys(args, "head_")
            if num_classes:
                aux_head_kwargs["num_classes"] = num_classes
            # aux_head: nn.Module = _get_head(task, aux_decoder_instance, num_classes=num_classes, **head_kwargs)
            # aux_decoder.decoder = nn.Sequential(aux_decoder_instance, aux_head)
            to_be_aux_decoders.append(
                AuxiliaryHeadWithDecoderWithoutInstantiatedHead(
                    aux_decoder.name, aux_decoder_instance, aux_head_kwargs))

        return _build_appropriate_model(
            task,
            encoder,
            decoder,
            head_kwargs,
            prepare_features_for_image_model,
            patch_size,
            padding,
            rescale,  # noqa: FBT001, FBT002
            to_be_aux_decoders,
        )


def _get_decoder(decoder: str | nn.Module) -> nn.Module:
    if isinstance(decoder, nn.Module):
        return decoder
    if isinstance(decoder, str):
        try:
            decoder = getattr(decoder_registry, decoder)
            return decoder
        except AttributeError as decoder_not_found_exception:
            msg = f"Decoder {decoder} was not found in the registry."
            raise DecoderNotFoundError(msg) from decoder_not_found_exception
    msg = "Decoder must be str or nn.Module"
    raise Exception(msg)


def _build_appropriate_model(
    task: str,
    backbone: nn.Module,
    decoder: nn.Module,
    head_kwargs: dict,
    prepare_features_for_image_model: Callable,
    patch_size: int | list | None,
    padding: str,
    rescale: bool = True,  # noqa: FBT001, FBT002
    auxiliary_heads: dict | None = None,
):
    if task in PIXEL_WISE_TASKS:
        return PixelWiseModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            patch_size=patch_size,
            padding=padding,
            rescale=rescale,
            auxiliary_heads=auxiliary_heads,
        )
    elif task in SCALAR_TASKS:
        return ScalarOutputModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            patch_size=patch_size,
            padding=padding,
            auxiliary_heads=auxiliary_heads,
        )
