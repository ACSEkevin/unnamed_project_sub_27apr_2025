# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build as build_detr
from .associatr import build as build_associatr


def build_model(args):
    if args.meta_arch == "detr":
        return build_detr(args)
    return build_associatr(args)
