#!/usr/bin/env python
# coding:utf-8

import argparse
import os

import torch
from models.model import HumanDetectionModel
from utils.utils import load_yaml_config, make_directory, update_args_with_yaml


def main(args):
    model = HumanDetectionModel(args.output["num_classes"])
    model.load_state_dict(torch.load(args.model_path)["model"])
    make_directory(args.output["output_directory"])

    output_model_path = os.path.join(args.output["output_directory"], args.output["output_model"])

    if args.fp16:
        with torch.autocast("cpu", dtype=torch.float16):
            dummy_input = torch.randn((1, 3, 480, 640), dtype=torch.float32)
            torch.onnx.export(
                model,
                dummy_input,
                output_model_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=[args.input["input_blob_name"]],
                output_names=args.output["output_blob_name"],
                dynamic_axes={
                    args.input["input_blob_name"]: {0: "batch_size", 2: "height", 3: "width"},
                    args.output["output_blob_name"][0]: {0: "batch_size", 2: "height", 3: "width"},
                    args.output["output_blob_name"][1]: {0: "batch_size", 2: "height", 3: "width"},
                    args.output["output_blob_name"][2]: {0: "batch_size", 2: "height", 3: "width"},
                },
            )
    else:
        dummy_input = torch.randn((1, 3, 480, 640), dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy_input,
            output_model_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=[args.input["input_blob_name"]],
            output_names=args.output["output_blob_name"],
            dynamic_axes={
                args.input["input_blob_name"]: {0: "batch_size", 2: "height", 3: "width"},
                args.output["output_blob_name"][0]: {0: "batch_size", 2: "height", 3: "width"},
                args.output["output_blob_name"][1]: {0: "batch_size", 2: "height", 3: "width"},
                args.output["output_blob_name"][2]: {0: "batch_size", 2: "height", 3: "width"},
            },
        )

    print("Model has been converted from Pytorch to ONNX")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert from pytorch model to onnx model")
    parser.add_argument(
        "--config", type=str, default="config/torch2onnx.yml", help="path to the yaml configuration file"
    )
    parser.add_argument("--fp16", action="store_true", help="use FP16 (half-precision floating point) for export")
    parser.add_argument("--model_path", type=str, required=True, help="path to the pytorch model to be converted")
    args = parser.parse_args()

    config_yml = load_yaml_config(args.config)
    args = update_args_with_yaml(args, config_yml)

    main(args)
