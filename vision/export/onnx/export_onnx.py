#!/usr/bin/env python
import argparse
import datetime
import os
import time
import warnings

import onnx
import onnxruntime as ort
import torch
import torch.nn.parallel
import torch.onnx
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import yaml
from easydict import EasyDict
from onnxruntime import SessionOptions, GraphOptimizationLevel, ExecutionMode
from torch.onnx import TrainingMode

from vision.models.seg_models_pytorch import SegModelPytorchClf
from vision.models.helpers import load_checkpoint
from vision.utils.general.custom_yaml import init_custom_yaml
from vision.utils.general.modifier import dict_modifier


def convert_onnx(model, dummy_input, onnx_path_fp32):
    model.eval()

    # export
    dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}
    # dynamic_axes = dict()
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      onnx_path_fp32,  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      training=TrainingMode.EVAL, opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes=dynamic_axes,  # variable length axes
                      keep_initializers_as_inputs=False,  # enable_onnx_checker=True,
                      # operator_export_type=torch.onnx.OperatorExportTypes.ONNX,   # ONNX ONNX_ATEN_FALLBACK
                      export_modules_as_functions=False, verbose=False)

    # Write human-readable graph representation to file as well.
    model_onnx = onnx.load(onnx_path_fp32)
    onnx.checker.check_model(model_onnx)
    pgraph = onnx.helper.printable_graph(model_onnx.graph)
    open(onnx_path_fp32 + ".readable", 'w').write(pgraph)

    # Save model with some metadata.
    meta = model_onnx.metadata_props.add()
    meta.key = "creation_date"
    meta.value = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    meta = model_onnx.metadata_props.add()
    meta.key, meta.value = "input_shape", str(dummy_input.shape)

    # Save model with metadata.
    onnx.save(model_onnx, onnx_path_fp32)


def export_and_test(args, n_rep=1, turn_on_warn=False):
    if not turn_on_warn:
        ort.set_default_logger_severity(3)
        warnings.simplefilter('ignore', Warning)

    # **************************************************************************************************************** #
    # Define ONNX session
    # https://fs-eire.github.io/onnxruntime/docs/performance/tune-performance.html
    # More CPU does not necessarily improve performance: https://github.com/Microsoft/onnxruntime/issues/897
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_options.enable_profiling = True
    # sess_options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
    sess_options.execution_mode = ExecutionMode.ORT_PARALLEL
    # sess_options.intra_op_num_threads = 0
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 2
    provider = 'CPUExecutionProvider'

    # **************************************************************************************************************** #
    # Model building
    init_custom_yaml()
    cfg_fn = "/home/dagnino/MyTmp/Vision/cfg/infer_clf.yaml"
    cfg = yaml.load(open(cfg_fn), Loader=yaml.Loader)
    cfg = dict_modifier(config=cfg, modifiers="modifiers",
                        pre_modifiers={"HOME": os.path.expanduser("~")})
    cfg = EasyDict(cfg)
    model = SegModelPytorchClf(cfg.engine.model)

    _, _, _, _ = load_checkpoint(model, cfg.engine.model.resume.load_model_fn, None, None, None, torch.device('cpu'),
                                 False, False, True)
    model.eval()
    if args.cuda:
        model.cuda()

    # **************************************************************************************************************** #
    dummy_input =
    dummy_input = dummy_input.reshape(1, -1)

    # Compute output with PyTorch
    start_time = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float32):
        for _ in range(n_rep):
            if args.cuda:
                dummy_input = torch.tensor(dummy_input).cuda()
            out_pytorch = model(dummy_input.clone())
    time_elapse = 1000 * (time.perf_counter() - start_time) / n_rep
    print(f"PyTorch model: time_elapse={time_elapse}ms")

    out_pytorch = out_pytorch.cpu().numpy()[0][0]
    print(f"            out_pytorch.shape = {out_pytorch.shape}")
    print(f'            out_pytorch = {out_pytorch}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--fn', type=str,
                        default = "/home/dagnino/MyTmp/Vision/cfg/infer_clf.yaml",
                        help='Image file')
    parser.add_argument('--cuda', action='store_true', help='Use cuda')
    _args = parser.parse_args()
    export_and_test(_args)
