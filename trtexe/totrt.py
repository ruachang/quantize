# parser onnx to tensorrt directly using python

import numpy as np
import tensorrt as trt 
import pycuda.driver as cuda 
import PIL 
import os 

import common

TRT_LOGGER = trt.Logger()

def build_engine(onnx_file_path, engine_file_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1 
        # Parse model file 
        if not os.path.exists(onnx_file_path):
            print("onnx file not found")
            exit(0)
        print("Load onnx file from {}".format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning onnx parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        print("Completed parsing")
        engine = builder.build_cuda_engine(network)
        print('Completed creating Engine')
        with open(engine_file_path, 'wb') as f:
            f.write(engine.serialize())
        return engine 

build_engine("/home/developers/liuchang/quantize/py-quant-ptq/ori_mfcnn.onnx",
             "/home/developers/liuchang/quantize/py-quant-ptq/ori_mfcnn.trt")