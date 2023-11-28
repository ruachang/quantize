# parser onnx to quantized model by calibrating the data 

import numpy as np
import tensorrt as trt 
import pycuda.driver as cuda 
from PIL import Image

import os 
import pandas as pd
import torchvision
import common

txt_path = "/home/developers/liuchang/FER数据集/Oulu-CASIA/NI200cropped/10_fold_txt".format(0)
img_path = "/home/developers/liuchang/FER数据集"
onnx_path = "/home/developers/liuchang/quantize/py-quant-ptq/ori_mfcnn.onnx"
engine_path = "/home/developers/liuchang/quantize/py-quant-ptq/fp16/fp16_mfcnn2.trt"
TRT_LOGGER = trt.Logger()

def load_calib_data(txt_path, img_dir, transforms):
    dp = pd.read_csv(txt_path, delimiter=" ")
    raw, _ = dp.shape 
    img_path_body = dp.iloc[0, 0]
    img_path = os.path.join(img_dir, img_path_body)
    img_tmp = Image.open(img_path)
    img_w, img_h = (60, 60)
    calib_data = np.zeros((raw, 1, img_w, img_h))
    
    for i in range(raw):
        img_path_body = dp.iloc[i, 0]
        img_path = os.path.join(img_dir, img_path_body)
        img_tmp = Image.open(img_path)
        img     = transforms(img_tmp)
        # img     = img_tmp.unsqueeze(0).numpy()
        img     = np.array(img, dtype=np.float32)
        calib_data[i, ...] = img    
    return calib_data    

class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, txt_path, img_dir, transforms, cache_file) -> None:
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.current_index = 0
        self.batch_size = 1
        self.data = load_calib_data(txt_path, img_dir, transforms)
        # assume that the batch size is 1
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)
    def get_batch_size(self):
        return self.batch_size
    def get_batch(self, names):
        # 如果剩下的不够一个batch用, 返回
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None
        # 确认current batch
        current_batch = int(self.current_index / self.batch_size)
        # 整十数print
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]
    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
def build_int8_engine(onnx_path, quan_model, calib, engine_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1 
        if quan_model == "fp16":
            builder.fp16_mode = True 
        elif quan_model == "int8":
            builder.int8_mode = True 
            builder.int8_calibrator = calib 
        # Parse model file 
        if not os.path.exists(onnx_path):
            print("onnx file not found")
            exit(0)
        print("Load onnx file from {}".format(onnx_path))
        with open(onnx_path, 'rb') as model:
            print('Beginning onnx parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        print("Completed parsing")
        # build engine and do int8 calibration(using calibration function before building)
        engine = builder.build_cuda_engine(network)
        print('Completed creating Engine')
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        return engine 
    
def main():
    calibration_cache = "calibrate.cache"
    transform_Fesim=torchvision.transforms.Compose([
            # transforms.ToPILImage(),
            torchvision.transforms.Resize((60, 60)),
            torchvision.transforms.Grayscale(1),
            torchvision.transforms.ToTensor(),
                                        ])
    calib = EntropyCalibrator(txt_path = os.path.join(txt_path, "1_10_fold_train.txt"), 
                              img_dir = img_path, 
                              transforms = transform_Fesim, 
                              cache_file = calibration_cache)
    build_int8_engine(onnx_path, "fp16", calib, engine_path)
    
if __name__ == '__main__':
    main()
