from sqlalchemy import outparam
import torch 
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

import itertools
import numpy as np
import onnx
import packaging.version as pv
import warnings
from onnx import helper, numpy_helper
from onnx import onnx_pb as onnx_proto
import onnx_graphsurgeon as gs

def _npfloat16_to_int(np_list):
    '''
    Convert numpy float16 to python int.

    :param np_list: numpy float16 list
    :return int_list: python int list
    '''
    return [int(bin(_.view('H'))[2:].zfill(16), 2) for _ in np_list]


def convert_np_to_float16(np_array, min_positive_val=1e-7, max_finite_val=1e4):
    '''
    Convert float32 numpy array to float16 without changing sign or finiteness.
    Positive values less than min_positive_val are mapped to min_positive_val.
    Positive finite values greater than max_finite_val are mapped to max_finite_val.
    Similar for negative values. NaN, 0, inf, and -inf are unchanged.
    '''
    def between(a, b, c):
        return np.logical_and(a < b, b < c)

    if (np_array[np.where(np_array > 0)].shape[0] > 0):
        pos_max = np_array[np.where(np_array > 0)].max()
        pos_min = np_array[np.where(np_array > 0)].min()

        if (pos_max >= max_finite_val):
            warnings.warn("the float32 number {} will be truncated to {}".format(pos_max, max_finite_val))

        if (pos_min <= min_positive_val):
            warnings.warn("the float32 number {} will be truncated to {}".format(pos_min, min_positive_val))

    if (np_array[np.where(np_array < 0)].shape[0] > 0):
        neg_max = np_array[np.where(np_array < 0)].max()
        neg_min = np_array[np.where(np_array < 0)].min()

        if (neg_min <= -max_finite_val):
            warnings.warn("the float32 number {} will be truncated to {}".format(neg_min, -max_finite_val))

        if (neg_max >= -min_positive_val):
            warnings.warn("the float32 number {} will be truncated to {}".format(neg_max, -min_positive_val))

    np_array = np.where(between(0, np_array, min_positive_val), min_positive_val, np_array)
    np_array = np.where(between(-min_positive_val, np_array, 0), -min_positive_val, np_array)
    np_array = np.where(between(max_finite_val, np_array, float('inf')), max_finite_val, np_array)
    np_array = np.where(between(float('-inf'), np_array, -max_finite_val), -max_finite_val, np_array)
    return np.float16(np_array)


def convert_tensor_float_to_float16(tensor, min_positive_val=1e-7, max_finite_val=1e4):
    '''
    Convert tensor float to float16.

    :param tensor: TensorProto object
    :return tensor_float16: converted TensorProto object

    Example:

    ::

        from onnxmltools.utils.float16_converter import convert_tensor_float_to_float16
        new_tensor = convert_tensor_float_to_float16(tensor)

    '''
    if not isinstance(tensor, onnx_proto.TensorProto):
        raise ValueError('Expected input type is an ONNX TensorProto but got %s' % type(tensor))

    if tensor.data_type == onnx_proto.TensorProto.FLOAT:
        tensor.data_type = onnx_proto.TensorProto.FLOAT16
        # convert float_data (float type) to float16 and write to int32_data
        if tensor.float_data:
            float16_data = convert_np_to_float16(np.array(tensor.float_data), min_positive_val, max_finite_val)
            int_list = _npfloat16_to_int(float16_data)
            tensor.int32_data[:] = int_list
            tensor.float_data[:] = []
        # convert raw_data (bytes type)
        if tensor.raw_data:
            # convert n.raw_data to float
            float32_list = np.fromstring(tensor.raw_data, dtype='float32')
            # convert float to float16
            float16_list = convert_np_to_float16(float32_list, min_positive_val, max_finite_val)
            # convert float16 to bytes and write back to raw_data
            tensor.raw_data = float16_list.tostring()
    return tensor


def make_value_info_from_tensor(tensor):
    shape = numpy_helper.to_array(tensor).shape
    return helper.make_tensor_value_info(tensor.name, tensor.data_type, shape)


DEFAULT_OP_BLOCK_LIST = ['ArrayFeatureExtractor', 'Binarizer', 'CastMap', 'CategoryMapper', 'DictVectorizer',
                         'FeatureVectorizer', 'Imputer', 'LabelEncoder', 'LinearClassifier', 'LinearRegressor',
                         'Normalizer', 'OneHotEncoder', 'RandomUniformLike', 'SVMClassifier', 'SVMRegressor', 'Scaler',
                         'TreeEnsembleClassifier', 'TreeEnsembleRegressor', 'ZipMap', 'NonMaxSuppression', 'TopK',
                         'RoiAlign', 'Resize', 'Range', 'CumSum', 'Min', 'Max', 'Upsample', 'QuantizeLinear', 'DequantizeLinear']

DEFAULT_NODE_ALLOW_LIST = ['Concat_118', 'Concat_127']
DEFAULT_START_NODE = 'Concat_118'
DEFAULT_START_IN = ['233', '196']
# DEFAULT_END_NODE = 'Concat_127'
DEFAULT_END_NODE = 'Mul_124'
DEFAULT_END_OUT = '240'

# for channel_qantization only quantize spatial attention module
DEFAULT_START_NODES = ['Concat_112', 'MaxPool_2916', 'MaxPool_2742', 'MaxPool_2568', 
                       'MaxPool_2394', 'MaxPool_2220', 'MaxPool_2046', 'MaxPool_1872',
                       'MaxPool_1698', 'MaxPool_1524', 'MaxPool_1350', 'MaxPool_1176', 
                       'MaxPool_1002', 'MaxPool_828', 'MaxPool_654', 'MaxPool_480', 'MaxPool_307']
DEFAULT_START_INS = [['226', '195'], ['3030'], ['2856'], ['2682'], 
                     ['2508'], ['2334'], ['2160'], ['1986'],
                     ['1812'], ['1638'], ['1464'], ['1290'], 
                     ['1116'], ['942'], ['768'], ['594'], ['421']]
DEFAULT_END_NODES = ['Mul_118', 'Mul_2922', 'Mul_2748', 'Mul_2574', 
                     'Mul_2400', 'Mul_2226', 'Mul_2052', 'Mul_1878',
                     'Mul_1704', 'Mul_1530', 'Mul_1356', 'Mul_1182', 
                     'Mul_1008', 'Mul_834', 'Mul_660', 'Mul_486', 'Mul_313']
DEFAULT_END_OUTS = ['233', '3037', '2863', '2689', 
                    '2515', '2341', '2167', '1993',
                    '1819', '1645', '1471', '1297', 
                    '1123', '949', '775', '601', '428']

# for maxpool, quantize all attention module
DEFAULT_START_NODES = ['Concat_118', 'MaxPool_1674', 'MaxPool_1578', 'MaxPool_1482', 
                       'MaxPool_1386', 'MaxPool_1290', 'MaxPool_1194', 'MaxPool_1098',
                       'MaxPool_1002', 'MaxPool_906', 'MaxPool_810', 'MaxPool_714', 
                       'MaxPool_618', 'MaxPool_522', 'MaxPool_426', 'MaxPool_330', 'MaxPool_235']
DEFAULT_START_INS = [['222', '185'], ['1778'], ['1682'], ['1586'], 
                     ['1490'], ['1394'], ['1298'], ['1202'],
                     ['1106'], ['1010'], ['914'], ['818'], 
                     ['722'], ['626'], ['530'], ['434'], ['339']]
DEFAULT_END_NODES = ['Mul_135', 'Mul_1691', 'Mul_1595', 'Mul_1499', 
                     'Mul_1403', 'Mul_1307', 'Mul_1211', 'Mul_1115',
                     'Mul_1019', 'Mul_923', 'Mul_827', 'Mul_731', 
                     'Mul_635', 'Mul_539', 'Mul_443', 'Mul_347', 'Mul_252']
DEFAULT_END_OUTS = ['240', '1796', '1700', '1604', 
                    '1508', '1412', '1316', '1220',
                    '1124', '1028', '932', '836', 
                    '740', '644', '548', '452', '357']





def insert_float16(model, min_positive_val=1e-7, max_finite_val=1e4,
                              disable_shape_infer=False,
                             start_node=None, end_node=None, start_in=None, end_out=None):
    '''
    Insert cast to specific nodes to convert nodes into float16
    
    :param model: ONNX ModelProto object
    :param disable_shape_infer: Type/shape information is needed for conversion to work.
                                Set to True only if the model already has type/shape information for all tensors.
    :return: converted ONNX ModelProto object
    '''
    if start_node == None:
        start_node = DEFAULT_START_NODE 
    if start_in == None:
        start_in = DEFAULT_START_IN
    if end_node == None:
        end_node = DEFAULT_END_NODE
    if end_out == None:
        end_out = DEFAULT_END_OUT 
    
    func_infer_shape = None
    if not disable_shape_infer and pv.Version(onnx.__version__) >= pv.Version('1.2'):
        try:
            from onnx.shape_inference import infer_shapes
            func_infer_shape = infer_shapes
        finally:
            pass
    
    if not isinstance(model, onnx_proto.ModelProto):
        raise ValueError('Expected model type is an ONNX ModelProto but got %s' % type(model))
    # type inference on input model
    if func_infer_shape is not None:
        model = func_infer_shape(model)
    
    node_list = []
    # all tensor in the graph
    out_list = []
    value_info_list = []
    # 由于不是嵌套的链表结构, 而是分立的字典, 因此不使用递归的求法
    for node in model.graph.node:
        # start node
        if node.name == start_node:
            node_list.append(node)
            for i in node.output:
                print(type(i), i)
                if i not in out_list: # if i not in current out list, add i
                    out_list.append(i)
                else:
                    continue
        else:
            flag = False 
            if len(node.input) > 0:
                # if any of the input of the nodes in list, it should be calculated as fp16
                for input in node.input:
                    # 真的是 出此下策
                    if input in out_list and "weight" not in input:
                        flag = True
                # 封闭的图应该没有这种情况, 但是对于多输入的节点, 就已经不是封闭的了, 不能注释
                for i in node.input:
                    if i not in out_list and flag:
                        out_list.append(i)
                    else:
                        continue
            if len(node.output) > 0:
                for output in node.output:
                    if output in out_list:
                        flag = True
                for i in node.output:
                    if i == end_out:
                        continue
                    if i not in out_list and flag:
                        out_list.append(i)
                    else:
                        continue
            if flag:
                node_list.append(node)
    out_list.append(end_out)
    out_list += start_in
    
    for q in model.graph.initializer:
        if q.data_type == onnx_proto.TensorProto.FLOAT and q.name in out_list:
            q = convert_tensor_float_to_float16(q, min_positive_val, max_finite_val)
            value_info_list.append(make_value_info_from_tensor(q))
    for q in model.graph.value_info:
        if q.name in out_list and q.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
            if q.name not in start_in and q.name != end_out:
                q.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
            value_info_list.append(q)
    for node in node_list:
        # for n in node.initializer:
        #     if n.data_type == onnx_proto.TensorProto.FLOAT:
        #         n = convert_tensor_float_to_float16(n, min_positive_val, max_finite_val)
        for attr in node.attribute:
            attr.t.CopyFrom(convert_tensor_float_to_float16(attr.t, min_positive_val, max_finite_val))  
            for n in attr.tensors:
                n = convert_tensor_float_to_float16(n, min_positive_val, max_finite_val)
    
    for i in node_list:
        # insert cast before start node to change fp32 => fp16, i.e.: after every input of start node
        if i.name == start_node:
            for j in range(len(i.input)):
                input = i.input[j]
                for value_info in value_info_list:
                    if input == value_info.name:
                        new_value_info = model.graph.value_info.add()
                        new_value_info.CopyFrom(value_info)
                        node_name = i.name + "_input_cast_" + str(j)
                        output_name = i.name + "_input_cast_" + str(j)
                        new_value_info.name = output_name
                        new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                        new_node = [helper.make_node('Cast', [input], [output_name], to=10, name=node_name)]
                        model.graph.node.extend(new_node)
                        i.input[j] = output_name
        if i.name == end_node:
            for j in range(len(i.output)):
                output = i.output[j]
                for value_info in value_info_list:
                    if output == value_info.name:
                        new_value_info = model.graph.value_info.add()
                        new_value_info.CopyFrom(value_info)
                        node_name = i.name + "_output_cast_" + str(j)
                        input_name = i.name + "_output_cast_" + str(j)
                        new_value_info.name = input_name
                        new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                        new_node = [helper.make_node('Cast', [input_name], [output], to=1, name=node_name)]
                        i.output[j] = input_name
                        model.graph.node.extend(new_node)
            
    return model 
    # graph = gs.import_onnx(model)
    # # weights and activations
    # tensors = graph.tensors()
    # graph.inputs = [tensors[DEFAULT_START_NODE1].to_variable(dtype=np.float32),
    #                 tensors[DEFAULT_START_NODE2].to_variable(dtype=np.float32)]
    # graph.outputs = [tensors[DEFAULT_END_NODE].to_variable(dtype=np.float32)]
    # graph.cleanup()
    # sub_model = gs.export_onnx(graph)
    # onnx.save(sub_model, "replaced.onnx")
    # sub_model = convert_float_to_float16(sub_model, keep_io_types=True)
    
def convert_float_to_float16(model, min_positive_val=1e-7, max_finite_val=1e4,
                             keep_io_types=False, disable_shape_infer=False,
                             op_block_list=None, node_block_list=None):
    '''
    Convert tensor float type in the ONNX ModelProto input to tensor float16.
    All node and params will be changed to fp16 expect the block nodes. Two cast nodes will be
    insert before the input and after the output of the block nodes, so for model that is allowed
    to change into fp16 of specific nodes, this method is not fittable.
    :param model: ONNX ModelProto object
    :param disable_shape_infer: Type/shape information is needed for conversion to work.
                                Set to True only if the model already has type/shape information for all tensors.
    :return: converted ONNX ModelProto object

    Examples:

    ::

        Example 1: Convert ONNX ModelProto object:
        from onnxmltools.utils.float16_converter import convert_float_to_float16
        new_onnx_model = convert_float_to_float16(onnx_model)

        Example 2: Convert ONNX model binary file:
        from onnxmltools.utils.float16_converter import convert_float_to_float16
        from onnxmltools.utils import load_model, save_model
        onnx_model = load_model('model.onnx')
        new_onnx_model = convert_float_to_float16(onnx_model)
        save_model(new_onnx_model, 'new_model.onnx')

    '''
    func_infer_shape = None
    if not disable_shape_infer and pv.Version(onnx.__version__) >= pv.Version('1.2'):
        try:
            from onnx.shape_inference import infer_shapes
            func_infer_shape = infer_shapes
        finally:
            pass

    if not isinstance(model, onnx_proto.ModelProto):
        raise ValueError('Expected model type is an ONNX ModelProto but got %s' % type(model))

    # create blocklists
    # op_allow_list = None 
    # node_allow_list = None
    if op_block_list is None:
        op_block_list = DEFAULT_OP_BLOCK_LIST
        # op_allow_list = DEFAULT_OP_ALLOW_LIST
    if node_block_list is None:
        node_block_list = []
        # node_allow_list = DEFAULT_NODE_ALLOW_LIST
    op_block_list = set(op_block_list)
    node_block_list = set(node_block_list)
    # op_allow_list = set(op_allow_list)
    # node_allow_list = set(node_allow_list)
    # create a queue for BFS
    queue = []
    value_info_list = []
    node_list = []
    # type inference on input model
    if func_infer_shape is not None:
        model = func_infer_shape(model)
    queue.append(model)
    name_mapping = {}
    graph_io_to_skip = set()
    io_casts = set()
    if keep_io_types:
        for i, n in enumerate(model.graph.input):
            if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                output_name = 'graph_input_cast_' + str(i)
                name_mapping[n.name] = output_name
                graph_io_to_skip.add(n.name)

                node_name = 'graph_input_cast' + str(i)
                new_value_info = model.graph.value_info.add()
                new_value_info.CopyFrom(n)
                new_value_info.name = output_name
                new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                # add Cast node (from tensor(float) to tensor(float16) after graph input
                new_node = [helper.make_node('Cast', [n.name], [output_name], to=10, name=node_name)]
                model.graph.node.extend(new_node)
                value_info_list.append(new_value_info)
                io_casts.add(node_name)

        for i, n in enumerate(model.graph.output):
            if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                input_name = 'graph_output_cast_' + str(i)
                name_mapping[n.name] = input_name
                graph_io_to_skip.add(n.name)

                node_name = 'graph_output_cast' + str(i)
                # add Cast node (from tensor(float16) to tensor(float) before graph output
                new_value_info = model.graph.value_info.add()
                new_value_info.CopyFrom(n)
                new_value_info.name = input_name
                new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                new_node = [helper.make_node('Cast', [input_name], [n.name], to=1, name=node_name)]
                model.graph.node.extend(new_node)
                value_info_list.append(new_value_info)
                io_casts.add(node_name)

    while queue:
        next_level = []
        for q in queue:
            # if q is model, push q.graph (GraphProto)
            if isinstance(q, onnx_proto.ModelProto):
                next_level.append(q.graph)
            # if q is model.graph, push q.node.attribute (AttributeProto)
            if isinstance(q, onnx_proto.GraphProto):
                for n in q.node:
                    # if n is in the block list (doesn't support float16), no conversion for the node,
                    # and save the node for further processing
                    if n.name in io_casts:
                        continue
                    for i in range(len(n.input)):
                        if n.input[i] in name_mapping:
                            n.input[i] = name_mapping[n.input[i]]
                    for i in range(len(n.output)):
                        if n.output[i] in name_mapping:
                            n.output[i] = name_mapping[n.output[i]]
                    # don't add the attr into next_level for the node in node_keep_data_type_list
                    # so it will not be converted to float16
                    if n.op_type in op_block_list or n.name in node_block_list:
                    # if n.op_type not in op_allow_list and n.name not in node_allow_list:
                        node_list.append(n)
                    else:
                        if n.op_type == 'Cast':
                            for attr in n.attribute:
                                if attr.name == 'to' and attr.i == 1:
                                    attr.i = 10
                                    break
                        for attr in n.attribute:
                            next_level.append(attr)
            # if q is model.graph.node.attribute, push q.g and q.graphs (GraphProto)
            # and process node.attribute.t and node.attribute.tensors (TensorProto)
            if isinstance(q, onnx_proto.AttributeProto):
                next_level.append(q.g)
                for n in q.graphs:
                    next_level.append(n)
                q.t.CopyFrom(convert_tensor_float_to_float16(q.t, min_positive_val, max_finite_val))
                for n in q.tensors:
                    n = convert_tensor_float_to_float16(n, min_positive_val, max_finite_val)
            # if q is graph, process graph.initializer(TensorProto), input, output and value_info (ValueInfoProto)
            if isinstance(q, onnx_proto.GraphProto):
                for n in q.initializer:  # TensorProto type: i.e for all parameters
                    if n.data_type == onnx_proto.TensorProto.FLOAT:
                        n = convert_tensor_float_to_float16(n, min_positive_val, max_finite_val)
                        value_info_list.append(make_value_info_from_tensor(n))
                # for all ValueInfoProto with tensor(float) type in input, output and value_info, convert them to
                # tensor(float16) except map and seq(map). And save them in value_info_list for further processing
                for n in itertools.chain(q.input, q.output, q.value_info):
                    if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                        if n.name not in graph_io_to_skip:
                            n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                            value_info_list.append(n)
        queue = next_level

    # process the nodes in block list that doesn't support tensor(float16)
    for node in node_list:
        # if input's name is in the value_info_list meaning input is tensor(float16) type,
        # insert a float16 to float Cast node before the node,
        # change current node's input name and create new value_info for the new name
        if "Resize" in node.name:
            DEBUG_FLAG = True
        for i in range(len(node.input)):
            input = node.input[i]
            for value_info in value_info_list :
                if input == value_info.name: #and input != '94' and input != '93':
                    # create new value_info for current node's new input name
                    new_value_info = model.graph.value_info.add()
                    new_value_info.CopyFrom(value_info)
                    output_name = node.name + '_input_cast_' + str(i)
                    new_value_info.name = output_name
                    new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
                    # add Cast node (from tensor(float16) to tensor(float) before current node
                    node_name = node.name + '_input_cast' + str(i)
                    new_node = [helper.make_node('Cast', [input], [output_name], to=1, name=node_name)]
                    model.graph.node.extend(new_node)
                    # change current node's input name
                    node.input[i] = output_name
                    break
        # if output's name is in the value_info_list meaning output is tensor(float16) type, insert a float to
        # float16 Cast node after the node, change current node's output name and create new value_info for the new name
        for i in range(len(node.output)):
            output = node.output[i]
            for value_info in value_info_list:
                if output == value_info.name:
                    # create new value_info for current node's new output
                    new_value_info = model.graph.value_info.add()
                    new_value_info.CopyFrom(value_info)
                    input_name = node.name + '_output_cast_' + str(i)
                    new_value_info.name = input_name
                    new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
                    # add Cast node (from tensor(float) to tensor(float16) after current node
                    node_name = node.name + '_output_cast' + str(i)
                    new_node = [helper.make_node('Cast', [input_name], [output], to=10, name=node_name)]
                    model.graph.node.extend(new_node)
                    # change current node's input name
                    node.output[i] = input_name
                    break

    return model


def convert_float_to_float16_model_path(model_path, min_positive_val=1e-7, max_finite_val=1e4, keep_io_types=False):
    '''
    Convert tensor float type in the ONNX Model to tensor float16.
    *It is to fix an issue that infer_shapes func cannot be used to infer >2GB models.
    *But this function can be applied to all model sizes.
    :param model_path: ONNX Model path
    :return: converted ONNX ModelProto object
    Examples
    ::
        #Convert to ONNX ModelProto object and save model binary file:
        from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path
        new_onnx_model = convert_float_to_float16_model_path('model.onnx')
        onnx.save(new_onnx_model, 'new_model.onnx')
    '''

    disable_shape_infer = False
    if pv.Version(onnx.__version__) >= pv.Version('1.8'):
        try:
            # infer_shapes_path can be applied to all model sizes
            from onnx.shape_inference import infer_shapes_path
            import tempfile
            import os
            # shape_infer_model_path should be in the same folder of model_path
            with tempfile.NamedTemporaryFile(dir=os.path.dirname(model_path)) as tmpfile:
                shape_infer_model_path = tmpfile.name
                infer_shapes_path(model_path, shape_infer_model_path)
                model = onnx.load(shape_infer_model_path)
                disable_shape_infer = True
        finally:
            pass
    if not disable_shape_infer:
        model = onnx.load(model_path)
    return convert_float_to_float16(model, min_positive_val, max_finite_val, keep_io_types, disable_shape_infer)
    # return insert_float16(model, min_positive_val, max_finite_val, keep_io_types, disable_shape_infer)

def insert_nodes(model_path, min_positive_val=1e-7, max_finite_val=1e4, start_nodes=None, end_nodes=None,
                 start_ins=None, end_outs=None):
    disable_shape_infer = False
    if pv.Version(onnx.__version__) >= pv.Version('1.8'):
        try:
            # infer_shapes_path can be applied to all model sizes
            from onnx.shape_inference import infer_shapes_path
            import tempfile
            import os
            # shape_infer_model_path should be in the same folder of model_path
            with tempfile.NamedTemporaryFile(dir=os.path.dirname(model_path)) as tmpfile:
                shape_infer_model_path = tmpfile.name
                infer_shapes_path(model_path, shape_infer_model_path)
                model = onnx.load(shape_infer_model_path)
                disable_shape_infer = True
        finally:
            pass
    if start_nodes == None:
        start_nodes = DEFAULT_START_NODES 
    if start_ins == None:
        start_ins = DEFAULT_START_INS
    if end_nodes == None:
        end_nodes = DEFAULT_END_NODES
    if end_outs == None:
        end_outs = DEFAULT_END_OUTS 
    if not disable_shape_infer:
        model = onnx.load(model_path)
    for i in range(len(start_nodes)):
        model = insert_float16(model, min_positive_val, max_finite_val, disable_shape_infer,
                               start_node=start_nodes[i], end_node=end_nodes[i],
                               start_in=start_ins[i], end_out=end_outs[i])
    # return convert_float_to_float16(model, min_positive_val, max_finite_val, keep_io_types, disable_shape_infer)
    return model





def network_to_half(model):
    """
    Convert model to half precision in a batchnorm-safe way. change from framework PyTorch
    """
    def bn_to_float(module):
        """
        BatchNorm layers need parameters in single precision. Find all layers and convert
        them back to float.
        """
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.float()
        for child in module.children():
            bn_to_float(child)
        return module
    return bn_to_float(model.half())