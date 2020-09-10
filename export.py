
import argparse
import os

import torch
import posenet

def valid_tensor(s):
    msg = "Not a valid resolution: '{0}' [CxHxW].".format(s)
    try:
        q = s.split('x')
        if len(q) != 3:
            raise argparse.ArgumentTypeError(msg)
        return [int(v) for v in q]
    except ValueError:
        raise argparse.ArgumentTypeError(msg)

def parse_args():
    parser = argparse.ArgumentParser(description='Posenet exporter')

    parser.add_argument('-m','--model', type=int, default=101) # integer depth multiplier (50, 75, 100, 101)

    parser.add_argument('-r', '--ONNX_resolution', default="3x480x640", type=valid_tensor,
                    help='ONNX input resolution')
    parser.add_argument('-o', '--outfile', default='./out.onnx',
                    help='output file path')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    model = posenet.load_model(args.model)
    
    # Export ONNX file
    input_names = [ "input:0" ]  # this are our standardized in/out nameing (required for runtime)
    output_names = [ "output:0" ]
    dummy_input = torch.randn([1]+args.ONNX_resolution)
    ONNX_path = args.outfile
    # Exporting -- CAFFE2 compatible
    # requires operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    # https://github.com/pytorch/pytorch/issues/41848
    # for CAFFE2 backend (old exports mode...)
    #torch.onnx.export(model, dummy_input, ONNX_path, input_names=input_names, output_names=output_names, 
    #    keep_initializers_as_inputs=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    # Exporting -- ONNX runtime compatible
    #   keep_initializers_as_inputs=True -> is required for onnx optimizer...
    torch.onnx.export(model, dummy_input, ONNX_path, input_names=input_names, output_names=output_names,
        keep_initializers_as_inputs=True, opset_version=11)

if __name__ == '__main__':
    main()
