
#os.system("python -m tf2onnx.convert --checkpoint ./ckpt/yolov2-voc-15.meta --output tfmodel.onnx --inputs input:0 --outputs output:0")

#python -m tf2onnx.convert --input built_graph/yolov2-voc_old.pb --inputs input:0[1,416,416,3] --outputs output:0 --output tensorflow-yolov3/model.onnx --verbose --opset 11


