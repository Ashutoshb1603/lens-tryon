"""Store a TFLite model's float32 weights as float16 or int8, with DEQUANTIZE ops in front.

The model computes exactly as before in float32: each constant weight tensor is replaced by a smaller one plus a
DEQUANTIZE op that restores it when the model is loaded (the standard "weight-only" quantization layout, run by
both the CPU (XNNPACK) and GPU delegates). Metadata (labels, normalisation) is kept as is.

usage: python3 shrink_tflite.py in.tflite out.tflite fp16|int8   (needs the flatc-generated tflite schema on PYTHONPATH)
"""
import sys, numpy as np, flatbuffers
from tflite.Model import Model, ModelT
from tflite.Tensor import TensorT
from tflite.Buffer import BufferT
from tflite.Operator import OperatorT
from tflite.OperatorCode import OperatorCodeT
from tflite.QuantizationParameters import QuantizationParametersT
from tflite.TensorType import TensorType
from tflite.BuiltinOperator import BuiltinOperator

src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
m = ModelT.InitFromObj(Model.GetRootAsModel(open(src, "rb").read(), 0))
g = m.subgraphs[0]

deq = next((i for i, c in enumerate(m.operatorCodes) if max(c.builtinCode, c.deprecatedBuiltinCode) == BuiltinOperator.DEQUANTIZE), None)
if deq is None:
    c = OperatorCodeT(); c.builtinCode = BuiltinOperator.DEQUANTIZE; c.deprecatedBuiltinCode = BuiltinOperator.DEQUANTIZE; c.version = 2 if mode == "int8" else 3
    m.operatorCodes.append(c); deq = len(m.operatorCodes) - 1

# which tensors are weights of which op (int8 per-channel needs to know the output-channel axis)
axis_of = {}
for op in g.operators:
    code = max(m.operatorCodes[op.opcodeIndex].builtinCode, m.operatorCodes[op.opcodeIndex].deprecatedBuiltinCode)
    if len(op.inputs) > 1:
        w = op.inputs[1]
        if code == BuiltinOperator.DEPTHWISE_CONV_2D: axis_of[w] = 3
        elif code in (BuiltinOperator.CONV_2D, BuiltinOperator.FULLY_CONNECTED): axis_of[w] = 0
    if code == BuiltinOperator.TRANSPOSE_CONV and len(op.inputs) > 1: axis_of[op.inputs[1]] = 0

new_ops, before, after = [], 0, 0
for ti, t in enumerate(list(g.tensors)):
    buf = m.buffers[t.buffer]
    if t.type != TensorType.FLOAT32 or buf.data is None or len(buf.data) < 1024: continue
    a = np.frombuffer(bytes(bytearray(buf.data)), "<f4").reshape(t.shape)
    q = TensorT(); q.shape = list(t.shape); q.name = (t.name or b"") + b"_" + mode.encode()
    nb = BufferT()
    if mode == "fp16" or ti not in axis_of:                    # biases and other constants: float16 either way
        q.type = TensorType.FLOAT16; nb.data = np.frombuffer(a.astype("<f2").tobytes(), np.uint8)
    else:                                                        # int8, symmetric, one scale per output channel
        ax = axis_of[ti]; mx = np.abs(np.moveaxis(a, ax, 0).reshape(a.shape[ax], -1)).max(1); sc = np.maximum(mx, 1e-12) / 127
        shp = [1] * a.ndim; shp[ax] = -1
        qa = np.clip(np.round(a / sc.reshape(shp)), -127, 127).astype(np.int8)
        q.type = TensorType.INT8; nb.data = np.frombuffer(qa.tobytes(), np.uint8)
        qp = QuantizationParametersT(); qp.scale = sc.astype(np.float32).tolist(); qp.zeroPoint = [0] * len(sc); qp.quantizedDimension = ax
        q.quantization = qp
    m.buffers.append(nb); q.buffer = len(m.buffers) - 1
    g.tensors.append(q); qi = len(g.tensors) - 1
    t.buffer = 0                                                 # the float32 tensor is now made by the DEQUANTIZE op
    op = OperatorT(); op.opcodeIndex = deq; op.inputs = [qi]; op.outputs = [ti]; new_ops.append(op)
    before += len(buf.data); after += len(nb.data); m.buffers[m.buffers.index(buf)] = BufferT()   # drop the float32 bytes
g.operators = new_ops + g.operators

b = flatbuffers.Builder(1 << 20); b.Finish(m.Pack(b), file_identifier=b"TFL3")
open(dst, "wb").write(b.Output())
# associated files (labels.txt and the like) live in a zip appended after the flatbuffer: carry them over
import zipfile, io
try:
    zin = zipfile.ZipFile(io.BytesIO(open(src, "rb").read()))
    with zipfile.ZipFile(dst, "a", zipfile.ZIP_STORED) as zout:
        for i in zin.infolist(): zout.writestr(i.filename, zin.read(i.filename))
    print("kept associated files:", zin.namelist())
except zipfile.BadZipFile: pass
import os; print(f"{len(new_ops)} weight tensors: {before/1e6:.1f} MB -> {after/1e6:.1f} MB; file {os.path.getsize(src)/1e6:.1f} -> {os.path.getsize(dst)/1e6:.1f} MB")
