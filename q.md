使用 ddddocr 自训练模型场景下，遇到的问题

```powershell
(dddd-trainer) PS C:\Users\12460\Documents\project\dddd_trainer> uv run .\test.py
2025-11-13 00:35:09.0336981 [E:onnxruntime:, sequential_executor.cc:572 onnxruntime::ExecuteKernel] Non-zero status code returned while running SequenceAt node. Name:'n0_424' Status Message: Invalid sequence index (57) specified for sequence of size (57)
Traceback (most recent call last):
  File "C:\Users\12460\Documents\project\dddd_trainer\test.py", line 14, in <module>
    res = ocr.classification(image)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\12460\Documents\project\dddd_trainer\.venv\Lib\site-packages\ddddocr\__init__.py", line 2643, in classification
    ort_outs = self.__ort_session.run(None, ort_inputs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\12460\Documents\project\dddd_trainer\.venv\Lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 287, in run
    return self._sess.run(output_names, input_feed, run_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running SequenceAt node. Name:'n0_424' Status Message: Invalid sequence index (57) specified for sequence of size (57)

```

测试的训练集数据：
https://wwm.lanzoum.com/itczd0b5z3yj

GitHub：https://github.com/sml2h3/dddd_trainer.git

```python
# test.py
import ddddocr

ocr = ddddocr.DdddOcr(
    det=False,
    ocr=False,
    show_ad=False,
    import_onnx_path=r"C:\Users\12460\Documents\project\dddd_trainer\projects\test_2\models\test_2_1.0_23_6000_2025-11-13-00-06-13.onnx",
    charsets_path=r"C:\Users\12460\Documents\project\dddd_trainer\projects\test_2\models\charsets.json",
)

with open(r"C:\Users\12460\Downloads\1112\new\PKKQ_1578462523867.jpg", "rb") as f:
    image = f.read()

res = ocr.classification(image)
print(res)
```

操作日志

```powershell
(dddd-trainer) PS C:\Users\12460\Documents\project\dddd_trainer> uv run .\app.py create   test_3
2025-11-13 00:29:53.496 | INFO     | __main__:__init__:12 -
Hello baby~
2025-11-13 00:29:53.497 | INFO     | __main__:create:15 -
Create Project ----> test_3
2025-11-13 00:29:53.497 | INFO     | utils.project_manager:create_project:13 - Creating Directory... ----> C:\Users\12460\Documents\project\dddd_trainer\projects\test_3
2025-11-13 00:29:53.497 | INFO     | utils.project_manager:create_project:20 - Creating Directory... ----> C:\Users\12460\Documents\project\dddd_trainer\projects\test_3\models
2025-11-13 00:29:53.498 | INFO     | utils.project_manager:create_project:24 - Creating Directory... ----> C:\Users\12460\Documents\project\dddd_trainer\projects\test_3\cache
2025-11-13 00:29:53.498 | INFO     | utils.project_manager:create_project:28 - Creating Directory... ----> C:\Users\12460\Documents\project\dddd_trainer\projects\test_3\checkpoints
2025-11-13 00:29:53.498 | INFO     | utils.project_manager:create_project:32 - Creating CRNN Config File... ----> C:\Users\12460\Documents\project\dddd_trainer\projects\test_3\config.yaml
2025-11-13 00:29:53.500 | INFO     | utils.project_manager:create_project:36 - Create Project Success! ----> test_3
(dddd-trainer) PS C:\Users\12460\Documents\project\dddd_trainer> uv run .\app.py cache test_3 C:\Users\12460\Downloads\1112\images
2025-11-13 00:30:10.911 | INFO     | __main__:__init__:12 -
Hello baby~
2025-11-13 00:30:10.913 | INFO     | __main__:cache:20 -
Caching Data ----> test_3
Path ----> C:\Users\12460\Downloads\1112\images
2025-11-13 00:30:10.919 | INFO     | utils.cache_data:__get_label_from_name:36 -
Files number is 8599.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8599/8599 [00:00<00:00, 1379439.31it/s]
2025-11-13 00:30:10.926 | INFO     | utils.cache_data:__collect_data:92 -
Coolect labels is [" ", "1", "D", "Q", "F", "R", "P", "5", "9", "6", "X", "G", "H", "S", "V", "Z", "3", "T", "K", "J", "W", "2", "8", "4", "U", "Y", "E", "I", "C", "B", "L", "A", "7"]
2025-11-13 00:30:10.929 | INFO     | utils.cache_data:__collect_data:96 -
Writing Cache Data!
2025-11-13 00:30:10.929 | INFO     | utils.cache_data:__collect_data:98 -
Cache Data Number is 8599
2025-11-13 00:30:10.929 | INFO     | utils.cache_data:__collect_data:99 -
Writing Train and Val File.
2025-11-13 00:30:10.931 | INFO     | utils.cache_data:__collect_data:116 -
Train Data Number is 8342
2025-11-13 00:30:10.932 | INFO     | utils.cache_data:__collect_data:117 -
Val Data Number is 257
(dddd-trainer) PS C:\Users\12460\Documents\project\dddd_trainer> uv run .\app.py train test_3
2025-11-13 00:30:26.382 | INFO     | __main__:__init__:12 -
Hello baby~
2025-11-13 00:30:26.383 | INFO     | __main__:train:26 -
Start Train ----> test_3

2025-11-13 00:30:26.384 | INFO     | utils.train:__init__:40 -
Taget:
min_Accuracy: 0.97
min_Epoch: 20
max_Loss: 0.05
2025-11-13 00:30:26.384 | INFO     | utils.train:__init__:45 -
USE GPU ----> 0
2025-11-13 00:30:26.384 | INFO     | utils.train:__init__:52 -
Search for history checkpoints...
2025-11-13 00:30:26.384 | INFO     | utils.train:__init__:69 -
Empty history checkpoints
2025-11-13 00:30:26.384 | INFO     | utils.train:__init__:71 -
Building Net...
C:\Users\12460\Documents\project\dddd_trainer\.venv\Lib\site-packages\torch\nn\modules\rnn.py:123: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.3 and num_layers=1
  warnings.warn(
2025-11-13 00:30:26.400 | INFO     | utils.train:__init__:75 - Net(
  (cnn): DdddOcr(
    (cnn): Sequential(
      (conv0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu0): ReLU(inplace=True)
      (pooling0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv1): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu1): ReLU(inplace=True)
      (pooling1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (batchnorm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU(inplace=True)
      (conv3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu3): ReLU(inplace=True)
      (pooling2): MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False)
      (conv4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (batchnorm4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu4): ReLU(inplace=True)
      (conv5): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu5): ReLU(inplace=True)
      (pooling3): MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False)
      (conv6): Conv2d(128, 128, kernel_size=(2, 2), stride=(1, 1))
      (batchnorm6): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu6): ReLU(inplace=True)
    )
  )
  (lstm): LSTM(384, 384, dropout=0.3, bidirectional=True)
  (loss): CTCLoss()
  (fc): Linear(in_features=768, out_features=33, bias=True)
)
2025-11-13 00:30:26.400 | INFO     | utils.train:__init__:76 -
Building End
2025-11-13 00:30:26.509 | INFO     | utils.train:__init__:81 -
Get Data Loader...
2025-11-13 00:30:26.510 | INFO     | utils.load_cache:__init__:102 -
Charsets is [" ", "1", "D", "Q", "F", "R", "P", "5", "9", "6", "X", "G", "H", "S", "V", "Z", "3", "T", "K", "J", "W", "2", "8", "4", "U", "Y", "E", "I", "C", "B", "L", "A", "7"]
2025-11-13 00:30:26.510 | INFO     | utils.load_cache:__init__:106 -
Image Resize is [-1, 64]
2025-11-13 00:30:26.510 | INFO     | utils.load_cache:__init__:118 -
Image Path is C:\Users\12460\Downloads\1112\images
2025-11-13 00:30:26.511 | INFO     | utils.load_cache:__init__:25 -
Reading Cache File... ----> C:\Users\12460\Documents\project\dddd_trainer\projects\test_3\cache\cache.train.tmp
2025-11-13 00:30:26.512 | INFO     | utils.load_cache:__init__:30 -
Read Cache File End! Caches Num is 8342.
2025-11-13 00:30:26.512 | INFO     | utils.load_cache:__init__:25 -
Reading Cache File... ----> C:\Users\12460\Documents\project\dddd_trainer\projects\test_3\cache\cache.val.tmp
2025-11-13 00:30:26.513 | INFO     | utils.load_cache:__init__:30 -
Read Cache File End! Caches Num is 257.
2025-11-13 00:30:26.513 | INFO     | utils.train:__init__:87 -
Get Data Loader End!
2025-11-13 00:30:30.161 | INFO     | utils.train:start:108 - [2025-11-13-00_30_30]      Epoch: 0        Step: 100       LastLoss: 3.5620920658111572    AvgLoss: 4.684982450008392      Lr: 0.01
2025-11-13 00:30:33.520 | INFO     | utils.train:start:108 - [2025-11-13-00_30_33]      Epoch: 0        Step: 200       LastLoss: 3.629438877105713     AvgLoss: 3.591061780452728      Lr: 0.01
2025-11-13 00:30:36.940 | INFO     | utils.train:start:108 - [2025-11-13-00_30_36]      Epoch: 1        Step: 300       LastLoss: 3.478365659713745     AvgLoss: 3.574828338623047      Lr: 0.01
2025-11-13 00:30:40.677 | INFO     | utils.train:start:108 - [2025-11-13-00_30_40]      Epoch: 1        Step: 400       LastLoss: 3.725684881210327     AvgLoss: 3.5830772018432615     Lr: 0.01
2025-11-13 00:30:44.054 | INFO     | utils.train:start:108 - [2025-11-13-00_30_44]      Epoch: 1        Step: 500       LastLoss: 3.6253044605255127    AvgLoss: 3.5884796619415282     Lr: 0.01
...
2025-11-13 00:33:39.470 | INFO     | utils.train:start:108 - [2025-11-13-00_33_39]      Epoch: 19       Step: 5200      LastLoss: 0.0010958763305097818 AvgLoss: 0.0016259117133449763  Lr: 0.009604
2025-11-13 00:33:44.079 | INFO     | utils.train:start:108 - [2025-11-13-00_33_44]      Epoch: 20       Step: 5300      LastLoss: 0.0009493848774582148 AvgLoss: 0.0010536348074674606  Lr: 0.009604
2025-11-13 00:33:48.775 | INFO     | utils.train:start:108 - [2025-11-13-00_33_48]      Epoch: 20       Step: 5400      LastLoss: 0.0010489150881767273 AvgLoss: 0.0012551895825890823  Lr: 0.009604
2025-11-13 00:33:53.303 | INFO     | utils.train:start:108 - [2025-11-13-00_33_53]      Epoch: 21       Step: 5500      LastLoss: 0.0011606556363403797 AvgLoss: 0.000944445063942112   Lr: 0.009604
2025-11-13 00:33:57.803 | INFO     | utils.train:start:108 - [2025-11-13-00_33_57]      Epoch: 21       Step: 5600      LastLoss: 0.0007763305329717696 AvgLoss: 0.0009715505503118038  Lr: 0.009604
2025-11-13 00:34:02.339 | INFO     | utils.train:start:108 - [2025-11-13-00_34_02]      Epoch: 21       Step: 5700      LastLoss: 0.0012899019056931138 AvgLoss: 0.0015180535794934258  Lr: 0.009604
2025-11-13 00:34:06.902 | INFO     | utils.train:start:108 - [2025-11-13-00_34_06]      Epoch: 22       Step: 5800      LastLoss: 0.0009287762804888189 AvgLoss: 0.0015425046352902428  Lr: 0.009604
2025-11-13 00:34:11.336 | INFO     | utils.train:start:108 - [2025-11-13-00_34_11]      Epoch: 22       Step: 5900      LastLoss: 0.0007486791582778096 AvgLoss: 0.0011510271998122334  Lr: 0.009604
2025-11-13 00:34:15.845 | INFO     | utils.train:start:137 - [2025-11-13-00_34_15]      Epoch: 23       Step: 6000      LastLoss: 0.0006851484067738056 AvgLoss: 0.0009158773045055568  Lr: 0.009604    Acc: 1.0
2025-11-13 00:34:15.846 | INFO     | utils.train:start:143 -
Training Finished!Exporting Model...
C:\Users\12460\Documents\project\dddd_trainer\nets\__init__.py:216: UserWarning: # 'dynamic_axes' is not recommended when dynamo=True, and may lead to 'torch._dynamo.exc.UserError: Constraints violated.' Supply the 'dynamic_shapes' argument instead if export is unsuccessful.
  torch.onnx.export(net, dummy_input, graph_path, export_params=True, verbose=False,
W1113 00:34:16.515000 3024 .venv\Lib\site-packages\torch\onnx\_internal\exporter\_compat.py:114] Setting ONNX exporter to use operator set version 18 because the requested opset_version 12 is a lower version than we have implementations for. Automatic version conversion will be performed, which may not be successful at converting to the requested version. If version conversion is unsuccessful, the opset version of the exported model will be kept at 18. Please consider setting opset_version >=18 to leverage latest ONNX features
The model version conversion is not supported by the onnxscript version converter and fallback is enabled. The model will be converted using the onnx C API (target version: 12).
Failed to convert the model to the target version 12 using the ONNX C API. The model was not modified
Traceback (most recent call last):
  File "C:\Users\12460\Documents\project\dddd_trainer\.venv\Lib\site-packages\onnxscript\version_converter\__init__.py", line 127, in call
    converted_proto = _c_api_utils.call_onnx_api(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\12460\Documents\project\dddd_trainer\.venv\Lib\site-packages\onnxscript\version_converter\_c_api_utils.py", line 65, in call_onnx_api
    result = func(proto)
             ^^^^^^^^^^^
  File "C:\Users\12460\Documents\project\dddd_trainer\.venv\Lib\site-packages\onnxscript\version_converter\__init__.py", line 122, in _partial_convert_version
    return onnx.version_converter.convert_version(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\12460\Documents\project\dddd_trainer\.venv\Lib\site-packages\onnx\version_converter.py", line 39, in convert_version
    converted_model_str = C.convert_version(model_str, target_version)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: D:\a\onnx\onnx\onnx/version_converter/BaseConverter.h:68: adapter_lookup: Assertion `false` failed: No Adapter From Version $18 for Split
Skipping constant folding for op Split with multiple outputs.
Skipping constant folding for op Split with multiple outputs.
Applied 3 of general pattern rewrite rules.
Skipping constant folding for op Split with multiple outputs.
Skipping constant folding for op Split with multiple outputs.
2025-11-13 00:34:45.655 | INFO     | utils.train:start:159 -
Export Finished!Using Time: 3.816666666666667min
(dddd-trainer) PS C:\Users\12460\Documents\project\dddd_trainer> uv run .\test.py
2025-11-13 00:35:09.0336981 [E:onnxruntime:, sequential_executor.cc:572 onnxruntime::ExecuteKernel] Non-zero status code returned while running SequenceAt node. Name:'n0_424' Status Message: Invalid sequence index (57) specified for sequence of size (57)
Traceback (most recent call last):
  File "C:\Users\12460\Documents\project\dddd_trainer\test.py", line 14, in <module>
    res = ocr.classification(image)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\12460\Documents\project\dddd_trainer\.venv\Lib\site-packages\ddddocr\__init__.py", line 2643, in classification
    ort_outs = self.__ort_session.run(None, ort_inputs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\12460\Documents\project\dddd_trainer\.venv\Lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 287, in run
    return self._sess.run(output_names, input_feed, run_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running SequenceAt node. Name:'n0_424' Status Message: Invalid sequence index (57) specified for sequence of size (57)
```
