
转换后模型：
[features_0] (ConvReLU2d)
  Input  : scale=0.022304, zp=0
  Weight : scale=0.002591, zp=0
  Output : scale=0.031766, zp=0
[features_3] (ConvReLU2d)
  Input  : scale=0.031766, zp=0
  Weight : scale=0.002261, zp=0
  Output : scale=0.037156, zp=0
[classifier_0] (LinearReLU)
  Input  : scale=0.037156, zp=0
  Weight : scale=0.004294, zp=0
  Output : scale=0.068156, zp=0
[classifier_2] (LinearReLU)
  Input  : scale=0.068156, zp=0
  Weight : scale=0.005105, zp=0
  Output : scale=0.129020, zp=0
[classifier_4] (Linear)
  Input  : scale=0.129020, zp=0
  Weight : scale=0.010312, zp=0
  Output : scale=0.248746, zp=0


lut配置 ：使用class LUTRLU  min=5 ，max= o_s/(w_s * in_s) * 255 output_sign=0 (输出无符号数)
例如：features_0 max =0.031766/(0.002591*0.022304)*255 =140169

x torch.float32 torch.Size([1, 1, 28, 28])
features_0 torch.int8 torch.Size([1, 6, 28, 28])
features_2 torch.int8 torch.Size([1, 6, 14, 14])
features_3 torch.int8 torch.Size([1, 16, 10, 10])
features_5 torch.int8 torch.Size([1, 16, 5, 5])
flatten torch.int8 torch.Size([1, 400])
classifier_0 torch.int8 torch.Size([1, 120])
classifier_2 torch.int8 torch.Size([1, 84])
classifier_4 torch.float32 torch.Size([1, 10])
output torch.float32 torch.Size([1, 10])


生成的目标底层模型图结构：
opcode         name          target                                                          args             kwargs
-------------  ------------  --------------------------------------------------------------  ---------------  --------
placeholder    x             x                                                               ()               {}
call_module    features_0    manual_quant_features_0                                         (x,)             {}
call_module    features_2    features.2                                                      (features_0,)    {}
call_module    features_3    manual_quant_features_3                                         (features_2,)    {}
call_module    features_5    features.5                                                      (features_3,)    {}
call_function  flatten       <built-in method flatten of type object at 0x00007FF9AA725A80>  (features_5, 1)  {}
call_module    classifier_0  manual_quant_classifier_0                                       (flatten,)       {}
call_module    classifier_2  manual_quant_classifier_2                                       (classifier_0,)  {}
call_module    classifier_4  manual_quant_classifier_4                                       (classifier_2,)  {}
output         output        output                                                          (classifier_4,)  {}

[5] 正在评估合并后的定点网络精度...

==================================================
===> 最终精度对比报告:
     FP32 (高精度):    98.63%
     Symmetric Int8: 98.56%
==================================================

