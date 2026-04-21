

lut配置 ：使用class LUTRLU  min=5 ，max= o_s/(w_s * in_s) * 255 output_sign=0 (输出无符号数)



[layer1_0] (LinearReLU)
  Input  : scale=0.022238, zp=0
  Weight : scale=0.000945, zp=0
  Output : scale=0.034190, zp=0
[layer2_0] (LinearReLU)
  Input  : scale=0.034190, zp=0
  Weight : scale=0.001758, zp=0
  Output : scale=0.029330, zp=0
[classifier] (Linear)
  Input  : scale=0.029330, zp=0
  Weight : scale=0.009708, zp=0
  Output : scale=0.266873, zp=0

[classifier] (Linear):不需要使用LUT，膜电平输出的最大的那个就是分类结果


x torch.float32 torch.Size([1, 1, 28, 28])
view torch.float32 torch.Size([1, 784])
layer1_0 torch.int8 torch.Size([1, 512])
layer2_0 torch.int8 torch.Size([1, 256])
classifier torch.float32 torch.Size([1, 10])
output torch.float32 torch.Size([1, 10])
生成的目标底层模型图结构：
opcode       name        target                   args           kwargs
-----------  ----------  -----------------------  -------------  --------
placeholder  x           x                        ()             {}
call_method  size        size                     (x, 0)         {}
call_method  view        view                     (x, size, -1)  {}
call_module  layer1_0    manual_quant_layer1_0    (view,)        {}
call_module  layer2_0    manual_quant_layer2_0    (layer1_0,)    {}
call_module  classifier  manual_quant_classifier  (layer2_0,)    {}
output       output      output                   (classifier,)  {}

[5] 正在评估合并后的定点网络精度...

==================================================
===> 最终精度对比报告:
     FP32 (高精度):    98.52%
     Symmetric Int8: 98.46%
==================================================

