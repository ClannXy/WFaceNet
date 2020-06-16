WFaceNet_112X112 = [  
    # oup, exp, k, s,    SE,      NL
    [ 32,   64, 3, 2,  True, "prelu" ],    
    [ 64,  128, 3, 2, False, "prelu" ],    
    [ 64,  128, 3, 1, False, "prelu" ],
    [128,  256, 3, 2,  True, "prelu" ],    
    [128,  256, 3, 1,  True, "prelu" ],
    [128,  256, 3, 1,  True, "prelu" ], 
    [128,  256, 3, 1,  True, "prelu" ],
    [128,  256, 3, 1,  True, "prelu" ],
    [128,  512, 3, 2,  True, "prelu" ],    
    [128,  256, 3, 1,  True, "prelu" ],
    [128,  256, 3, 1,  True, "prelu" ],
]
# parameters: 1163392.
# flops: 338.071456m
# ave: 99.3500%  47E,64E,66E

WFaceNet_112X112_hs = [  
    # oup, exp, k, s,    SE,      NL
    [ 32,  64,  3, 2,  True, "hswish" ],    
    [ 64,  128, 3, 2, False, "hswish" ],   
    [ 64,  128, 3, 1, False, "hswish" ],
    [128,  256, 3, 2,  True, "hswish" ],   
    [128,  256, 3, 1,  True, "hswish" ],
    [128,  256, 3, 1,  True, "hswish" ], 
    [128,  256, 3, 1,  True, "hswish" ],
    [128,  256, 3, 1,  True, "hswish" ],
    [128,  512, 3, 2,  True, "hswish" ],   
    [128,  256, 3, 1,  True, "hswish" ],
    [128,  256, 3, 1,  True, "hswish" ],
]
# flops: 338.071456m
# params: 1.163392 m
# ave: 99.3500%  47E,64E,66E

MobileFaceNet_Setting= [ 
    # oup, exp, k, s,    SE,      NL
    [  64, 128, 3, 2, False,  "prelu" ],
    [  64, 128, 3, 1, False,  "prelu" ],
    [  64, 128, 3, 1, False,  "prelu" ],
    [  64, 128, 3, 1, False,  "prelu" ],
    [  64, 128, 3, 1, False,  "prelu" ],
    [ 128, 256, 3, 2, False,  "prelu" ], 
    [ 128, 256, 3, 1, False,  "prelu" ],
    [ 128, 256, 3, 1, False,  "prelu" ],
    [ 128, 256, 3, 1, False,  "prelu" ],
    [ 128, 256, 3, 1, False,  "prelu" ],
    [ 128, 256, 3, 1, False,  "prelu" ],
    [ 128, 256, 3, 1, False,  "prelu" ],
    [ 128, 512, 3, 2, False,  "prelu" ],
    [ 128, 256, 3, 1, False,  "prelu" ],
    [ 128, 256, 3, 1, False,  "prelu" ],
]
# paramerters: 995104.

MOBILENET_V3_SMALL_BOTTLENECK_SETTING= [    
    # oup, exp, k, s,    SE,      NL
    [ 16,   16, 3, 2,  True,   "relu" ],   
    [ 24,   72, 3, 2, False,   "relu" ],   
    [ 24,   88, 3, 1, False,   "relu" ],
    [ 40,   96, 5, 2,  True, "hswish" ],   
    [ 40,  240, 5, 1,  True, "hswish" ],
    [ 40,  240, 5, 1,  True, "hswish" ], 
    [ 48,  120, 5, 1,  True, "hswish" ],
    [ 48,  144, 5, 1,  True, "hswish" ],
    [ 96,  288, 5, 2,  True, "hswish" ],    
    [ 96,  576, 5, 1,  True, "hswish" ],
    [ 96,  576, 5, 1,  True, "hswish" ],
]
# paramerters: 1005288.

Test_Setting_21= [ 
    # oup, exp, k, s,    SE,      NL
    [ 32,  64, 3, 2,  True, "prelu" ],    
    [ 64,  128, 3, 2, False, "prelu" ],    
    [ 64,  128, 3, 1, False, "prelu" ],
    [128,  256, 5, 2,  True, "prelu" ],    
    [128,  256, 5, 1,  True, "prelu" ],
    [128,  256, 5, 1,  True, "prelu" ], 
    [128,  256, 5, 1,  True, "prelu" ],
    [128,  256, 5, 1,  True, "prelu" ],
    [128,  512, 5, 2,  True, "prelu" ],   
    [128,  256, 5, 1,  True, "prelu" ],
    [128,  256, 5, 1,  True, "prelu" ],
]
# params: 1200256.
# ave: 99.3167%

Test_Setting_33 = [  
    # oup, exp, k, s,    SE,      NL
    [ 32,  64, 3, 2,  False, "prelu" ],   
    [ 64,  128, 3, 2, False, "prelu" ],    
    [ 64,  128, 3, 1, False, "prelu" ],
    [128,  256, 3, 2, False, "prelu" ],   
    [128,  256, 3, 1, False, "prelu" ],
    [128,  256, 3, 1, False, "prelu" ], 
    [128,  256, 3, 1, False, "prelu" ],
    [128,  256, 3, 1, False, "prelu" ],
    [128,  512, 3, 2, False, "prelu" ],   
    [128,  256, 3, 1, False, "prelu" ],
    [128,  256, 3, 1, False, "prelu" ],
]
# paramerters: 806688.
# flops: 336.338208m
# Epoch 65: 99.13%

Min_Setting = [
    # oup, exp, k, s,    SE,      NL
    [ 32,  64, 3, 2,  False, "prelu" ],   
    [ 64,  128, 3, 2, False, "prelu" ],    
    [ 64,  128, 3, 2, False, "prelu" ],
]
# paramerters: 163872.



# current 
# 当前网络调用的
current_setting = WFaceNet_112X112