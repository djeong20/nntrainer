nntr_app_resdir = nntrainer_resdir / 'app'
run_command('mkdir', '-p', nntr_app_resdir)

subdir('utils')
subdir('KNN/jni')
subdir('LogisticRegression/jni')
if enable_ccapi
  subdir('MNIST/jni')
  if get_option('enable-fp16')
    subdir('MixedPrecision/jni')
  endif
endif
subdir('VGG/jni')
subdir('Resnet/jni')
subdir('YOLOv2/jni')
subdir('YOLOv3/jni')
subdir('LLaMA/jni')
subdir('Multi_input/jni')
subdir('ReinforcementLearning/DeepQ/jni')
subdir('TransferLearning/CIFAR_Classification/jni')
# if enable_capi
#   subdir('TransferLearning/Draw_Classification/jni')
# endif
subdir('Custom')
subdir('ProductRatings/jni')
subdir('AlexNet/jni')
subdir('Layers/jni')
if get_option('enable-tflite-backbone')
  subdir('SimpleShot')
endif
subdir('PicoGPT/jni')

subdir('SimpleFC/jni')
