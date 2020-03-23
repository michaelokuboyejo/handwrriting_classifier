NN Based Handwriting Classifier Using Keras
-----

This is a simple project that demonstrates the use of a Neural Network (NN) to classify and identify numeric digits from hand writings. The project uses the MNIST dataset.
This example project is taken from François Charlotte's [Deep Learrning With Python Book](https://www.manning.com/books/deep-learning-with-python)

# Dependencies
- Keras

# How to run this project

1. Install the project's dependencies using the command : `pip install -r requirements.txt`
2. Run main.py: `python main.py`

## DEMO

```shell
/Users/michael.okuboyejo/PycharmProjects/keras_handwriting_classifier/venv/bin/python /Users/michael.okuboyejo/PycharmProjects/keras_handwriting_classifier/main.py
Using TensorFlow backend.
2020-03-23 00:56:40.245344: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-03-23 00:56:40.267013: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7fd892cc5130 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-03-23 00:56:40.267029: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz

    8192/11490434 [..............................] - ETA: 0s
   16384/11490434 [..............................] - ETA: 1:12
   57344/11490434 [..............................] - ETA: 41s 
  122880/11490434 [..............................] - ETA: 28s
  262144/11490434 [..............................] - ETA: 17s
  524288/11490434 [>.............................] - ETA: 10s
  786432/11490434 [=>............................] - ETA: 8s 
 1556480/11490434 [===>..........................] - ETA: 4s
 3104768/11490434 [=======>......................] - ETA: 2s
 5120000/11490434 [============>.................] - ETA: 1s
 7168000/11490434 [=================>............] - ETA: 0s
 9199616/11490434 [=======================>......] - ETA: 0s
11280384/11490434 [============================>.] - ETA: 0s
11493376/11490434 [==============================] - 1s 0us/step

training_images.shape: (60000, 28, 28)
len(training_images): 60000
test_images.shape: (10000, 28, 28)
len(test_images): 10000

Epoch 1/5

  128/60000 [..............................] - ETA: 50s - loss: 2.3167 - accuracy: 0.0938
 2432/60000 [>.............................] - ETA: 3s - loss: 0.9373 - accuracy: 0.7344 
 5120/60000 [=>............................] - ETA: 2s - loss: 0.6922 - accuracy: 0.8055
 7936/60000 [==>...........................] - ETA: 1s - loss: 0.5788 - accuracy: 0.8367
10752/60000 [====>.........................] - ETA: 1s - loss: 0.5217 - accuracy: 0.8526
13568/60000 [=====>........................] - ETA: 1s - loss: 0.4740 - accuracy: 0.8634
16512/60000 [=======>......................] - ETA: 1s - loss: 0.4423 - accuracy: 0.8722
19328/60000 [========>.....................] - ETA: 0s - loss: 0.4149 - accuracy: 0.8789
22144/60000 [==========>...................] - ETA: 0s - loss: 0.3942 - accuracy: 0.8850
25088/60000 [===========>..................] - ETA: 0s - loss: 0.3750 - accuracy: 0.8907
28032/60000 [=============>................] - ETA: 0s - loss: 0.3588 - accuracy: 0.8954
30848/60000 [==============>...............] - ETA: 0s - loss: 0.3451 - accuracy: 0.8999
33792/60000 [===============>..............] - ETA: 0s - loss: 0.3336 - accuracy: 0.9032
36736/60000 [=================>............] - ETA: 0s - loss: 0.3194 - accuracy: 0.9070
39680/60000 [==================>...........] - ETA: 0s - loss: 0.3101 - accuracy: 0.9096
42496/60000 [====================>.........] - ETA: 0s - loss: 0.3001 - accuracy: 0.9126
45440/60000 [=====================>........] - ETA: 0s - loss: 0.2905 - accuracy: 0.9153
48384/60000 [=======================>......] - ETA: 0s - loss: 0.2829 - accuracy: 0.9174
51328/60000 [========================>.....] - ETA: 0s - loss: 0.2746 - accuracy: 0.9198
54272/60000 [==========================>...] - ETA: 0s - loss: 0.2679 - accuracy: 0.9220
57216/60000 [===========================>..] - ETA: 0s - loss: 0.2609 - accuracy: 0.9239
60000/60000 [==============================] - 1s 20us/step - loss: 0.2552 - accuracy: 0.9257
Epoch 2/5

  128/60000 [..............................] - ETA: 1s - loss: 0.1043 - accuracy: 0.9844
 3072/60000 [>.............................] - ETA: 0s - loss: 0.1184 - accuracy: 0.9652
 5888/60000 [=>............................] - ETA: 0s - loss: 0.1186 - accuracy: 0.9643
 8576/60000 [===>..........................] - ETA: 0s - loss: 0.1167 - accuracy: 0.9654
11520/60000 [====>.........................] - ETA: 0s - loss: 0.1205 - accuracy: 0.9652
14464/60000 [======>.......................] - ETA: 0s - loss: 0.1178 - accuracy: 0.9656
17280/60000 [=======>......................] - ETA: 0s - loss: 0.1155 - accuracy: 0.9666
20224/60000 [=========>....................] - ETA: 0s - loss: 0.1130 - accuracy: 0.9675
23168/60000 [==========>...................] - ETA: 0s - loss: 0.1128 - accuracy: 0.9672
26112/60000 [============>.................] - ETA: 0s - loss: 0.1104 - accuracy: 0.9678
29056/60000 [=============>................] - ETA: 0s - loss: 0.1086 - accuracy: 0.9682
31872/60000 [==============>...............] - ETA: 0s - loss: 0.1085 - accuracy: 0.9682
34816/60000 [================>.............] - ETA: 0s - loss: 0.1075 - accuracy: 0.9688
37760/60000 [=================>............] - ETA: 0s - loss: 0.1062 - accuracy: 0.9690
40576/60000 [===================>..........] - ETA: 0s - loss: 0.1059 - accuracy: 0.9690
43136/60000 [====================>.........] - ETA: 0s - loss: 0.1066 - accuracy: 0.9687
45440/60000 [=====================>........] - ETA: 0s - loss: 0.1064 - accuracy: 0.9688
48000/60000 [=======================>......] - ETA: 0s - loss: 0.1060 - accuracy: 0.9690
50944/60000 [========================>.....] - ETA: 0s - loss: 0.1051 - accuracy: 0.9693
53888/60000 [=========================>....] - ETA: 0s - loss: 0.1048 - accuracy: 0.9695
56704/60000 [===========================>..] - ETA: 0s - loss: 0.1042 - accuracy: 0.9696
59520/60000 [============================>.] - ETA: 0s - loss: 0.1029 - accuracy: 0.9699
60000/60000 [==============================] - 1s 18us/step - loss: 0.1026 - accuracy: 0.9700
Epoch 3/5

  128/60000 [..............................] - ETA: 1s - loss: 0.0469 - accuracy: 0.9844
 2944/60000 [>.............................] - ETA: 1s - loss: 0.0758 - accuracy: 0.9803
 5376/60000 [=>............................] - ETA: 1s - loss: 0.0707 - accuracy: 0.9799
 7424/60000 [==>...........................] - ETA: 1s - loss: 0.0737 - accuracy: 0.9776
 9600/60000 [===>..........................] - ETA: 1s - loss: 0.0715 - accuracy: 0.9790
12032/60000 [=====>........................] - ETA: 1s - loss: 0.0744 - accuracy: 0.9780
14848/60000 [======>.......................] - ETA: 0s - loss: 0.0727 - accuracy: 0.9782
17664/60000 [=======>......................] - ETA: 0s - loss: 0.0719 - accuracy: 0.9783
20480/60000 [=========>....................] - ETA: 0s - loss: 0.0712 - accuracy: 0.9786
23296/60000 [==========>...................] - ETA: 0s - loss: 0.0696 - accuracy: 0.9793
26112/60000 [============>.................] - ETA: 0s - loss: 0.0706 - accuracy: 0.9791
28928/60000 [=============>................] - ETA: 0s - loss: 0.0700 - accuracy: 0.9796
31616/60000 [==============>...............] - ETA: 0s - loss: 0.0706 - accuracy: 0.9797
34432/60000 [================>.............] - ETA: 0s - loss: 0.0701 - accuracy: 0.9798
37248/60000 [=================>............] - ETA: 0s - loss: 0.0696 - accuracy: 0.9800
40064/60000 [===================>..........] - ETA: 0s - loss: 0.0691 - accuracy: 0.9800
42880/60000 [====================>.........] - ETA: 0s - loss: 0.0684 - accuracy: 0.9802
45696/60000 [=====================>........] - ETA: 0s - loss: 0.0684 - accuracy: 0.9801
48512/60000 [=======================>......] - ETA: 0s - loss: 0.0682 - accuracy: 0.9801
51328/60000 [========================>.....] - ETA: 0s - loss: 0.0674 - accuracy: 0.9802
54144/60000 [==========================>...] - ETA: 0s - loss: 0.0678 - accuracy: 0.9802
56960/60000 [===========================>..] - ETA: 0s - loss: 0.0676 - accuracy: 0.9802
59776/60000 [============================>.] - ETA: 0s - loss: 0.0680 - accuracy: 0.9803
60000/60000 [==============================] - 1s 19us/step - loss: 0.0678 - accuracy: 0.9803
Epoch 4/5

  128/60000 [..............................] - ETA: 1s - loss: 0.0714 - accuracy: 0.9844
 2560/60000 [>.............................] - ETA: 1s - loss: 0.0380 - accuracy: 0.9887
 5248/60000 [=>............................] - ETA: 1s - loss: 0.0429 - accuracy: 0.9872
 8064/60000 [===>..........................] - ETA: 1s - loss: 0.0432 - accuracy: 0.9874
10880/60000 [====>.........................] - ETA: 0s - loss: 0.0412 - accuracy: 0.9879
13440/60000 [=====>........................] - ETA: 0s - loss: 0.0425 - accuracy: 0.9874
16128/60000 [=======>......................] - ETA: 0s - loss: 0.0439 - accuracy: 0.9866
18944/60000 [========>.....................] - ETA: 0s - loss: 0.0449 - accuracy: 0.9866
21760/60000 [=========>....................] - ETA: 0s - loss: 0.0489 - accuracy: 0.9856
24576/60000 [===========>..................] - ETA: 0s - loss: 0.0489 - accuracy: 0.9858
27392/60000 [============>.................] - ETA: 0s - loss: 0.0501 - accuracy: 0.9857
30208/60000 [==============>...............] - ETA: 0s - loss: 0.0506 - accuracy: 0.9854
33024/60000 [===============>..............] - ETA: 0s - loss: 0.0506 - accuracy: 0.9855
35840/60000 [================>.............] - ETA: 0s - loss: 0.0506 - accuracy: 0.9854
38656/60000 [==================>...........] - ETA: 0s - loss: 0.0497 - accuracy: 0.9856
41472/60000 [===================>..........] - ETA: 0s - loss: 0.0498 - accuracy: 0.9857
44288/60000 [=====================>........] - ETA: 0s - loss: 0.0503 - accuracy: 0.9855
47104/60000 [======================>.......] - ETA: 0s - loss: 0.0503 - accuracy: 0.9853
49792/60000 [=======================>......] - ETA: 0s - loss: 0.0500 - accuracy: 0.9853
52608/60000 [=========================>....] - ETA: 0s - loss: 0.0497 - accuracy: 0.9852
55424/60000 [==========================>...] - ETA: 0s - loss: 0.0494 - accuracy: 0.9854
58112/60000 [============================>.] - ETA: 0s - loss: 0.0497 - accuracy: 0.9853
60000/60000 [==============================] - 1s 19us/step - loss: 0.0493 - accuracy: 0.9854
Epoch 5/5

  128/60000 [..............................] - ETA: 1s - loss: 0.0131 - accuracy: 1.0000
 2944/60000 [>.............................] - ETA: 1s - loss: 0.0360 - accuracy: 0.9905
 5760/60000 [=>............................] - ETA: 0s - loss: 0.0319 - accuracy: 0.9911
 8448/60000 [===>..........................] - ETA: 0s - loss: 0.0339 - accuracy: 0.9902
11008/60000 [====>.........................] - ETA: 0s - loss: 0.0345 - accuracy: 0.9905
13824/60000 [=====>........................] - ETA: 0s - loss: 0.0364 - accuracy: 0.9894
16640/60000 [=======>......................] - ETA: 0s - loss: 0.0368 - accuracy: 0.9893
19456/60000 [========>.....................] - ETA: 0s - loss: 0.0369 - accuracy: 0.9892
22272/60000 [==========>...................] - ETA: 0s - loss: 0.0374 - accuracy: 0.9887
24960/60000 [===========>..................] - ETA: 0s - loss: 0.0370 - accuracy: 0.9890
27776/60000 [============>.................] - ETA: 0s - loss: 0.0365 - accuracy: 0.9893
30592/60000 [==============>...............] - ETA: 0s - loss: 0.0358 - accuracy: 0.9895
33280/60000 [===============>..............] - ETA: 0s - loss: 0.0354 - accuracy: 0.9896
35968/60000 [================>.............] - ETA: 0s - loss: 0.0354 - accuracy: 0.9896
38784/60000 [==================>...........] - ETA: 0s - loss: 0.0368 - accuracy: 0.9892
41600/60000 [===================>..........] - ETA: 0s - loss: 0.0363 - accuracy: 0.9892
44160/60000 [=====================>........] - ETA: 0s - loss: 0.0362 - accuracy: 0.9893
46848/60000 [======================>.......] - ETA: 0s - loss: 0.0365 - accuracy: 0.9892
49664/60000 [=======================>......] - ETA: 0s - loss: 0.0366 - accuracy: 0.9892
52480/60000 [=========================>....] - ETA: 0s - loss: 0.0360 - accuracy: 0.9892
55296/60000 [==========================>...] - ETA: 0s - loss: 0.0365 - accuracy: 0.9890
58112/60000 [============================>.] - ETA: 0s - loss: 0.0364 - accuracy: 0.9891
60000/60000 [==============================] - 1s 19us/step - loss: 0.0366 - accuracy: 0.9889

   32/10000 [..............................] - ETA: 5s
 3712/10000 [==========>...................] - ETA: 0s
 7648/10000 [=====================>........] - ETA: 0s
10000/10000 [==============================] - 0s 15us/step

test_loss: 0.07262361633204854
test_accuracy: 97.93999791145325 %

```

## TODO
- Write the programme in a Jupyter Notebook
- Build an HTTP API to serve requests and classify handwritings 😉
