# Motivation

I believe that at times where machines become able to learn, humans shall learn to teach! Mesmerized by the tremendous progress going on in _artificial intelligence (AI)_, I wondered what it takes to plan and implement an end-to-end deep learning scenario.

Starting from a platter of cute fluffy creatures, named _Noelinis_, this project takes you on a versatile journey through the worlds of digitization, model training and deployment. At the outcome stands a neat app that recognizes the _Noelinis_ remarkably well.

![Demo](https://github.com/ckauth/AI_Noelinis/blob/master/illustrations/demo.gif)

The project combines many interesting topics - data collection, image processing, deep learning, model training, model deployment and app programming - into a practical AI solution without requiring an excessive depth of knowledge in any single area.

# Data

My thanks go to all those who helped me collect the _Noelinis_. The first step consisted in digitizing representatives of the 14 different types. To later empower the learner to understand that the _Noelinis_ matter, while the background is irrelevant, I shot movies of the _Noelinis_ in front of a screen showing video clips.

![Noelinis](https://github.com/ckauth/AI_Noelinis/blob/master/illustrations/noelinis.jpg)

I collected 9 minutes' worth of video per _Noelini_ (plus the case where no _Noelini_ was on stage), zooming, translating and rotating the camera, to foster the necessary position and angle invariance when learning the model.

Next I cropped and scaled the [videos to 64 by 64 pixels](https://github.com/ckauth/AI_Noelinis/blob/master/data/movies/64x64) only. That is by far enough to recognize the _Noelinis_ reliably. Although I do not provide the original movies, you may find the [script](https://github.com/ckauth/AI_Noelinis/blob/master/data/mute_scale_crop_movies.py), which I used for that operation, useful. It calls [ffmpeg](https://www.ffmpeg.org/) under the hood.

A second [script](https://github.com/ckauth/AI_Noelinis/blob/master/data/extract_frames.py) saves the movies' frames as individual images. Shootings 1 through 5 are used for training, while shooting 6 is reserved for testing. Additionally, the script computes the [average image](https://github.com/ckauth/AI_Noelinis/blob/master/data/images/64x64/mean_image.xml), which is used for model training, and it composes the [training](https://github.com/ckauth/AI_Noelinis/blob/master/data/images/64x64/train_map.txt) and [test maps](https://github.com/ckauth/AI_Noelinis/blob/master/data/images/64x64/test_map.txt). A few of those nearly [70'000 64 by 64 pixels images](https://github.com/ckauth/AI_Noelinis/blob/master/data/images/64x64) are shown here below.

![Patchwork](https://github.com/ckauth/AI_Noelinis/blob/master/illustrations/patchwork.png)


# Training

I opted for a relatively traditional model as skeleton of the _Noelini_ classifier: It consists of 4 convolutional layers, with max-pooling and batch normalization, followed by two dense layers and a dropout layer to prevent overfitting.

For the training, I used Microsoft's Cognitive Toolkit (CNTK) via Python.  I let it iterate for 20 epochs on the dataset that included 26 different _Noelinis_, with decreasing learning rate. If you are interested in the details, feel free to have a close look at the [code](https://github.com/ckauth/AI_Noelinis/blob/master/training/train_model.py). I had no aspirations whatsoever to optimize the parameters - sometimes good enough is good enough.

The success rate on the training data approaches 99%, and exceeds 95% on the test data. Note that this slight discrepancy has a partial explanation in the fact that the training and testing data are not randomly sampled, but belong to very different shootings and background clips.

```bash
Training 665135 parameters in 22 parameter tensors.
Learning rate per minibatch: 0.1
Momentum per 1 samples: 0.9983550962823424
Finished Epoch[1 of 20]: [Training] loss = 1.648993 * 50000, metric = 53.82% * 50000 1044.211s (47.9 samples/s);
Finished Epoch[2 of 20]: [Training] loss = 0.615672 * 50000, metric = 19.42% * 50000 999.177s (50.0 samples/s);
Finished Epoch[3 of 20]: [Training] loss = 0.376024 * 50000, metric = 11.44% * 50000 1008.585s (49.6 samples/s);
Finished Epoch[4 of 20]: [Training] loss = 0.285115 * 50000, metric = 8.62% * 50000 1004.547s (49.8 samples/s);
Finished Epoch[5 of 20]: [Training] loss = 0.249327 * 50000, metric = 7.38% * 50000 1001.715s (49.9 samples/s);
Finished Epoch[6 of 20]: [Training] loss = 0.213227 * 50000, metric = 6.37% * 50000 997.387s (50.1 samples/s);
Learning rate per minibatch: 0.08
Finished Epoch[7 of 20]: [Training] loss = 0.171094 * 50000, metric = 4.96% * 50000 998.168s (50.1 samples/s);
Finished Epoch[8 of 20]: [Training] loss = 0.164531 * 50000, metric = 4.79% * 50000 997.267s (50.1 samples/s);
Learning rate per minibatch: 0.06
Finished Epoch[9 of 20]: [Training] loss = 0.131650 * 50000, metric = 3.78% * 50000 996.060s (50.2 samples/s);
Finished Epoch[10 of 20]: [Training] loss = 0.123007 * 50000, metric = 3.50% * 50000 996.607s (50.2 samples/s);
Learning rate per minibatch: 0.04
Finished Epoch[11 of 20]: [Training] loss = 0.105990 * 50000, metric = 3.01% * 50000 995.522s (50.2 samples/s);
Finished Epoch[12 of 20]: [Training] loss = 0.097483 * 50000, metric = 2.71% * 50000 997.967s (50.1 samples/s);
Learning rate per minibatch: 0.02
Finished Epoch[13 of 20]: [Training] loss = 0.083811 * 50000, metric = 2.34% * 50000 996.111s (50.2 samples/s);
Finished Epoch[14 of 20]: [Training] loss = 0.074809 * 50000, metric = 2.03% * 50000 999.263s (50.0 samples/s);
Learning rate per minibatch: 0.01
Finished Epoch[15 of 20]: [Training] loss = 0.064654 * 50000, metric = 1.69% * 50000 997.940s (50.1 samples/s);
Finished Epoch[16 of 20]: [Training] loss = 0.063490 * 50000, metric = 1.69% * 50000 998.272s (50.1 samples/s);
Learning rate per minibatch: 0.005
Finished Epoch[17 of 20]: [Training] loss = 0.058620 * 50000, metric = 1.52% * 50000 997.771s (50.1 samples/s);
Finished Epoch[18 of 20]: [Training] loss = 0.059092 * 50000, metric = 1.47% * 50000 998.230s (50.1 samples/s);
Finished Epoch[19 of 20]: [Training] loss = 0.055348 * 50000, metric = 1.49% * 50000 995.986s (50.2 samples/s);
Finished Epoch[20 of 20]: [Training] loss = 0.054067 * 50000, metric = 1.39% * 50000 1007.074s (49.6 samples/s);

Final Results: Minibatch[1-626]: errs = 3.7% * 10000
```

Upon completion of the training, the algorithm stores the [model](https://github.com/ckauth/AI_Noelinis/blob/master/training/NoeliniModel.dnn). You may test the performance of the model on single images of your choice with this [script](https://github.com/ckauth/AI_Noelinis/blob/master/training/evaluate_model.py).

# Deployment

The model is used within a WPF .NET 4.5 app, coded in C# and Xaml. I followed the MVVM pattern to my best efforts, while handling the screen-capturing from the code-behind. The laptop's webcam is interfaced via [Aforge's controls](http://www.aforgenet.com/aforge/framework/docs/), and the [MVVMLight toolkit](http://www.mvvmlight.net/) came in handy when architecting the code.

The app shows a continuous preview of the webcam and takes a screen capture of the preview once per second. This preview is handed to the model, which is evaluated via the [CNTK Library](https://docs.microsoft.com/en-us/cognitive-toolkit/cntk-eval-examples).

As of today, the code compiles for 64-bit targets only. Note that you may need to customize a few parameters for your machine too - my objective here was to implement an end-to-end AI scenario, not to write a store-grade app.

_Note:_ Out of the 32 _Noelinis_ featured in the demo video, 21 were part of the training set, while 11 were seen for the first time during the demo.

# User Guide

To run the code in the [data folder](https://github.com/ckauth/AI_Noelinis/blob/master/data/), you must have [ffmpeg](https://www.ffmpeg.org/) installed.

To run the code in the [training folder](https://github.com/ckauth/AI_Noelinis/blob/master/training/), you need an environment with matching versions of [CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine) and [Python](https://www.python.org/downloads/).

Finally, to run the code in the [deployment folder](https://github.com/ckauth/AI_Noelinis/blob/master/deployment/), you need [Visual Studio](https://www.visualstudio.com/) on Windows.


# Backstage

Before you go, let me give you a glimpse behind the scenes of my film studio ;)

![Backstage](https://github.com/ckauth/AI_Noelinis/blob/master/illustrations/backstage.jpg)
