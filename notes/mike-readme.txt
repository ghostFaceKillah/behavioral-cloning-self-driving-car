# Fuken comma ai showing how it's done, bitch
https://github.com/commaai/research/blob/master/train_steering_model.py

# Fuken forum, giving u good advice to listen to, nigga
https://carnd-forums.udacity.com/questions/26214464/behavioral-cloning-cheatsheet

# Fuken nvidia paper, to give u sth to think bout. Food 4 thought, fool
http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

# Fuken black guys doin it better then u
https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.qpvuux17j



Note: These points below are not the 'only' way of solving this problem. Think of them as pointers and feel free to pick and choose as you see fit.

0) How to use Python generators in Keras. This was critical as I was running
   out of memory on my laptop just trying to read in all the image data. Using
   generators allows me to only read in what I need at any point in time. Very
   useful.

1) Use a GPU. This should almost be a prerequisite. It is too frustrating
   waiting for hours for results on CPU. I must have run training 100 times over
   the past 3 weeks and it was driving me crazy. Using a GTX980M was around 20x
   faster in training that a quad-core Haswell CPU.

2) Use an analog joystick. This also should be a prerequisite. I'm not sure if
   its even possible to train with keyboard input. I think some have managed it,
   bu for me it's a case of garbage in, garbage out.

3) Use successive refinement of a 'good' model. This really saves time and
   ensures that you converge on a solution faster. So when you get a model working
   a little bit, say passing the first corner, then use that model as a starting
   point for your next training session (kinda like Transfer Learning). Generate
   some new IMG data, lower the learning rate, and 'fine tune' this model.

4) Use the 50Hz simulator. This generates much smoother driving angle data.
   Should be the default. You can find a link to download this on the Slack
   channel.

5) You need at least 40k samples to get a useful initial model. Anything less
   was not producing anything good for me.

6) Copy the Nvidia pipeline. It works :) And it's not too complex.

7) Re-size the input image. I was able to size the image down by 2, reducing
   the number of pixels by 4. This really helped speed up model training and did
   not seem to impact the accuracy.

8) I made use of the left and right camera angles also, where I modified the
   steering angles slightly in these cases. This helped up the number of test
   cases, and these help cases where the car is off center and teaches it to steer
   back to the middle.

9) Around 5 epochs seems to be enough training. Any more does not reduce the
   mse much if at all.

0) When you're starting out, pick three images from the .csv file, one with
   negative steering, one with straight, and one with right steering. Train a
   model with just those three images and see if you can get it to predict them
   correctly. This will tell you that your model is good and your turn-around time
   will be very quick. Then you can start adding more training data.
