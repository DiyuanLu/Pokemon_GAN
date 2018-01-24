#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# code adapted from 
import tensorflow as tf #machine learning
import numpy as np #matrix math
import datetime #logging the time for model checkpoints and training
import matplotlib
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt #visualize results
import gan_prepro as pre_data
import ipdb
#from tensorflow.examples.tutorials.mnist import input_data
##will ensure that the correct data has been downloaded to your
##local training folder and then unpack that data to return a dictionary of DataSet instances.
#mnist = input_data.read_data_sets("MNIST_data/")

# Generator seeks to take a d-dimensional noise vector and upsample it to become a 28 x 28 image.
#example of CNN blocks http://cs231n.github.io/convolutional-networks/

version = 'p228'      #'newspectrogram''MNIST'
results_dir = 'results/' + version + "/{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
logdir = 'model/' + version + "/{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())

if not os.path.exists(logdir) or not os.path.exists(results_dir):
    os.makedirs(logdir)
    os.makedirs(results_dir)

def generator(batch_size, z_dim):
    '''
    Param:
        batch_size: the number of images we generated
        z_dim: the latent representation dimension where the generator start of generating images'''
    z = tf.truncated_normal([batch_size, z_dim], mean=0, stddev=1, name='z')
    #first deconv block
    img_dim1 = 56
    # transform the latent vector to a 56 * 56 image
    g_w1 = tf.get_variable('g_w1', [z_dim, img_dim1*img_dim1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [img_dim1*img_dim1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, img_dim1, img_dim1, 1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)

    # Generate feature vector in half the size of latent representation
    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [img_dim1, img_dim1])

    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [img_dim1, img_dim1])

    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)

    # No batch normalization at the final layer, but we do add
    # a sigmoid activator to make the generated images crisper.
    # Dimensions of g4: batch_size x 28 x 28 x 1

    return g4

def discriminator(x_image, reuse=False):
    if (reuse):
        tf.get_variable_scope().reuse_variables()

    # First convolutional and pool layers
    # These search for 32 different 5 x 5 pixel features
    #Our first weight matrix (or filter) will be of size 5x5 and will have a output depth of 32.
    #It will be randomly initialized from a normal distribution.
    d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))   
    #tf.constant_init generates tensors with constant values.
    d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))

    #strides = [batch, height, width, channels]
    d1 = tf.nn.conv2d(input=x_image, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
    #add the bias
    d1 = d1 + d_b1
    #squash with nonlinearity (ReLU)
    d1 = tf.nn.relu(d1)
    ##An average pooling layer performs down-sampling by dividing the input into
    #rectangular pooling regions and computing the average of each region.
    #It returns the averages for the pooling regions.
    d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second convolutional and pool layers
    # These search for 64 different 5 x 5 pixel features
    # search for 5*5 feature when taking 32 input from last layer and get 64 features
    d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
    d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
    d2 = d2 + d_b2
    d2 = tf.nn.relu(d2)
    d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # First fully connected layer
    d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
    d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
    d3 = tf.matmul(d3, d_w3)    # batch_size * 1024
    d3 = d3 + d_b3
    d3 = tf.nn.relu(d3)         # batch_size * 1024

    #The last fully-connected layer holds the output, such as the class scores.
    # Second fully connected layer
    d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))

    #At the end of the network, we do a final matrix multiply and
    #return the activation value.
    #For those of you comfortable with CNNs, this is just a simple binary classifier. Nothing fancy.
    # Final layer
    d4 = tf.matmul(d3, d_w4) + d_b4
    # d4 dimensions: batch_size x 1

    return d4

# start TensorFlow session
sess = tf.Session()
batch_size = 10
z_dimensions = 100     # the dimensionality of latent representation

x_placeholder = tf.placeholder("float", shape = [None,28,28,1], name='x_placeholder')
# x_placeholder is for feeding input images to the discriminator
#The generator is constantly improving to produce more and more realistic images, while the discriminator is
#trying to get better and better at distinguishing between real and generated images.
#This means that we need to formulate loss functions that affect both networks.

# Gz: G(z)) holds the generated images
Gz = generator(batch_size, z_dimensions)

# Dx: D(x) hold the discriminator's prediction probabilities for real MNIST images
Dx = discriminator(x_placeholder)

# Dg: D(G(z)) holds discriminator prediction probabilities for generated images
Dg = discriminator(Gz, reuse=True)



# The generator wants the discriminator to output a 1 (positive example) for it's generated
# images'.
#Therefore, we want to compute the loss between the Dg and label of 1. This can be done through the tf.nn.sigmoid_cross_entropy_with_logits function.
# This means that the cross entropy loss will
#be taken between the two arguments. The "with_logits" component means that the function will operate
#on unscaled values. Basically, this means that instead of using a softmax function to squish the output
#activations to probability values from 0 to 1, we simply return the unscaled value of the matrix multiplication.
#Take a look at the last line of our discriminator. There's no softmax or sigmoid layer at the end.
#The reduce mean function just takes the mean value of all of the components in the matrixx returned
#by the cross entropy function. This is just a way of reducing the loss to a single scalar value, instead of a vector or matrix.
#https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

# Discriminator’s goal is to just get the correct labels (output 1 for each MNIST digit and 0 for the generated ones). We’d like to compute the loss between Dx
#and the correct label of 1 as well as the loss between Dg and the correct label of 0.
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.fill([batch_size, 1], 0.9)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
d_loss = d_loss_real + d_loss_fake

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

# Train the discriminator
# Increasing from 0.001 in GitHub version
with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
    #Next, we specify our two optimizers. In today’s era of deep learning, Adam seems to be the best SGD optimizer as it utilizes adaptive learning rates and momentum.
    # D 1: minimize the error when doing--the images which are FAKE being classified as FAKE
    #d_trainer_fake = tf.train.AdamOptimizer().minimize(d_loss_fake, var_list=d_vars)
    d_trainer_fake = tf.train.GradientDescentOptimizer(1e-3).minimize(d_loss_fake, var_list=d_vars)
    # D 1: minimize the error when doing-- the images which are REAL being classified as REAL
    #d_trainer_real = tf.train.AdamOptimizer().minimize(d_loss_real, var_list=d_vars)
    d_trainer_real = tf.train.GradientDescentOptimizer(1e-3).minimize(d_loss_real, var_list=d_vars)

    # Train the generator
    # G: minimize the error when doing--FAKE images being classified as REAL 1
    #g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)
    g_trainer = tf.train.GradientDescentOptimizer(1e-3).minimize(g_loss, var_list=g_vars)


#Outputs a Summary protocol buffer containing a single scalar value.
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_on_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_on_fake', d_loss_fake)

d_real_count_ph = tf.placeholder(tf.float32)
d_fake_count_ph = tf.placeholder(tf.float32)
g_count_ph = tf.placeholder(tf.float32)

tf.summary.scalar('d_real_count', d_real_count_ph)
tf.summary.scalar('d_fake_count', d_fake_count_ph)
tf.summary.scalar('g_count', g_count_ph)

# Sanity check to see how the discriminator evaluates
# generated and real MNIST images
d_on_generated = tf.reduce_mean(discriminator(generator(batch_size, z_dimensions)))
d_on_real = tf.reduce_mean(discriminator(x_placeholder))

tf.summary.scalar('d_on_generated_eval', d_on_generated)
tf.summary.scalar('d_on_real_eval', d_on_real)

images_for_tensorboard = generator(batch_size, z_dimensions)
tf.summary.image('Generated_images', images_for_tensorboard, 10)
merged = tf.summary.merge_all()
#logdir = "tensorboard/gan/"
writer = tf.summary.FileWriter(logdir+'/', sess.graph)
print(logdir)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

#During every iteration, there will be two updates being made, one to the discriminator and one to the generator.
#For the generator update, we’ll feed in a random z vector to the generator and pass that output to the discriminator
#to obtain a probability score (this is the Dg variable we specified earlier).
#As we remember from our loss function, the cross entropy loss gets minimized,
#and only the generator’s weights and biases get updated.
#We'll do the same for the discriminator update. We’ll be taking a batch of images
#from the mnist variable we created way at the beginning of our program.
#These will serve as the positive examples, while the images in the previous section are the negative ones.

gLoss = 0
dLossFake, dLossReal = 1, 1
d_real_count, d_fake_count, g_count = 0, 0, 0
total_steps = 100
audio_dir = '/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Pokemon_GAN/audio_data/p228'
for i in range(total_steps):
    spectro, magnit, num_batch = pre_data.get_batch(audio_dir)
    ipdb.set_trace()
    #real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    # Train D on generated images
    if dLossFake > 0.6:    # FAKE being classified as FAKE
        # Train discriminator on generated images
        _, dLossReal, dLossFake, gLoss = sess.run([d_trainer_fake, d_loss_real, d_loss_fake                         , g_loss], {x_placeholder: real_image_batch})
        d_fake_count += 1

    if dLossReal > 0.45:   # REAL being classified as REAL
        # If the discriminator classifies real images as fake,
        # train discriminator on real values
        _, dLossReal, dLossFake, gLoss = sess.run([d_trainer_real, d_loss_real, d_loss_fake                         , g_loss], {x_placeholder: real_image_batch})
        d_real_count += 1
        
    if gLoss > 0.5:
        # Train the generator, FAKE being classified as REAL
        _, dLossReal, dLossFake, gLoss = sess.run([g_trainer, d_loss_real, d_loss_fake,                         g_loss], {x_placeholder: real_image_batch})
        g_count += 1

    

    if i % 10 == 0:
        print "training step: ", i
        real_image_batch = mnist.validation.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
        summary = sess.run(merged, {x_placeholder: real_image_batch, d_real_count_ph:                           d_real_count, d_fake_count_ph: d_fake_count,
                                    g_count_ph: g_count})
        writer.add_summary(summary, i)
        d_real_count, d_fake_count, g_count = 0, 0, 0

    if i % 50 == 0:
        gen_num = 5
        # Use current trained model to generate 3 images
        images = sess.run(generator(gen_num, z_dimensions))
        # D classify these images
        d_result = sess.run(discriminator(x_placeholder), {x_placeholder: images})

        for j in range(gen_num):
            print("Discriminator classification", d_result[j])
            im = images[j, :, :, 0]
            plt.imshow(im.reshape([28, 28]), cmap='Greys')
            plt.title("score on generated images")
            plt.xlabel("score={}".format(d_result[j]))
            plt.savefig(results_dir + '/Gen_step{}'.format(i))
            plt.close()

    if i % 5000 == 0:
        save_path = saver.save(sess, logdir + "/pretrained_gan{}.ckpt".format(i), global_step=i)
        print("saved to %s" % save_path)

# testing
test_images = sess.run(generator(10, 100))
test_eval = sess.run(discriminator(x_placeholder), {x_placeholder: test_images})

real_images = mnist.validation.next_batch(10)[0].reshape([10, 28, 28, 1])
real_eval = sess.run(discriminator(x_placeholder), {x_placeholder: real_images})

# Show discriminator's probabilities for the generated images,
# and display the images
for ii in range(10):
    plt.imshow(test_images[ii, :, :, 0], cmap='Greys')
    plt.title("Score on generated images")
    plt.xlabel("score={}".format(test_eval[ii]))
    plt.savefig(results_dir + '/testOnGen_{}_after_{}_step'.format(ii, total_steps))
    plt.close()

# Now do the same for real MNIST images
for ii in range(10):
    plt.imshow(real_images[ii, :, :, 0], cmap='Greys')
    plt.title("Score on MNIST images")
    plt.xlabel("score={}".format(real_eval[ii]))
    plt.savefig(results_dir + '/testOnReal_{}_after_{}_step'.format(ii, total_steps))
    plt.close()
