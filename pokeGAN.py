
import os
import tensorflow as tf
import numpy as np
import cv2
import random
import scipy.misc
from utils import *
import datetime
import ipdb
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#will ensure that the correct data has been downloaded to your
#local training folder and then unpack that data to return a dictionary of DataSet instances.
mnist = input_data.read_data_sets("MNIST_data/")



slim = tf.contrib.slim
save_every = 200
plot_every = 10
HEIGHT, WIDTH, CHANNEL = 28, 28, 1
BATCH_SIZE = 64
EPOCH = 5000
gen_eval_num = 20
os.environ['CUDA_VISIBLE_DEVICES'] = '15'
version =  'mnist_pm'      #FSD_pretty_128FSD_pretty_128_pad'newspectrogram' 'OLLO_NO1'
#results_dir = 'results/' + version
#logdir = './model/' + version

results_dir = 'results/' + version + "/{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
logdir = './model/' + version + "/{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)


#def load(saver, sess, logdir):
    #print("Trying to restore saved checkpoints from {} ...".format(logdir),
          #end="")

    #ckpt = tf.train.get_checkpoint_state(logdir)
    #if ckpt:
        #print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        #global_step = int(ckpt.model_checkpoint_path
                          #.split('/')[-1]
                          #.split('-')[-1])
        #print("  Global step was: {}".format(global_step))
        #print("  Restoring...", end="")
        #saver.restore(sess, ckpt.model_checkpoint_path)
        #print(" Done.")
        #return global_step
    #else:
        #print(" No checkpoint found.")
        #return None

        
def lrelu(x, n, leak=0.2): 
    return tf.maximum(x, leak * x, name=n) 
 
def process_data(data_folder):   
    current_dir = os.getcwd()
    # parent = os.path.dirname(current_dir)
    pokemon_dir = os.path.join(current_dir, data_folder)     #'data'
    images = []
    for each in os.listdir(pokemon_dir):
        images.append(os.path.join(pokemon_dir,each))
    # print images    
    all_images = tf.convert_to_tensor(images, dtype = tf.string)
    
    images_queue = tf.train.slice_input_producer(
                                        [all_images])
                                        
    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels = CHANNEL)
    # sess1 = tf.Session()
    # print sess1.run(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta = 0.1)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    # noise = tf.Variable(tf.truncated_normal(shape = [HEIGHT,WIDTH,CHANNEL], dtype = tf.float32, stddev = 1e-3, name = 'noise')) 
    # print image.get_shape()
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT,WIDTH,CHANNEL])
    # image = image + noise
    # image = tf.transpose(image, perm=[2, 0, 1])
    # print image.get_shape()
    
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    iamges_batch = tf.train.shuffle_batch(
                                    [image], batch_size = BATCH_SIZE,
                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE,
                                    min_after_dequeue = 200)
    num_images = len(images)

    return iamges_batch, num_images

def generator(input, random_dim, is_train, reuse=False):
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32 # channel num
    s4 = 4
    output_dim = CHANNEL  # RGB image
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=[random_dim, s4 * s4 * c4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
         #Convolution, bias, activation, repeat! 
        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
        # 8*8*256
        #Convolution, bias, activation, repeat! 
        conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')
        # 16*16*128
        conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')
        # 32*32*64
        conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')
        # 64*64*32
        conv5 = tf.layers.conv2d_transpose(act4, c64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv5')
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
        act5 = tf.nn.relu(bn5, name='act5')
        
        #128*128*3
        conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv6')
        # bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')
        act6 = tf.nn.tanh(conv6, name='act6')
        return act6


def discriminator(input, is_train, reuse=False):
    c2, c4, c8, c16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 512
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()

        #Convolution, activation, bias, repeat!
        #  if feed in padded spectrogram, first mask out the zeros
        
        #zeros = tf.cast(tf.zeros_like(input), dtype=tf.bool)
        #ones = tf.cast(tf.ones_like(input), dtype=tf.bool)
        #loc = tf.where(input!=0, ones, zeros)
        #result = tf.boolean_mask(input, loc).reshape(128, -1)
        conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
        act1 = lrelu(conv1, n='act1')
         #Convolution, activation, bias, repeat! 
        conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = lrelu(bn2, n='act2')
        #Convolution, activation, bias, repeat! 
        conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = lrelu(bn3, n='act3')
         #Convolution, activation, bias, repeat! 
        conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = lrelu(bn4, n='act4')
       
        # start from act4
        dim = int(np.prod(act4.get_shape()[1:]))
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')
        
        
        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        # wgan just get rid of the sigmoid
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        # dcgan
        acted_out = tf.nn.sigmoid(logits)
        
        return logits #, acted_out



def train(data_folder):
    random_dim = 100
    x_placeholder = tf.placeholder("float", shape = [None,28,28,1], name='x_placeholder')
    
    print('CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])

    
    with tf.variable_scope('input'):
        #real and fake image placholders
        real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        #real_image = mnist.train.next_batch(BATCH_SIZE)[0].reshape([BATCH_SIZE, 28, 28, 1])
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')
    
    # wgan
    fake_image = generator(random_input, random_dim, is_train)
    
    real_result = discriminator(real_image, is_train)
    fake_result = discriminator(fake_image, is_train, reuse=True)
    
    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
    g_loss = - tf.reduce_mean(fake_result)  # This optimizes the generator.
            

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    # test
    # print(d_vars)   # lr = 2e-4
    trainer_d = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars)
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    # load data
    batch_size = BATCH_SIZE
    image_batch, samples_num = process_data(data_folder)
    
    batch_num = int(samples_num / batch_size)
    total_batch = 0
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # continue training
    save_path = saver.save(sess, "/tmp/model.ckpt")  # "model/newspectrogram/"
    ckpt = tf.train.latest_checkpoint(logdir + version)
    saver.restore(sess, save_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #print('total training sample num:%d' % samples_num)
    #print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, EPOCH))
    print('start training...')

    for i in range(EPOCH):
        t1 = time.time()
        print("epoch: ", i)
        for j in range(batch_num):
            #print("batch_num", j)
            d_iters = 3
            g_iters = 1

            train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            for k in range(d_iters):
                print("d_iters", k)
                #train_image = sess.run(image_batch)
                train_image = mnist.train.next_batch(BATCH_SIZE)[0].reshape([BATCH_SIZE, 28, 28, 1])
                #wgan clip weights
                sess.run(d_clip)
                
                # Update the discriminator
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})

            # Update the generator
            for k in range(g_iters):
                # train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
                _, gLoss = sess.run([trainer_g, g_loss],
                                    feed_dict={random_input: train_noise, is_train: True})

            print 'train:[%d/%d],d_loss:%f,g_loss:%f' % (i, j, dLoss, gLoss)
            
        # save check point every 500 epoch
        if i % save_every == 0:
            saver.save(sess, logdir +version + str(i))  
        if i % plot_every == 0:
            # save images
            sample_noise = np.random.uniform(-1.0, 1.0, size=[16, random_dim]).astype(np.float32)
            imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
            # imgtest = imgtest * 255.0
            # imgtest.astype(np.uint8)
            save_images(imgtest, [4,4] ,results_dir + '/{}_epoch'.format(i) + '.jpg')
            print(imgtest.shape)
            # D classify these images
            d_result = sess.run(fake_result, feed_dict={fake_image: imgtest, is_train: False})

            print "start evaluating"
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.title("score on generated images")
            #ipdb.set_trace()
            for j in range(16):
                ax1 = fig.add_subplot(4, 4, j+1)
                #ax1.set_axis_off()
                #fig.add_axes(ax1)
                im = imgtest[j, :, :, 0]
                plt.imshow(im.reshape([128, 128]), cmap='Greys')
                ax1.get_xaxis().set_ticks([])
                ax1.get_yaxis().set_ticks([])
                plt.ylabel("score={}".format(np.int(d_result[j]*10000)/10000.))
            plt.subplots_adjust(left=0.07, bottom=0.02, right=0.93, top=0.98,
                wspace=0.02, hspace=0.02)
            plt.savefig(results_dir + '/{}_epoch_D_sore_G'.format(i))
            plt.close()
            
            #print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
        print time.time() - t1
    coord.request_stop()
    coord.join(threads)


# def test():
    # random_dim = 100
    # with tf.variable_scope('input'):
        # real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        # random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        # is_train = tf.placeholder(tf.bool, name='is_train')
    
    # # wgan
    # fake_image = generator(random_input, random_dim, is_train)
    # real_result = discriminator(real_image, is_train)
    # fake_result = discriminator(fake_image, is_train, reuse=True)
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # variables_to_restore = slim.get_variables_to_restore(include=['gen'])
    # print(variables_to_restore)
    # saver = tf.train.Saver(variables_to_restore)
    # ckpt = tf.train.latest_checkpoint('./model/' + version)
    # saver.restore(sess, ckpt)


if __name__ == "__main__":
    data_dir = 'image_data/' + version + '/'
    train(data_dir)
# test()

## -*- coding: utf-8 -*-

## generate new kinds of pokemons

#import os
#import tensorflow as tf
#import numpy as np
#import cv2
#import random
#import scipy.misc
#from utils import *
#import ipdb
#import time
#slim = tf.contrib.slim

#HEIGHT, WIDTH, CHANNEL = 128, 128, 1      #  3
#BATCH_SIZE = 64
#EPOCH = 5000
#os.environ['CUDA_VISIBLE_DEVICES'] = '15'
#version = 'newSpectrogram'   #'newPokemon'
#results_dir = './' + version

#def lrelu(x, n, leak=0.2): 
    #return tf.maximum(x, leak * x, name=n) 

#def load(saver, sess, logdir):
    #print("Trying to restore saved checkpoints from {} ...".format(logdir))

    #ckpt = tf.train.get_checkpoint_state(logdir)
    #if ckpt:
        #print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        #global_step = int(ckpt.model_checkpoint_path
                          #.split('/')[-1]
                          #.split('-')[-1])
        #print("  Global step was: {}".format(global_step))
        #print("  Restoring...")        # , end=""
        #saver.restore(sess, ckpt.model_checkpoint_path)
        #print(" Done.")
        #return global_step
    #else:
        #print(" No checkpoint found.")
        #return None
        
#def process_data(data_folder):   
    #current_dir = os.getcwd()
    
    #print("current dir: ", current_dir)
    ## parent = os.path.dirname(current_dir)
    #pokemon_dir = os.path.join(current_dir, data_folder)
    #images = []
    #for each in os.listdir(pokemon_dir):
        #images.append(os.path.join(pokemon_dir,each))
    ##ipdb.set_trace()
    ##print(images)    
    #all_images = tf.convert_to_tensor(images, dtype = tf.string)
    
    #images_queue = tf.train.slice_input_producer(
                                        #[all_images])
                                        
    #content = tf.read_file(images_queue[0])
    #image = tf.image.decode_jpeg(content, channels = CHANNEL)
    ## sess1 = tf.Session()
    ## print sess1.run(image)
    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_brightness(image, max_delta = 0.1)
    #image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    ## noise = tf.Variable(tf.truncated_normal(shape = [HEIGHT,WIDTH,CHANNEL], dtype = tf.float32, stddev = 1e-3, name = 'noise')) 
    ## print image.get_shape()
    #size = [HEIGHT, WIDTH]
    #image = tf.image.resize_images(image, size)
    #image.set_shape([HEIGHT,WIDTH,CHANNEL])
    ## image = image + noise
    ## image = tf.transpose(image, perm=[2, 0, 1])
    ## print image.get_shape()
    
    #image = tf.cast(image, tf.float32)
    #image = image / 255.0
    ##ipdb.set_trace()
    #images_batch = tf.train.shuffle_batch(
                                    #[image], batch_size = BATCH_SIZE,
                                    #num_threads = 4, capacity = 200 + 3* BATCH_SIZE,
                                    #min_after_dequeue = 200)
    #num_images = len(images)

    #return images_batch, num_images

#def generator(input, random_dim, is_train, reuse=False):
    #c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32 # channel num
    #s4 = 4
    #output_dim = CHANNEL  # RGB image
    #with tf.variable_scope('gen') as scope:
        #if reuse:
            #scope.reuse_variables()
        #w1 = tf.get_variable('w1', shape=[random_dim, s4 * s4 * c4], dtype=tf.float32,
                             #initializer=tf.truncated_normal_initializer(stddev=0.02))
        #b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             #initializer=tf.constant_initializer(0.0))
        #flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
         ##Convolution, bias, activation, repeat! 
        #conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        #bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')
        #act1 = tf.nn.relu(bn1, name='act1')
        ## 8*8*256
        ##Convolution, bias, activation, repeat! 
        #conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           #kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           #name='conv2')
        #bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        #act2 = tf.nn.relu(bn2, name='act2')
        ## 16*16*128
        #conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           #kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           #name='conv3')
        #bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        #act3 = tf.nn.relu(bn3, name='act3')
        ## 32*32*64
        #conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           #kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           #name='conv4')
        #bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        #act4 = tf.nn.relu(bn4, name='act4')
        ## 64*64*32
        #conv5 = tf.layers.conv2d_transpose(act4, c64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           #kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           #name='conv5')
        #bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
        #act5 = tf.nn.relu(bn5, name='act5')
        
        ##128*128*3
        #conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           #kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           #name='conv6')
        ## bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')
        #act6 = tf.nn.tanh(conv6, name='act6')
        #return act6


#def discriminator(input, is_train, reuse=False):
    #c2, c4, c8, c16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 512
    #with tf.variable_scope('dis') as scope:
        #if reuse:
            #scope.reuse_variables()

        ##Convolution, activation, bias, repeat! 
        #conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 #kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 #name='conv1')
        #bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
        #act1 = lrelu(conv1, n='act1')
         ##Convolution, activation, bias, repeat! 
        #conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 #kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 #name='conv2')
        #bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        #act2 = lrelu(bn2, n='act2')
        ##Convolution, activation, bias, repeat! 
        #conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 #kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 #name='conv3')
        #bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        #act3 = lrelu(bn3, n='act3')
         ##Convolution, activation, bias, repeat! 
        #conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 #kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 #name='conv4')
        #bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        #act4 = lrelu(bn4, n='act4')
       
        ## start from act4
        #dim = int(np.prod(act4.get_shape()[1:]))
        #fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')
      
        
        #w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             #initializer=tf.truncated_normal_initializer(stddev=0.02))
        #b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             #initializer=tf.constant_initializer(0.0))

        ## wgan just get rid of the sigmoid
        #logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        ## dcgan
        #acted_out = tf.nn.sigmoid(logits)
        #return logits #, acted_out


#def train(data_folder):
    #random_dim = 100
    #print('CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    
    #with tf.variable_scope('input'):
        ##real and fake image placholders
        #real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        #random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        #is_train = tf.placeholder(tf.bool, name='is_train')
    
    ## wgan
    #fake_image = generator(random_input, random_dim, is_train)
    #real_result = discriminator(real_image, is_train)
    #fake_result = discriminator(fake_image, is_train, reuse=True)

    ## Define loss and optimizers
    #d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
    #g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.
    #tf.summary.scalar("d_loss", d_loss)
    #tf.summary.scalar("g_loss", g_loss)

    #t_vars = tf.trainable_variables()
    #d_vars = [var for var in t_vars if 'dis' in var.name]
    #g_vars = [var for var in t_vars if 'gen' in var.name]

    #trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(d_loss, var_list=d_vars)
    #trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)
    ## clip discriminator weights
    #d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    
    #batch_size = BATCH_SIZE
    #image_batch, samples_num = process_data(data_folder)
    
    #batch_num = int(samples_num / batch_size)
    #total_batch = 0
    #sess = tf.Session()
    #saver = tf.train.Saver()
    #sess.run(tf.global_variables_initializer())
    #sess.run(tf.local_variables_initializer())
    
    ## continue training
    #save_path = saver.save(sess, "/tmp/model.ckpt")
    #ckpt = tf.train.latest_checkpoint('./model/' + version)
    #saver.restore(sess, save_path)
    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #print('total training sample num:%d' % samples_num)
    #print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, EPOCH))
    #print('start training...')
    #for i in range(EPOCH):
        #t0 = time.time()
        #for j in range(batch_num):
            #print(j)
            #d_iters = 5
            #g_iters = 1

            #train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            #for k in range(d_iters):
                #print(k)
                #train_image = sess.run(image_batch)
                ##wgan clip weights
                #sess.run(d_clip)
                
                ## Update the discriminator
                #_, dLoss = sess.run([trainer_d, d_loss],
                                    #feed_dict={random_input: train_noise, real_image: train_image, is_train: True})
                
            ## Update the generator
            #for k in range(g_iters):
                ## train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
                #_, gLoss = sess.run([trainer_g, g_loss],
                                    #feed_dict={random_input: train_noise, is_train: True})

            ## print 'train:[%d/%d],d_loss:%f,g_loss:%f' % (i, j, dLoss, gLoss)
        #print("One epoch training time: {}".format(time.time() - t0))
        ## save check point every 500 epoch
        #if i%500 == 0:
            #if not os.path.exists('./model/' + version):
                #os.makedirs('./model/' + version)
            #saver.save(sess, './model/' +version + '/' + str(i))  
        #if i%50 == 0:
            ## save images
            #if not os.path.exists(results_dir):
                #os.makedirs(results_dir)
            #sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            #imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
            ## imgtest = imgtest * 255.0
            ## imgtest.astype(np.uint8)
            #save_images(imgtest, [8,8] ,results_dir + '/epoch' + str(i) + '.jpg')
            
            #print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
    #coord.request_stop()
    #coord.join(threads)


##def test():
    ##random_dim = 100
    ##with tf.variable_scope('input'):
        ##real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        ##random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        ##is_train = tf.placeholder(tf.bool, name='is_train')

    ### wgan
    ##fake_image = generator(random_input, random_dim, is_train)
    ##real_result = discriminator(real_image, is_train)
    ##fake_result = discriminator(fake_image, is_train, reuse=True)
    ##sess = tf.InteractiveSession()
    ##sess.run(tf.global_variables_initializer())
    ##variables_to_restore = slim.get_variables_to_restore(include=['gen'])
    ##print(variables_to_restore)
    ##saver = tf.train.Saver(variables_to_restore)

        
    ##save_path = saver.save(sess, "./newPokemon/model.ckpt")
    ##ckpt = tf.train.latest_checkpoint('./model' + version)
    ##saver.restore(sess, save_path)
    ##save_path = saver.save(sess, "./newPokemon/")
    ##ckpt = tf.train.latest_checkpoint('./newPokemon/' + version)
    ##saver.restore(sess, save_path)


#if __name__ == "__main__":
    #data_folder = 'data'               #"spectrogram256"
    #train(data_folder)
    ##test()

