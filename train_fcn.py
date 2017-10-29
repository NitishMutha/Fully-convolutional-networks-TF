import os
import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm
from datetime import datetime
import time

import fcn8_vgg as fcn8
import loss as cost
from dataGenerator import ImageDataGenerator
import constant

MODE = True
TENSORBOARD = True
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
NUMBER_CHANNELS = 3

LEARNING_RATE = 1e-6
EPOCHS = 5000000
BATCH_SIZE = 5
DISPLAY_STEP = 20
SAVE_STEP = 5000
EVALUATE_STE = 100

IMG_OUT_DIR = './outputImages/'
IMG_IN_DIR = './data/'
LOG_DIR = './logs/'
MODEL = './model/'
MODEL_NAME = 'fcn8_vgg_new'
DATASET = 'MITPlaces'
GPU = '0'

NUMBER_CLASSES = 151
IGNORE_LABEL = 255
KEEP_PROB = 0.5


def parseArguments():
    parser = argparse.ArgumentParser(description='Train setting for FCN8')
    parser.add_argument('-m', '--mode', action='store_true', default=MODE, dest='mode',
                        help='True: Train model, False: Test model')
    parser.add_argument('-t''--tensorboard', action='store_true', default=TENSORBOARD, dest='tensorboard',
                        help='Tensorboard logging and visualization')
    parser.add_argument("-v", "--verbose", action="store", type=int, dest="verbose", default=0,
                        help="Verbosity level")
    parser.add_argument("-c", "--clean", action="store_true", dest="clean", default=True,
                        help="Clean and train from scratch")
    parser.add_argument("--eval-no-img-save", action="store_true", dest="evaluateStepDontSaveImages",
                        default=False, help="Don't save images on evaluate step")
    parser.add_argument("--gpu", action="store", dest="gpu",
                        default=GPU, help="Select GPU for training")

    parser.add_argument('--data-dir', action='store', default=constant.DATA_DIR, dest='dataDir',
                        help='Root directory of dataset including train and test/validation')
    parser.add_argument('--split-data', action='store_true', default=False, dest='splitData',
                        help='Split data into train and test/validation')

    parser.add_argument('--dataset', action='store', default=DATASET,
                        choices=['PascalVOCContext', 'MITPlaces', 'PascalVOC', 'COCO'], dest='dataset',
                        help='Select dataset')
    parser.add_argument('--image-height', action='store', type=int, default=IMAGE_HEIGHT, dest='imageHeight',
                        help='Input image height feeding in network')
    parser.add_argument('--image-width', action='store', type=int, default=IMAGE_WIDTH, dest='imageWidth',
                        help='Input image width feeding in network')
    parser.add_argument("--image-channels", action="store", type=int, dest="imageChannels", default=NUMBER_CHANNELS,
                        help="Number of channels in image for feeding into the network")
    parser.add_argument("--random-fetch", action="store_true", dest="random", default=False,
                        help="Fetech random images for each batch")
    parser.add_argument('--input-text-present', action='store_true', default=True, dest='inputTextPresent',
                        help='Input text for loading images present')

    parser.add_argument('--learning-rate', action='store', type=float, default=LEARNING_RATE, dest='learningRate',
                        help='Learning rate of the model')
    parser.add_argument("--epochs", action="store", type=int, dest="trainingEpochs", default=EPOCHS,
                        help="Training epochs")
    parser.add_argument("--batchSize", action="store", type=int, dest="batchSize", default=BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--display-step", action="store", type=int, dest="displayStep", default=DISPLAY_STEP,
                        help="Progress display step")
    parser.add_argument("--save-step", action="store", type=int, dest="saveStep", default=SAVE_STEP,
                        help="Progress save step")
    parser.add_argument("--evaluate-step", action="store", type=int, dest="evaluateStep", default=EVALUATE_STE,
                        help="Progress evaluation step")

    parser.add_argument("--images-in-dir", action="store", dest="imagesInDir",
                        default=IMG_IN_DIR, help="Directory for list of input images and annotated labels")
    parser.add_argument("--pretrained-dir", action="store", dest="pretrained",
                        default=constant.PRETRAINED_ROOT+constant.VGG_PRETRAIN_MODEL, help="Path to the pretrained model to load weights")
    parser.add_argument("--images-out-dir", action="store", dest="imagesOutDir",
                        default=IMG_OUT_DIR, help="Directory for saving output images")
    parser.add_argument("--log-dir", action="store", dest="logsDir", default=LOG_DIR,
                        help="Directory for saving logs")
    parser.add_argument("--model-dir", action="store", dest="modelDir", default=MODEL,
                        help="Directory for saving the model")
    parser.add_argument("--model-name", action="store", dest="modelName", default=MODEL_NAME,
                        help="Name to be used for saving the model")

    parser.add_argument("--num-classes", action="store", type=int, dest="numClasses", default=NUMBER_CLASSES,
                        help="Number of classes")
    parser.add_argument("--ignore-label", action="store", type=int, dest="ignoreLabel", default=IGNORE_LABEL,
                        help="Label to ignore for loss computation")
    parser.add_argument("--keep-prob", action="store", type=float, dest="keepProb", default=KEEP_PROB,
                        help="Probability of keeping a neuron active during training")

    return parser.parse_args()


def trainModel(args):
    step = 1
    print('Train mode')
    with tf.variable_scope('FCN8_VGG'):

        trainDataGen = ImageDataGenerator(args, args.imagesInDir + 'train.txt',
                                          args.numClasses,
                                          'training',
                                          args.batchSize,
                                          num_preprocess_threads=8,
                                          shuffle=True,
                                          min_queue_examples=1000)

        images = trainDataGen.img_batch
        labels = trainDataGen.label_batch

        NUM_GPU = 1

        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=2 * NUM_GPU)

        image_batch, label_batch = batch_queue.dequeue()

    # tf.summary.image('Image', image_batch, max_outputs=10)
    tf.summary.image('label', label_batch, max_outputs=10)

    vgg_fcn = fcn8.FCN8VGG(enableTensorboard=args.tensorboard,
                           vgg16_npy_path=args.pretrained)

    with tf.name_scope('Model'):
        vgg_fcn.build(rgb=image_batch, keepProb=args.keepProb, num_classes=args.numClasses,
                      random_init_fc8=True, debug=(args.verbose > 0))

    with tf.name_scope('Loss'):
        # weights = tf.cast(batchInputLabels != args.ignoreLabel, dtype=tf.float32)
        loss = cost.loss(vgg_fcn.upscore32_pred, label_batch, trainDataGen.getAnnotationClasses())

    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learningRate)

        gradients = tf.gradients(loss, tf.trainable_variables())
        gradients = list(zip(gradients, tf.trainable_variables()))
        applyGradients = optimizer.apply_gradients(grads_and_vars=gradients)

    init = tf.global_variables_initializer()

    if args.tensorboard:
        tf.summary.scalar("loss", loss)
        if False:
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
            for grad, var in gradients:
                tf.summary.histogram(var.name + '/gradient', grad)
        mergedSummaryOp = tf.summary.merge_all()

    saver = tf.train.Saver()

    ###
    config = tf.ConfigProto(
        device_count={'GPU': 1}
    )

    with tf.Session() as sess:

        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        if args.clean:
            print("Removing previous checkpoints and logs")
            os.system("rm -rf " + args.logsDir)
            os.system("rm -rf " + args.imagesOutDir)
            os.system("rm -rf " + args.modelDir)
            os.system("mkdir " + args.imagesOutDir)
            os.system("mkdir " + args.modelDir)
        else:
            # Restore checkpoint
            print("Restoring from checkpoint")
            # saver = tf.train.import_meta_graph(args.modelDir + args.modelName + ".meta")
            saver.restore(sess, args.modelDir + args.modelName)

        img, img_l = sess.run([image_batch, label_batch])
        np.save('batch', [img])
        np.save('batch_l', [img_l])

        if args.tensorboard:
            # Op for writing logs to Tensorboard
            summaryWriter = tf.summary.FileWriter(args.logsDir, graph=tf.get_default_graph())

        print("Starting network training")

        # Keep training until reach max iterations
        for step in range(1, EPOCHS):
            print('Steps: ' + str(step))

            # Run optimization op (backprop)
            start_time = time.time()
            if args.tensorboard:
                _, summary = sess.run([applyGradients, mergedSummaryOp])
                summaryWriter.add_summary(summary, step)
            else:
                [trainLoss, _] = sess.run([loss, applyGradients])
                print("Iteration: %d, Minibatch Loss: %f" % (step, trainLoss))

                if (np.isnan(trainLoss)):
                    print("Nan reached. Terminating training.")
                    break
            duration = time.time() - start_time

            if step % 10 == 0:
                num_examples_per_step = args.batchSize
                examples_per_sec = num_examples_per_step / duration

                format_str = ('%s: step %d, (%.1f examples/sec')
                print(format_str % (datetime.now(), step,
                                    examples_per_sec))

            if step % args.saveStep == 0:
                # Save model weights to disk
                saver.save(sess, args.modelDir + args.modelName + str(step))
                print("######## Intermediate Model saved: %s" % (args.modelDir + args.modelName))

        # Save final model weights to disk
        saver.save(sess, args.modelDir + args.modelName)
        print("Final Model saved: %s" % (args.modelDir + args.modelName))

        print("Optimization Finished!")

    return


def testModel(args, inputLoader):
    print("Testing saved model")

    os.system("rm -rf " + args.imagesOutDir)
    os.system("mkdir " + args.imagesOutDir)

    args.imageHeight = 500
    args.imageWidth = 500

    # Now we make sure the variable is now a constant, and that the graph still produces the expected result.
    with tf.Session() as session:
        with tf.variable_scope('FCN8_VGG'):
            batchInputImages = tf.placeholder(dtype=tf.float32,
                                              shape=[None, args.imageHeight, args.imageWidth, args.imageChannels],
                                              name="batchInputImages")
            batchInputLabels = tf.placeholder(dtype=tf.float32,
                                              shape=[None, args.imageHeight, args.imageWidth, 1],
                                              name="batchInputLabels")
            keepProb = tf.placeholder(dtype=tf.float32, name="keepProb")

        vgg_fcn = fcn8.FCN8VGG(batchSize=args.batchSize, enableTensorboard=args.tensorboard,
                               vgg16_npy_path=args.pretrained)

        with tf.name_scope('Model'):
            vgg_fcn.build(rgb=batchInputImages, keepProb=keepProb, num_classes=args.numClasses,
                          random_init_fc8=True, debug=(args.verbose > 0))

        with tf.name_scope('Loss'):
            # weights = tf.cast(batchInputLabels != args.ignoreLabel, dtype=tf.float32)
            loss = cost.loss(vgg_fcn.upscore32_pred, batchInputLabels, inputLoader.getAnnotationClasses())

        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=args.learningRate)

            gradients = tf.gradients(loss, tf.trainable_variables())
            gradients = list(zip(gradients, tf.trainable_variables()))
            applyGradients = optimizer.apply_gradients(grads_and_vars=gradients)

        # saver = tf.train.import_meta_graph(args.modelDir + args.modelName + ".meta")
        saver = tf.train.Saver()
        saver.restore(session, args.modelDir + args.modelName)

        # Get reference to placeholders
        # outputNode = session.graph.get_tensor_by_name("Model/probabilities:0")
        # inputBatchImages = session.graph.get_tensor_by_name("FCN8_VGG/batchInputImages:0")
        # inputKeepProbability = session.graph.get_tensor_by_name("FCN8_VGG/keepProb:0")

        # Sample 50 test batches
        args.batchSize = 1  # 50
        numBatch = 8
        for i in tqdm(range(1, numBatch), desc='Testing'):
            # print("Processing batch # %d" % i)
            batchImagesTest, _ = inputLoader.getTestBatch(readMask=False)  # For testing without GT mask
            imagesProbabilityMap = session.run(vgg_fcn.probabilities,
                                               feed_dict={batchInputImages: batchImagesTest, keepProb: 1.0})
            # Save image results
            print("Saving images...")
            inputLoader.saveLastBatchResults(imagesProbabilityMap, isTrain=False)

    print("Model tested!")
    return


def main1():
    # read parameters and defaults
    args = parseArguments()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.mode:
        trainModel(args)
    else:
        testModel(args)


if __name__ == '__main__':
    main1()
