import tensorflow as tf
import numpy as np
from scipy import stats
from fcn8_vgg import FCN8VGG

# Create following folders
PRETRAINED_ROOT = './pretrainedWeightsRepo/'
TRAINED_FCN8_MODEL = 'segmentation_trained/fcn8_vgg_new51000'
PRETRAINED_VGG = 'vgg16/vgg16.npy'


class SegmentationGenerator():
    def __init__(self, input_shape, sess=None):
        self.seg_sess = sess
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.channels = input_shape[3]

        # self.saver = tf.train.import_meta_graph(TRAINED_FCN8_MODEL + ".meta")
        # self.fcn8_graph = tf.get_default_graph()
        self.graph = tf.Graph()
        with tf.device('/gpu:0'):
            with self.graph.as_default():
                self.batchInputImages = tf.placeholder(dtype=tf.float32,
                                                       shape=[None, self.height, self.width, self.channels],
                                                       name="batchInputImages")

                vgg_fcn = FCN8VGG(vgg16_npy_path=PRETRAINED_ROOT + PRETRAINED_VGG)

                with tf.name_scope('Model'):
                    vgg_fcn.build(rgb=self.batchInputImages, keepProb=1, random_init_fc8=True)
                self.outputNode = vgg_fcn.probabilities
                self.saver = tf.train.Saver()
            if sess is None:
                # todo remove this later
                config = tf.ConfigProto(
                    device_count={'GPU': 0}
                )

                self.seg_sess = tf.Session(graph=self.graph, config=config)

            self.saver.restore(sess=self.seg_sess, save_path=PRETRAINED_ROOT + TRAINED_FCN8_MODEL)

    def getSegmentation(self, batch_images, smashed=True):
        seg_images = self.seg_sess.run(self.outputNode, feed_dict={self.batchInputImages: batch_images})

        if smashed:
            def map_channels(i_x):
                i, x = i_x
                x = (x * 255).astype(np.uint8)
                if x.max() > 0.35 * 255:
                    threshold = np.fabs(x.max() - x.max() * .65)
                else:
                    threshold = 255
                threshImage = stats.threshold(x, threshmin=threshold)
                threshImage[threshImage > 0] = i
                return threshImage

            def smash_channels(channels):
                base = channels[0]
                for i, x in enumerate(channels):
                    base[x > 0] = i
                return base

            imgchannels = list(map(map_channels, enumerate(np.transpose(seg_images[0], [2, 0, 1]))))
            smashed = smash_channels(imgchannels)
            return smashed

        return seg_images


def generate():
    with tf.Session() as sess:
        SegmentationGenerator([1, 350, 350, 3], sess)


if __name__ == '__main__':
    generate()
