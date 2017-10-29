import tensorflow as tf
from utils import *

def loss(logits, labels, classes):
    with tf.name_scope('loss'):
        valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(
            annotation_batch_tensor=tf.squeeze(labels, 3), logits_batch_tensor=logits, class_labels=classes)

        softmax = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                          labels=valid_labels_batch_tensor)
        loss = tf.reduce_mean(softmax, name='total_loss')
    return loss
