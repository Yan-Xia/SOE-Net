import tensorflow as tf
import numpy as np
import math
import sys
import os

import tf_util
from transform_nets import input_transform_net, feature_transform_net
from pointnet_util import pointnet_sa_module, pointnet_fp_module, pointnet_sa_module_msg, pointSIFT_res_module, pointSIFT_module
import tensorflow.contrib.slim as slim

def placeholder_inputs(batch_num_queries, num_pointclouds_per_query, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_num_queries, num_pointclouds_per_query, num_point, 3))
    return pointclouds_pl

#Adopted from the original pointnet code
def forward(point_cloud, is_training, bn_decay=None):
    """PointNetVLAD,    INPUT is batch_num_queries X num_pointclouds_per_query X num_points_per_pointcloud X 3, 
                        OUTPUT batch_num_queries X num_pointclouds_per_query X output_dim """
    batch_num_queries = point_cloud.get_shape()[0].value
    num_pointclouds_per_query = point_cloud.get_shape()[1].value
    num_points = point_cloud.get_shape()[2].value
    CLUSTER_SIZE=64
    OUTPUT_DIM=256
    point_cloud = tf.reshape(point_cloud, [batch_num_queries*num_pointclouds_per_query, num_points,3])

    point_cloud_xyz = point_cloud

    _, c0_l0_points, _ = pointSIFT_module(point_cloud_xyz, None, radius=0.1, out_channel=64, is_training=is_training, bn_decay=bn_decay, scope='layer0_c0')
    net0 = tf_util.conv2d(c0_l0_points, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net0 = tf.squeeze(net0, [2])

    _, net1, _ = pointSIFT_module(point_cloud_xyz, net0, radius=0.1, out_channel=64, is_training=is_training, bn_decay=bn_decay, scope='layer0_c1')

    net1 = tf_util.conv2d(net1, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    net1 = tf.squeeze(net1, [2]) 

    _, net2, _ = pointSIFT_module(point_cloud_xyz, net1, radius=0.25, out_channel=128, is_training=is_training, bn_decay=bn_decay, scope='layer0_c2')
    net2 = tf_util.conv2d(net2, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net2 = tf.squeeze(net2, [2]) 

    _, net3, _ = pointSIFT_module(point_cloud_xyz, net2, radius=0.5, out_channel=256, is_training=is_training, bn_decay=bn_decay, scope='layer0_c3')
    net = tf_util.conv2d(net3, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
 
    print ('net:', net)

    net= tf.reshape(net,[-1,1024])
    net = tf.nn.l2_normalize(net,1)

    output = vlad_forward(point_cloud_xyz, net, max_samples=num_points, is_training=is_training)


    #normalize to have norm 1
    output = tf.nn.l2_normalize(output,1)
    output =  tf.reshape(output,[batch_num_queries,num_pointclouds_per_query,OUTPUT_DIM])

    return output


def best_pos_distance(query, pos_vecs):
    with tf.name_scope('best_pos_distance') as scope:
        #batch = query.get_shape()[0]
        num_pos = pos_vecs.get_shape()[1]
        query_copies = tf.tile(query, [1,int(num_pos),1]) #shape num_pos x output_dim
        best_pos=tf.reduce_max(tf.reduce_sum(tf.squared_difference(pos_vecs,query_copies),2),1)
        return best_pos



##########Losses for PointNetVLAD###########

def SARE_loss(q_vec, pos_vecs, neg_vecs):
    num_pos = pos_vecs.get_shape()[1]
    query_copies_p = tf.tile(q_vec, [1, int(num_pos), 1])
    num_neg = neg_vecs.get_shape()[1]
    dif_p = -tf.reduce_sum(tf.squared_difference(pos_vecs, query_copies_p), 2)
    print('dif_p', dif_p)
    p_exp = tf.reduce_sum(tf.exp(dif_p), 1)
    #p_exp = tf.reduce_sum(p_exp, 1)
    print('p_exp', p_exp)
    query_copies_n = tf.tile(q_vec, [1, int(num_neg), 1])
    dif_n = -tf.reduce_sum(tf.squared_difference(neg_vecs, query_copies_n), 2)
    print('dif_p', dif_n)
    n_exp = tf.reduce_sum(tf.exp(dif_n), 1)
    #n_exp = tf.reduce_sum(n_exp, 1)
    print('n_exp', n_exp)
    loss = tf.reduce_sum(-tf.log(tf.div(p_exp, (p_exp + n_exp))))
    return loss



#Returns average loss across the query tuples in a batch, loss in each is the average loss of the definite negatives against the best positive
def triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
     # ''', end_points, reg_weight=0.001):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m=tf.fill([int(batch), int(num_neg)],margin)
    triplet_loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))
    return triplet_loss

#Lazy variant
def lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m=tf.fill([int(batch), int(num_neg)],margin)
    triplet_loss=tf.reduce_mean(tf.reduce_max(tf.maximum(tf.add(m,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))
    return triplet_loss


def softmargin_loss(q_vec, pos_vecs, neg_vecs):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    ones=tf.fill([int(batch), int(num_neg)],1.0)
    soft_loss=tf.reduce_mean(tf.reduce_sum(tf.log(tf.exp(tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2)))+1.0),1))
    return soft_loss

def lazy_softmargin_loss(q_vec, pos_vecs, neg_vecs):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    ones=tf.fill([int(batch), int(num_neg)],1.0)
    soft_loss=tf.reduce_mean(tf.reduce_max(tf.log(tf.exp(tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2)))+1.0),1))
    return soft_loss

def quadruplet_loss_sm(q_vec, pos_vecs, neg_vecs, other_neg, m2):
    soft_loss= softmargin_loss(q_vec, pos_vecs, neg_vecs)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    total_loss= soft_loss+second_loss

    return total_loss   

def lazy_quadruplet_loss_sm(q_vec, pos_vecs, neg_vecs, other_neg, m2):
    soft_loss= lazy_softmargin_loss(q_vec, pos_vecs, neg_vecs)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_max(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    total_loss= soft_loss+second_loss

    return total_loss   

def quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2):
    trip_loss= triplet_loss(q_vec, pos_vecs, neg_vecs, m1)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    total_loss= trip_loss+second_loss

    return total_loss 

def lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2):
    trip_loss= lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, m1)

    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_max(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    '''
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.001

    reg_loss = sum(reg_losses)*reg_constant
    '''
    #tv = tf.trainable_variables()
    '''
    for v in tv:
        print(type(v))
    '''
    #regularization_cost = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

    total_loss= trip_loss+second_loss

    return total_loss


def HPHN_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m2):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    hard_neg = tf.reduce_min(tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2), 1)

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    hard_other_neg = tf.reduce_min(tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2), 1)

    final_hard_neg = tf.minimum(hard_neg, hard_other_neg)
    loss=tf.reduce_mean(tf.maximum(tf.add(m2,tf.subtract(best_pos, final_hard_neg)), tf.zeros(1)))
    return loss

def vlad_forward(xyz, reshaped_input, feature_size=1024, max_samples=4096, cluster_size=64,
                output_dim=256, gating=True, add_batch_norm=True,
                is_training=True, bn_decay=None):
    """Forward pass of a NetVLAD block.

    Args:
    reshaped_input: If your input is in that form:
    'batch_size' x 'max_samples' x 'feature_size'
    It should be reshaped in the following form:
    'batch_size*max_samples' x 'feature_size'
    by performing:
    reshaped_input = tf.reshape(input, [-1, features_size])

    Returns:
    vlad: the pooled vector of size: 'batch_size' x 'output_dim'
    """
    input = tf.reshape(reshaped_input, [-1,
                                    max_samples, 1, feature_size])
    reshaped_input_pointwise = attention_unit(input, is_training=is_training)
    
    reshaped_input = tf.reshape(reshaped_input_pointwise, [-1, feature_size])
    #msg grouping
 
    # print('m:', m)

    cluster_weights = tf.get_variable("cluster_weights",
                                      [feature_size, cluster_size],
                                      initializer=tf.random_normal_initializer(
                                          stddev=1 / math.sqrt(feature_size)))

    activation = tf.matmul(reshaped_input, cluster_weights)

    if add_batch_norm:
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="cluster_bn", fused=False)
    else:
        cluster_biases = tf.get_variable("cluster_biases",
                                         [cluster_size],
                                         initializer=tf.random_normal_initializer(
                                             stddev=1 / math.sqrt(feature_size)))
        activation = activation + cluster_biases

    activation = tf.nn.softmax(activation)

    activation = tf.reshape(activation,
                            [-1, max_samples, cluster_size])

    a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

    cluster_weights2 = tf.get_variable("cluster_weights2",
                                       [1, feature_size, cluster_size],
                                       initializer=tf.random_normal_initializer(
                                           stddev=1 / math.sqrt(feature_size)))

    a = tf.multiply(a_sum, cluster_weights2)

    activation = tf.transpose(activation, perm=[0, 2, 1])

    reshaped_input = tf.reshape(reshaped_input, [-1,
                                                 max_samples, feature_size])

    vlad = tf.matmul(activation, reshaped_input)
    vlad = tf.transpose(vlad, perm=[0, 2, 1])
    vlad = tf.subtract(vlad, a)

    vlad = tf.nn.l2_normalize(vlad, 1)

    vlad = tf.reshape(vlad, [-1, cluster_size * feature_size])
    vlad = tf.nn.l2_normalize(vlad, 1)

    hidden1_weights = tf.get_variable("hidden1_weights",
                                      [cluster_size * feature_size, output_dim],
                                      initializer=tf.random_normal_initializer(
                                          stddev=1 / math.sqrt(cluster_size)))

    ##Tried using dropout
    # vlad=tf.layers.dropout(vlad,rate=0.5,training=self.is_training)

    vlad = tf.matmul(vlad, hidden1_weights)

    ##Added a batch norm
    vlad = tf.contrib.layers.batch_norm(vlad,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope='bn')

    if gating:
        vlad = context_gating(vlad, add_batch_norm, is_training)

    return vlad

def context_gating(input_layer, add_batch_norm=True, is_training=True):
    """Context Gating

    Args:
    input_layer: Input layer in the following shape:
    'batch_size' x 'number_of_activation'

    Returns:
    activation: gated layer in the following shape:
    'batch_size' x 'number_of_activation'
    """

    input_dim = input_layer.get_shape().as_list()[1]

    gating_weights = tf.get_variable("gating_weights",
                                     [input_dim, input_dim],
                                     initializer=tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(input_dim)))

    gates = tf.matmul(input_layer, gating_weights)

    if add_batch_norm:
        gates = slim.batch_norm(
            gates,
            center=True,
            scale=True,
            is_training=is_training,
            scope="gating_bn")
    else:
        gating_biases = tf.get_variable("gating_biases",
                                        [input_dim],
                                        initializer=tf.random_normal(stddev=1 / math.sqrt(input_dim)))
        gates = gates + gating_biases

    gates = tf.sigmoid(gates)

    activation = tf.multiply(input_layer, gates)

    return activation

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def attention_unit(inputs, scope='attention_unit',is_training=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        dim = inputs.get_shape()[-1].value
        layer = dim//4
        f = tf_util.conv2d(inputs,layer, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=False, is_training=is_training,
                              scope='conv_f', bn_decay=None)

        g = tf_util.conv2d(inputs, layer, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_g', bn_decay=None)

        h = tf_util.conv2d(inputs, dim, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_h', bn_decay=None)


        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))   # [bs, N, N]*[bs, N, c]->[bs, N, c]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=inputs.shape)  # [bs, h, w, C]
        x = gamma * o + inputs

    return x
