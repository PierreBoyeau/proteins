import tensorflow as tf

from riken.protein_io import prot_features
from riken.protein_io.prot_features import chars
from tensorboard.plugins import projector


class SimilarityModel:
    def __init__(self, inputs, labels, margin, n_classes, lstm_size=16,
                 optimizer=tf.train.AdamOptimizer(), batch_size=128, predict=False):
        self.lstm_size = lstm_size
        self.cell_fn = tf.nn.rnn_cell.LSTMCell
        self.margin = margin
        self.loss = None
        self.optimizer_fn = optimizer
        self.optimizer = None
        self.init_op = None
        self.batch_size = batch_size
        self.n_classes = n_classes

        with tf.variable_scope('vector_representation'):
            self.vectors = self.to_vector_representation(inputs=inputs['tokens'],
                                                         pssm_input=inputs['pssm_li'])
            # self.vectors = tf.nn.l2_normalize(self.vectors, axis=1, name='l2_norm')
        if not predict:
            self.get_triplets_loss(inputs, labels)

    def get_triplets_loss(self, inputs, labels):
        """
        USING L2 FOR NOW
        :param inputs:
        :param labels:
        :return:
        """
        with tf.variable_scope('masks'):
            p_mask, n_mask = self.masks(labels)

        with tf.variable_scope('hard_triplet_search'):
            norms = tf.reduce_sum(self.vectors * self.vectors, 1)
            # L2 : compute pairwise matrix
            norms_mat = tf.tile(tf.expand_dims(norms, axis=1), multiples=[1, norms.get_shape()[0]],
                                name='get_norm_matrix')
            pairwise_dists = norms_mat \
                             - 2*tf.matmul(a=self.vectors, b=self.vectors, transpose_b=True) \
                             + tf.transpose(norms_mat)
            pairwise_dists = tf.maximum(pairwise_dists, 0.0, name='get_good_pairwise_dists')

            # L1: compute pairwise matrix
            # norms = tf.expand_dims(norms, axis=1)
            # prod_norms_mat = 1e-16 + tf.matmul(a=norms, b=norms, transpose_b=True)
            # prod_norms_mat = tf.sqrt(prod_norms_mat)
            # p_scal_mat = tf.matmul(a=self.vectors, b=self.vectors, transpose_b=True)
            # pairwise_dists = 1.0 - tf.divide(p_scal_mat, prod_norms_mat)

            tf.summary.histogram('pairwise_distances', pairwise_dists)

            p_dists = tf.multiply(p_mask, pairwise_dists, name='pos_mask_mul')
            n_dists = tf.multiply(n_mask, pairwise_dists, name='neg_mask_mul')
            p_count = tf.reduce_sum(p_mask, axis=1, keepdims=True)
            n_count = tf.reduce_sum(n_mask, axis=1, keepdims=True)

            # More complicated here as we need the smallest non null negative distance value later.
            # n_bias = tf.multiply(1 - n_mask, tf.reduce_max(n_dists), 'negative_bias')
            # n_dists = n_dists + n_bias
            # p_dist = tf.reduce_max(p_dists, axis=1, name='get_hard_pos_dist', keepdims=True)
            # n_dist = tf.reduce_min(n_dists, axis=1, name='get_hard_neg_dist', keepdims=True)

            p_dist = tf.reduce_sum(p_dists, axis=1, name='get_pos_dist', keepdims=True) \
                        / p_count
            n_dist = tf.reduce_sum(n_dists, axis=1, name='get_neg_dist', keepdims=True) \
                        / n_count

        tf.summary.histogram('positive_distances', p_dist)
        tf.summary.histogram('negative_distances', n_dist)

        tf.summary.scalar('positive_distances', tf.reduce_mean(p_dist))
        tf.summary.scalar('negative_distances', tf.reduce_mean(n_dist))

        dist_diff = tf.maximum(p_dist - n_dist + self.margin * tf.ones(p_dist.get_shape()), 0.0,
                               name='max_with_zero')

        self.loss = tf.reduce_mean(dist_diff)
        # valid_triplets = tf.to_float(tf.greater(dist_diff, 1e-16))
        # num_positive_triplets = tf.reduce_sum(valid_triplets)
        # self.loss = tf.reduce_sum(dist_diff) / (num_positive_triplets + 1e-16)

    def masks(self, labels):
        onehot_labels = tf.one_hot(labels, depth=self.n_classes)
        same_labels_mat = tf.matmul(a=onehot_labels, b=onehot_labels, transpose_b=True)

        pos_mask = same_labels_mat - tf.eye(num_rows=self.batch_size)
        neg_mask = tf.ones(shape=[self.batch_size, self.batch_size]) - same_labels_mat
        return pos_mask, neg_mask

    def build(self):
        self.optimizer = self.optimizer_fn.minimize(self.loss,
                                                    global_step=tf.train.get_global_step())
        self.init_op = tf.initialize_all_variables()

        return

    def to_vector_representation(self, inputs, pssm_input):
        # embed = tf.one_hot(inputs, depth=len(chars))
        # static_feat_mat = tf.Variable(
        #     # initial_value=prot_features.create_blosom_80_mat(),
        #     initial_value=prot_features.create_overall_static_aa_mat(normalize=True),
        #     dtype=tf.float32, trainable=False, name='static_feat_mat')
        #
        # inputs_replace_m1 = tf.nn.relu(inputs)
        # # Create indexes where -1 values (fill) are replaced by 0
        # aa_static_feat = tf.nn.embedding_lookup(params=static_feat_mat, ids=inputs_replace_m1)
        #
        # h = tf.concat([
        #     embed,
        #     aa_static_feat,
        #     pssm_input
        # ], axis=2, name='all_features')

        # Only properties
        static_feat_mat = tf.Variable(
            # initial_value=prot_features.create_blosom_80_mat(),
            initial_value=prot_features.create_overall_static_aa_mat_feature_selection(normalize=True),
            dtype=tf.float32, trainable=False, name='static_feat_mat')
        inputs_replace_m1 = tf.nn.relu(inputs)
        # Create indexes where -1 values (fill) are replaced by 0
        h = tf.nn.embedding_lookup(params=static_feat_mat, ids=inputs_replace_m1)

        scalar_rep = tf.layers.dense(h, units=1, activation=tf.nn.relu, name='feature_importances')

        scalar_rep = tf.layers.conv1d(scalar_rep, filters=10, kernel_size=3, padding='same',
                                      activation=tf.nn.relu)
        scalar_rep = tf.layers.dropout(scalar_rep, rate=0.3)

        tf.summary.histogram('post_conv1d', scalar_rep)

        # fw_lstm = self.cell_fn(num_units=self.lstm_size)
        # bw_lstm = self.cell_fn(num_units=self.lstm_size)
        # outputs, state = tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, scalar_rep, dtype=tf.float32)
        # outputs = tf.concat(outputs, 2)
        outputs, state = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.lstm_size,
                                                        direction='bidirectional')(scalar_rep)

        attention = tf.layers.Dense(1)(outputs)
        attention = tf.squeeze(attention, axis=2)
        attention = tf.nn.softmax(attention, axis=1)
        attention = tf.tile(tf.expand_dims(attention, axis=2), multiples=[1, 1, 2 * self.lstm_size])
        last_output = tf.multiply(outputs, attention)
        last_output = tf.reduce_sum(last_output, axis=1, name='representation')
        # last_output = tf.layers.dense(last_output, units=10, kernel_initializer=tf.initializers.random_uniform(-1, 1))
        # representations = tf.Variable(last_output)

        return last_output
