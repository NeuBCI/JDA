import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # specify which GPU to use
import tensorflow as tf
import numpy as np
import cPickle as pkl
from sklearn.manifold import TSNE
import tensorflow.contrib.slim as slim
from flip_gradient import flip_gradient
from utils import *

batch_size = 96
log_dir = './tensorBoardLog'

def variable_summaries(var):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var)) 
      tf.summary.histogram('histogram', var)

class EmotionModel(object):
    def __init__(self):
        self._build_model()
            
    def add_average(self, variable):
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.ema.apply([variable]))
        average_variable = tf.identity(self.ema.average(variable), name=variable.name[:-2] + '_avg')
        return average_variable
    
    def create_walk_statistic(self, p_aba, equality_matrix):
        per_row_accuracy = 1.0 - tf.reduce_sum((equality_matrix * p_aba), 1)**0.5
        estimate_error = tf.reduce_mean(1.0 - per_row_accuracy, name=p_aba.name[:-2] + '_esterr')
        self.add_average(estimate_error)
        self.add_average(p_aba)
    
    def _build_model(self):
        self.X = tf.placeholder(tf.float32, [None, 310])
        self.y = tf.placeholder(tf.float32, [None, 3])
        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.train = tf.placeholder(tf.bool, [])
        self.l = tf.placeholder(tf.float32, [])
        self.step = slim.get_or_create_global_step()
        self.ema = tf.train.ExponentialMovingAverage(0.99, self.step)
        
        with tf.variable_scope('feature_extractor'):
            self.W1 = tf.Variable(tf.truncated_normal([310, 128], stddev = 0.1))
            self.b1 = tf.Variable(tf.constant(0.1, shape = [128]))
            self.fc1 = tf.nn.relu(tf.matmul(self.X, self.W1) + self.b1)
            self.feature = tf.nn.dropout(self.fc1, 1.0)
            variable_summaries(self.W1) 
            variable_summaries(self.b1) 
            tf.summary.histogram('activations', self.fc1)
            
        with tf.variable_scope('label_predictor'):
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0], [batch_size/2, -1])
            classify_feats = tf.cond(self.train, source_features, all_features)
            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0, 0], [batch_size/2, -1])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)
            self.W2 = tf.Variable(tf.truncated_normal([128, 3], stddev = 0.1))
            self.b2 = tf.Variable(tf.constant(0.1, shape = [3]))
            self.logits = tf.matmul(classify_feats, self.W2) + self.b2
            variable_summaries(self.W2)
            variable_summaries(self.b2)
            tf.summary.histogram('logits', self.logits)
            
            self.pred = tf.nn.softmax(self.logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits = tf.slice(self.logits, [0, 0], [batch_size / 2, -1]),
                                                                     labels = tf.slice(self.y, [0, 0], [batch_size/2, -1]))
            # create walker loss
            y_sparse = tf.argmax(input=tf.slice(self.y, [0, 0], [batch_size / 2, -1]), dimension=1)
            equality_matrix = tf.equal(tf.reshape(y_sparse, [-1, 1]), y_sparse)
            equality_matrix = tf.cast(equality_matrix, tf.float32)
            p_target = (equality_matrix / tf.reduce_sum(equality_matrix, [1], keep_dims=True))
            match_ab = tf.matmul(tf.slice(self.feature, [0, 0], [batch_size / 2, -1]),
                                 tf.slice(self.feature, [batch_size / 2, 0], [batch_size / 2, -1]),
                                 transpose_b=True, name='match_ab')
            p_ab = tf.nn.softmax(match_ab, name='p_ab')
            p_ba = tf.nn.softmax(tf.transpose(match_ab), name='p_ba')
            p_aba = tf.matmul(p_ab, p_ba, name='p_aba')
            self.create_walk_statistic(p_aba, equality_matrix)
            self.walker_loss = tf.losses.softmax_cross_entropy(p_target, tf.log(1e-8 + p_aba),  scope='walker_loss')
            # create visit loss
            visit_probability = tf.reduce_mean(p_ab, [0], keep_dims=True, name='visit_prob')
            t_nb = tf.shape(p_ab)[1]
            self.visit_loss = tf.losses.softmax_cross_entropy(tf.fill([1, t_nb], 1.0 / tf.cast(t_nb, tf.float32)),
                                                              tf.log(1e-8 + visit_probability), scope='loss_visit')     
        
        with tf.variable_scope('domain_predictor'):
            feat = flip_gradient(self.feature, self.l)
            self.W3 = tf.Variable(tf.truncated_normal([128, 2], stddev = 0.1))
            self.b3 = tf.Variable(tf.constant(0.1, shape = [2]))
            self.domain_pred = tf.nn.softmax(tf.matmul(feat, self.W3) + self.b3)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits = self.domain_pred, labels = self.domain)
            
graph = tf.get_default_graph()

with graph.as_default():
    model = EmotionModel()
    learning_rate = tf.placeholder(tf.float32, [])
    regularizer = tf.contrib.layers.l2_regularizer(0.001)(model.W1) + tf.contrib.layers.l2_regularizer(0.001)(model.W2) + tf.contrib.layers.l2_regularizer(0.001)(model.W3)
    
    pred_loss = tf.reduce_mean(model.pred_loss)
    walker_loss = tf.reduce_mean(model.walker_loss)
    visit_loss = tf.reduce_mean(model.visit_loss)
    assoc_loss = walker_loss + 0.6*visit_loss # hyperparameter
    domain_loss = tf.reduce_mean(model.domain_loss)
    
    A_loss = tf.reduce_mean(pred_loss + regularizer)
    B_loss = tf.reduce_mean(pred_loss + assoc_loss + regularizer)
    C_loss = tf.reduce_mean(pred_loss + domain_loss +regularizer)
    D_loss = tf.reduce_mean(pred_loss + assoc_loss + domain_loss + regularizer)
    '''
    A_op = tf.train.AdamOptimizer(learning_rate).minimize(A_loss)
    B_op = tf.train.AdamOptimizer(learning_rate).minimize(B_loss)
    C_op = tf.train.AdamOptimizer(learning_rate).minimize(C_loss)
    D_op = tf.train.AdamOptimizer(learning_rate).minimize(D_loss)
    '''
    A_op = tf.train.AdamOptimizer(1e-4).minimize(A_loss)
    B_op = tf.train.AdamOptimizer(1e-4).minimize(B_loss)
    C_op = tf.train.AdamOptimizer(1e-4).minimize(C_loss)
    D_op = tf.train.AdamOptimizer(1e-4).minimize(D_loss)
    
    # Evaluation
    correct_label_pred = tf.equal(tf.argmax(model.y, 1), tf.argmax(model.pred, 1))
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
    domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

tf.summary.scalar('B_loss', B_loss)
tf.summary.scalar('accuracy', label_acc)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir + '/train', tf.get_default_graph())
test_writer = tf.summary.FileWriter(log_dir + '/test')

num_steps = 10000

def train_and_evaluate(training_mode, graph, model, verbose=False):
    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        domain_labels = np.vstack([np.tile([1., 0.], [batch_size / 2, 1]),
                                   np.tile([0., 1.], [batch_size / 2, 1])])
        # Training loop
        for i in range(num_steps):
            p = float(i) / num_steps
            l = 2. / (1. + np.exp(-10. * p)) - 1
            lr = 0.01 / (1. + 10 * p)**0.75
            # no adaptation
            if training_mode == 'A':
                X, y = batch_generator_balance([source, source_label], batch_size, 3)
                _, loss, training_acc, summary = sess.run([A_op, A_loss, label_acc, merged], feed_dict={model.X:X, model.y:y, model.train:False, model.l:l, learning_rate:lr})
                if verbose and i % 100 == 0:
                    target_acc = sess.run(label_acc, feed_dict={model.X:target, model.y:target_label, model.train:False})
                    print 'loss: %f  training_acc: %f  test_acc: %f' % (loss, training_acc, target_acc)
            # associate domain adaptation                
            elif training_mode == 'B':
                X0, y0 = batch_generator_balance([source, source_label], batch_size/2, 3)
                X1, y1 = batch_generator_balance([target, target_label], batch_size/2, 3)
                X = np.vstack([X0, X1])
                y = np.vstack([y0, y1])
                _, loss, training_acc, summary_train = sess.run([B_op, B_loss, label_acc, merged], feed_dict={model.X:X, model.y:y, model.train:False, model.l:l, learning_rate:lr})
                train_writer.add_summary(summary_train, i)
                if verbose and i % 100 == 0:
                    target_acc, summary_test = sess.run([label_acc, merged], feed_dict={model.X:target, model.y:target_label, model.train:False})
                    test_writer.add_summary(summary_test, i)
                    print 'loss: %f  training_acc: %f  test_acc: %f' % (loss, training_acc, target_acc)
            # adversarial domian adaptation
            elif training_mode == 'C':
                X0, y0 = batch_generator_balance([source, source_label], batch_size/2, 3)
                X1, y1 = batch_generator_balance([target, target_label], batch_size/2, 3)
                X = np.vstack([X0, X1])
                y = np.vstack([y0, y1])
                _, loss, training_acc, adversarial_acc, summary = sess.run([C_op, C_loss, label_acc, domain_acc, merged], feed_dict={model.X:X, model.y:y, model.train:False, 
                                                                           model.domain:domain_labels, model.l:l, learning_rate:lr})
                train_writer.add_summary(summary, i)
                if verbose and i % 100 == 0:
                    target_acc, summary = sess.run([label_acc, merged], feed_dict={model.X:target, model.y:target_label, model.train:False})
                    test_writer.add_summary(summary, i)
                    print 'loss: %f  training_acc: %f  test_acc: %f  domian_acc: %f' % (loss, training_acc, target_acc, adversarial_acc)
            # associate and adversarial adaptation                
            elif training_mode == 'D':
                X0, y0 = batch_generator_balance([source, source_label], batch_size/2, 3)
                X1, y1 = batch_generator_balance([target, target_label], batch_size/2, 3)
                X = np.vstack([X0, X1])
                y = np.vstack([y0, y1])
                _, loss, training_acc, adversarial_acc, summary = sess.run([D_op, D_loss, label_acc, domain_acc, merged], feed_dict={model.X:X, model.y:y, model.train:False,
                                                                           model.domain:domain_labels, model.l:l, learning_rate:lr})
                train_writer.add_summary(summary, i)
                if verbose and i % 100 == 0:
                    target_acc = sess.run(label_acc, feed_dict={model.X:target, model.y:target_label, model.train:False})
                    print 'loss: %f  training_acc: %f  test_acc: %f  domian_acc: %f' % (loss, training_acc, target_acc, adversarial_acc)
                    test_writer.add_summary(summary, i)
                       
        source_acc = sess.run(label_acc, feed_dict={model.X:source, model.y:source_label, model.train:False})
        target_acc = sess.run(label_acc, feed_dict={model.X:target, model.y:target_label, model.train:False})        
        #test_domain_acc = sess.run(domain_acc, feed_dict={model.X:combined_test_imgs, model.domain:combined_test_domain, model.l:1.0})
        combined_test_imgs = np.vstack([source[:250], target[:250]])
        test_emb_middle, test_emb_last = sess.run([model.feature, model.logits], feed_dict={model.X:combined_test_imgs, model.train:False}) 
    return source_acc, target_acc


# source, source_label, target, target_label, target_train, target_test, target_label_train, target_label_test = load_data_subject(1)
source, source_label, target, target_label = load_data_session('djc')

source_acc, target_acc, test_emb_middle, test_emb_last = train_and_evaluate('D', graph, model, verbose=True)
print 'Source accuracy:', source_acc
print 'Target accuracy:', target_acc



# Create a mixed dataset for TSNE visualization
combined_test_labels = np.vstack([source_label[:250], target_label[:250]])
combined_test_domain = np.vstack([np.tile([1., 0.], [250, 1]), np.tile([0., 1.], [250, 1])])

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
dann_tsne_middle = tsne.fit_transform(test_emb_middle)

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
dann_tsne_last   = tsne.fit_transform(test_emb_last)
        
plot_embedding(dann_tsne_middle, combined_test_labels.argmax(1), combined_test_domain.argmax(1), 'middle')
plot_embedding(dann_tsne_last, combined_test_labels.argmax(1), combined_test_domain.argmax(1), 'last')  


'''
# subject to subject
for subjectIndex in range(14):
    source, source_label, target, target_label = load_data_subject(subjectIndex)
    source_acc, target_acc = train_and_evaluate('A', graph, model, verbose=False)
    print 'Source accuracy:', source_acc
    print 'Target accuracy:', target_acc
print 'fine'

for subjectIndex in range(14):
    source, source_label, target, target_label = load_data_subject(subjectIndex)
    source_acc, target_acc = train_and_evaluate('B', graph, model, verbose=False)
    print 'Source accuracy:', source_acc
    print 'Target accuracy:', target_acc
print 'fine'

for subjectIndex in range(14):
    source, source_label, target, target_label = load_data_subject(subjectIndex)
    source_acc, target_acc = train_and_evaluate('C', graph, model, verbose=False)
    print 'Source accuracy:', source_acc
    print 'Target accuracy:', target_acc
print 'fine'

for subjectIndex in range(14):
    source, source_label, target, target_label = load_data_subject(subjectIndex)
    source_acc, target_acc = train_and_evaluate('D', graph, model, verbose=False)
    print 'Source accuracy:', source_acc
    print 'Target accuracy:', target_acc
print 'fine'
'''


'''
# session to session
nameList = ['djc', 'jj', 'lqj', 'ly', 'mhw', 'phl', 'sxy', 'wk', 'wsf', 'ww', 'wyw', 'xyl', 'ys', 'zjy']

for item in nameList:
    source, source_label, target, target_label = load_data_session(item)
    source_acc, target_acc = train_and_evaluate('A', graph, model, verbose=False)
    print 'Source accuracy:', source_acc
    print 'Target accuracy:', target_acc
print 'fine'
    
for item in nameList:
    source, source_label, target, target_label = load_data_session(item)
    source_acc, target_acc = train_and_evaluate('B', graph, model, verbose=False)
    print 'Source accuracy:', source_acc
    print 'Target accuracy:', target_acc
print 'fine'
    
for item in nameList:
    source, source_label, target, target_label = load_data_session(item)
    source_acc, target_acc = train_and_evaluate('C', graph, model, verbose=False)
    print 'Source accuracy:', source_acc
    print 'Target accuracy:', target_acc
print 'fine'

for item in nameList:
    source, source_label, target, target_label = load_data_session(item)
    source_acc, target_acc = train_and_evaluate('D', graph, model, verbose=False)
    print 'Source accuracy:', source_acc
    print 'Target accuracy:', target_acc   
print 'fine'
'''

