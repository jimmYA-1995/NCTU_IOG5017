import sys
import os
import argparse
from tempfile import gettempdir
import urllib
import zipfile

import tensorflow as tf
import numpy as np
import collections
import random
import math
import matplotlib.pyplot as plt

from tensorflow.contrib.tensorboard.plugins import projector

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log-dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for Tensorboard summaries')

FLAGS, unparsed = parser.parse_known_args()

# Create dir for Tensorboard if there is not
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

# Step 1 download the data
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it is the right size"""
    local_filename = os.path.join(gettempdir(), filename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url + filename, local_filename)
    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + local_filename +
                        '. Can you get to it with a browser?')
    return local_filename

filename = maybe_download('text8.zip', 31344016)

# Read the data into a list of strings.


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list og words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulary = read_data(filename)
print("Data size: ", len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token
vocabulary_size = 50000


def build_dataset(words, n_words):
    """Process inputs into a dataset.

       Arguments:
       words: word list to count
       n_words: most common "n" words
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)  # juat code from 1, 2, 3, ...
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

# count, dictionary, reversed_dictionary only have most_common words
data, count, dictionary, reversed_dictionary = build_dataset(vocabulary, vocabulary_size)

# Hint to reduce memory
del vocabulary
print("Most common words(+UNK)", count[:5])
print('Smaple data', data[:10], [reversed_dictionary[i] for i in data[:10]])

data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= (2 * skip_window)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index: data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        #                           (population, k length of list)
        words_to_use = random.sample(context_words, num_skips)
        # print(words_to_use)
        for j, context_words in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_words]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
# for i in range(8):
#     print(batch[i], reversed_dictionary[batch[i]], '->', labels[i, 0],
#           reversed_dictionary[labels[i, 0]])


# Step 4: Build and train a skip-gram model


batch_size = 128
embedding_size = 128  # Dimension of the embedding vector
skip_window = 1  # How many words to consider left and right
num_skip = 2  # How many times to reuse an input to create a label
num_sampled = 64  # Number of negative examples to sample.

valid_size = 16  # Random set of words to evaluate similarity on
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()

with graph.as_default():

    # Input data.
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinnedto the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embedding for inputs.
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
            )
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal(
                    [vocabulary_size, embedding_size],
                    stddev=1.0 / math.sqrt(embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate he loss.
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))

    # Add the loss value as a scalar to summary
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between mini-batch example and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Merge all summaries
    merged = tf.summary.merge_all()

    # Add global initializer
    init = tf.global_variables_initializer()

    # Create a saver
    saver = tf.train.Saver()

# Step 5: begin training
num_steps = 100001

with tf.Session(graph=graph) as session:
    # Open a writer to write summary
    writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

    # We must initialize all variables before we use them
    init.run()
    print('Initialized')

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skip, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # Define metadata variable.
        run_metadata = tf.RunMetadata()

        _, summary, loss_val = session.run(
            [optimizer, merged, loss],
            feed_dict=feed_dict,
            run_metadata=run_metadata)
        # Feed meradata for visualizing the graph in the tensorboard
        average_loss += loss_val

        # Add return summaries to writer in each step.
        writer.add_summary(summary, step)
        # Add metadata to visualize the graph for the last run.
        if step == (num_steps - 1):
            writer.add_run_metadata(run_metadata, 'step{}'.format(step))

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
        # The average loss is an estimate of the loss over the last 2000 batches.
        print('Average loss at step ', step, ': ', average_loss)
        average_loss = 0

        # Note that this is expensive (~20% slowdown if compute every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reversed_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to {}:'.format(valid_word)
                for k in range(top_k):
                    close_word = reversed_dictionary[nearest[k]]
                    log_str = '{} {}'.format(log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

    # Write corresponding labels for the embeddings.
    with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
        for i in range(vocabulary_size):
            f.write(reversed_dictionary[i] + '\n')

    # Save the model for checkpoint
    saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))

    # Create a configuration for visualizing embeddings with the labels in Tensorboard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

writer.close()

# Step 6: Visualize the embeddings.


# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')

    plt.savefig(filename)

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE(
        perplexity=30,
        n_components=2,
        init='pca',
        n_iter=5000,
        method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reversed_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    print(ex)
