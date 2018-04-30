
import tensorflow as tf
import configuration
import model
import argparse
import utils
import numpy as np
import scipy.misc
from tensorflow.examples.tutorials.mnist import input_data
import pylab as pl

def parse_arguments(parser):
	""" Parse input arguments.
		Args : Parser
		Output: Arguments
	"""

	parser.add_argument('--number_of_steps', type=int,default= 1000000, metavar='<number_of_steps>', help='Number of training steps.')
	parser.add_argument('--train_dir', type=str,default= "", metavar='<train_dir>', help='Directory for saving and loading model checkpoints.')
	args = parser.parse_args()
	return args

def plot_image(image, title, path):
    pl.figure()
    pl.imshow(image)
    pl.title(title)
    pl.savefig(path)

def main(unused_argv):
	# Parse arguments.
	parser = argparse.ArgumentParser()
	args = parse_arguments(parser)

	# Model configuration.
	model_config = configuration.ModelConfig()
	training_config = configuration.TrainingConfig()


	# Create training directory.
	train_dir = args.train_dir
	if not tf.gfile.IsDirectory(train_dir):
		tf.logging.info("Creating training directory: %s", train_dir)
		tf.gfile.MakeDirs(train_dir)

	# Load MNIST data.
	mnist = input_data.read_data_sets('MNIST')


	# Build the TensorFlow graph.
	g = tf.Graph()

	with g.as_default():

		# Build the model.
		the_model = model.DAE(model_config)
		the_model.build()

		# Set up the learning rate.
		learning_rate = tf.constant(training_config.learning_rate)


		# Set up the training ops.
		train_op = tf.contrib.layers.optimize_loss(
			loss=the_model.total_loss,
			global_step=the_model.global_step,
			learning_rate=learning_rate,
			optimizer=training_config.optimizer)

		# Set up the Saver for saving and restoring model checkpoints.
		saver = tf.train.Saver()



		# Run training.
		print("Training")

		with tf.Session() as sess:

			print("Initializing parameters")
			sess.run(tf.global_variables_initializer())

			for step in range(1, args.number_of_steps):

			    # Read batch.
			    batch = mnist.train.next_batch(model_config.batch_size)[0]

			    # Create a noisy version of the batch.
			    noisy_batch = utils.add_noise(batch)

			    # Prepare the dictionnary to feed the data to the graph.
			    feed_dict = {"images:0": batch, "noisy_images:0": noisy_batch, "phase_train:0": True}

			    # Run training
			    _,loss = sess.run([train_op,the_model.total_loss], feed_dict=feed_dict)


			    if step % 50 == 0:
			    	# Save checkpoint.
			        ave_path = saver.save(sess, train_dir+'/model.ckpt')

			        # Print Loss.
			        print("Step:", '%06d' % (step),"cost=", "{:.9f}".format(loss))

			print('Finished training ...')

			print('Start testing ...')

			# load batch.
			testing_data = mnist.test.images

			# Create a noisy version of the data.
			corrupted_testing = utils.add_noise(testing_data)

			# Prepare the dictionnary to feed the data to the graph.
			feed_dict = {"images:0": testing_data, "noisy_images:0": corrupted_testing, "phase_train:0": False}

    			# Compute the loss
			denoised_img,loss = sess.run([the_model.reconstructed_images,the_model.total_loss], feed_dict=feed_dict)
			for i in range(10):
				dim=np.sqrt(corrupted_testing[i].shape[0])
				dim=int(dim)
				org_img=np.reshape(corrupted_testing[i],(dim,dim))
				plot_image(org_img,'Original_Image'+str(i),'img/Original_Image'+str(i))
				deno_img=np.squeeze(denoised_img[i])
				denoised=np.reshape(deno_img,(dim,dim))
				plot_image(denoised,'denoised_img'+str(i),'img/Denoised_Image'+str(i))




			print(loss)

			print("Testing loss= ", loss)




if __name__ == "__main__":
	tf.app.run()
