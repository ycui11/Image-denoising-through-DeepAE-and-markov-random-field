#########################################################################################################################
# Author: Safa Messaoud                                                                                                 #
# E-Mail: messaou2@illinois.edu                                                                                         #
# Instituation: University of Illinois at Urbana-Champaign  															#
# Course: ECE 544_na Fall 2017                                                            								#
# Date: July 2017                                                                                                   	#
#                                                                                                                       #
# Description: Script to evaluate the model. Run the evaluation script in a separate process in parallel 				#
# with the training Script.																								#
#																														#
#########################################################################################################################


import math
import os.path
import time
import argparse
import numpy as np
import tensorflow as tf
import configuration
import model
import utils
from tensorflow.examples.tutorials.mnist import input_data


def evaluate_model(data, sess, model, global_step, num_eval_examples):
	"""Computes the cost associated with the model.
	Args:
		data: pointer to the MNIST data
	    sess: Session object.
	    model: Instance of DAE; the model to evaluate.
	    global_step: Global step of the model checkpoint.
	    num_eval_examples: Number of examples to run the evaluation on.
	"""	  

	# Determine the number of batches to run the evaluation.
	num_eval_batches = int(math.ceil(num_eval_examples / model.config.batch_size))  

	# Initialise the loss.
	sum_losses = 0.


	for i in range(num_eval_batches):
		
		# Read batch.
		batch = data.validation.next_batch(model.config.batch_size)[0]

		# Create a noisy version of the batch.
		noisy_batch = utils.add_noise(batch)

		# Prepare the dictionnary to feed the data to the graph.
		feed_dict = {"images:0": batch, "noisy_images:0": noisy_batch, "phase_train:0": False}

		# Evaluate the loss.
		loss = sess.run([model.total_loss], feed_dict=feed_dict)
		sum_losses += np.sum(loss)

	sum_losses=sum_losses/num_eval_batches
	    

	print("Step:", '%06d' % (global_step),",cost=", "{:.9f}".format(sum_losses))  



def run_once(data, model, saver, checkpoint_dir, min_global_step,num_eval_examples):
	"""Evaluates the latest model checkpoint.
	Args:
		data: A pointer to the MNIST data.
		model: Instance of DAE; the model to evaluate.
		saver: Instance of tf.train.Saver for restoring model Variables.
		checkpoint_dir: Directory containing model checkpoints.    
		min_global_step: Number of steps until the first evaluation.
		num_eval_examples: Number of examples to run the evaluation on.
	"""

	# Name of the global step.
	global_step_name=model.global_step.name

	# Path to the latest checkpoint directory.
	model_path = tf.train.latest_checkpoint(checkpoint_dir)

	with tf.Session() as sess:
		# Load model from checkpoint.
		saver.restore(sess, model_path)

		# Load the global step.
		global_step = tf.train.global_step(sess, global_step_name)

		# Check if it is time to run the evaluation
		if global_step < min_global_step:
			print("Skipping evaluation. No checkpoint found ")
			return

		# Start the queue runners.
		#coord = tf.train.Coordinator()
		#threads = tf.train.start_queue_runners(coord=coord)

		# Run evaluation on the latest checkpoint.	
		evaluate_model(
			data=data,
			num_eval_examples=num_eval_examples,
			sess=sess,
			model=model,
			global_step=global_step)
		

		#coord.request_stop()
		#coord.join(threads, stop_grace_period_secs=10)


def run(data, checkpoint_dir, eval_interval_secs, min_global_step, num_eval_examples):
	"""Runs evaluation in a loop.
	Args:
		data: a pointer to teh MNIST data
		checkpoint_dir: Directory containing model checkpoints.    
		eval_interval_secs: Interval between consecutive evaluations.
		min_global_step: Number of steps until the first evaluation.
		num_eval_examples: Number of examples to run the evaluation on.
	"""
	g = tf.Graph()

	with g.as_default():
		# Build the model for evaluation.
		model_config = configuration.ModelConfig()
		the_model = model.DAE(model_config)
		the_model.build()
		

		# Create the Saver to restore model Variables.
		saver = tf.train.Saver()

		g.finalize()

		# Run a new evaluation run every eval_interval_secs.
		while True:
			start = time.time()

			# Run evaluation.
			run_once(data, the_model, saver, checkpoint_dir, min_global_step, num_eval_examples)

			time_to_next_eval = start + eval_interval_secs - time.time()

			# Wait until the time to next evaluation elapses
			if time_to_next_eval > 0:
				time.sleep(time_to_next_eval)



def parse_arguments(parser):
	"""Parse input arguments."""

	parser.add_argument('--checkpoint_dir', type=str,default= '', metavar='<checkpoint_dir>', help='Directory containing model checkpoints.')	
	parser.add_argument('--eval_interval_secs', type=int,default= 600, metavar='<eval_interval_secs>', help='Interval between evaluation runs.')	
	parser.add_argument('--min_global_step', type=int,default= 50, metavar='<min_global_step>', help='Minimum global step to run evaluation.')	
	parser.add_argument('--num_eval_examples', type=int,default= 500, metavar='<num_eval_examples>', help='Number of examples for evaluation.')	

	args = parser.parse_args()
	return args


def main(_):

	# Parse input arguments.
	parser = argparse.ArgumentParser()
	args = parse_arguments(parser)

	# Load the MNIST dataset
	mnist = input_data.read_data_sets('MNIST')

	# Run the evaluation script.
	run(mnist, args.checkpoint_dir, args.eval_interval_secs, args.min_global_step, args.num_eval_examples)


if __name__ == "__main__":
	tf.app.run()

