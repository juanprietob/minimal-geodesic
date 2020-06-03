
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import time
import os
import json

class CircleMean(tf.keras.Model):
	def __init__(self):
		super(CircleMean, self).__init__()

	def Fit(self, points):

		dim = tf.shape(points)[-1]
		north = np.zeros(dim)
		north[-1] = 1
		north = tf.constant(north, dtype=tf.float32)

		self.optimizer = tf.keras.optimizers.SGD(1e-2)

		self.model = tf.keras.Sequential([
			layers.Dense(1, use_bias=False, input_shape=(dim,), activation=tf.math.acos, kernel_initializer=tf.constant_initializer(north.numpy()), kernel_constraint=tf.keras.constraints.UnitNorm())
			])

		start = time.time()

		loss_min = 9999999
		did_not_improve = 0

		for epoch in range(10000):
			step = 0
			avg_loss = 0
			for points_ds in dataset:
				avg_loss += self.train_step(points_ds)
				step += 1
			avg_loss /= step

			if(avg_loss < loss_min):
				loss_min = avg_loss
				did_not_improve = 0
				print("min loss:", loss_min.numpy())
			else:
				did_not_improve += 1

			if(did_not_improve > 50):
				break

		print("Circle mean fitting took", time.time() - start)

		mean_vector_v1 = tf.cast(tf.reshape(self.model.layers[0].get_weights()[0], [-1]), dtype=tf.float32)
		return mean_vector_v1

	@tf.function
	def train_step(self, points):

		with tf.GradientTape() as tape:

			logits = self.model(points)
			loss = tf.reduce_mean(tf.math.abs(logits))
			# loss = tf.reduce_mean(logits*logits)
			
			# loss = mse(logits, self.angle_r1)

			var_list = self.trainable_variables

			gradients = tape.gradient(loss, var_list)
			self.optimizer.apply_gradients(zip(gradients, var_list))

			return loss

class SubSphere(tf.keras.Model):

	def __init__(self, dimension=2):
		super(SubSphere, self).__init__()
		self.output_dir = "./out"

	@tf.function
	def NormalizeVectors(self, x, axis=1):
		return tf.linalg.normalize(x, axis=axis)[0]

	@tf.function
	def GeodesicDistance(self, x, y):
		return tf.math.acos(tf.reduce_sum(tf.multiply(x, y)))

	@tf.function
	def GetRMatrix(self, a, b):

		dim = tf.shape(a)[-1]
		a = tf.reshape(a, [dim, 1])
		b = tf.reshape(b, [dim, 1])
		
		c = (b - tf.matmul(a, tf.matmul(tf.transpose(a), b)))/tf.linalg.normalize(b - tf.matmul(a, tf.matmul(tf.transpose(a), b)))[1]
		
		A = tf.matmul(a, tf.transpose(c)) - tf.matmul(c, tf.transpose(a))

		theta = tf.math.acos(tf.matmul(tf.transpose(a), b))
		
		return tf.eye(dim) + tf.math.sin(theta)*A + (tf.math.cos(theta) - 1.)*(tf.matmul(a, tf.transpose(a)) + tf.matmul(c, tf.transpose(c)))
	
	@tf.function
	def GetProjectedPoints(self, points, angle_r1, circle_center_v1):
		circle_center_v1 = tf.reshape(circle_center_v1, shape=(tf.shape(circle_center_v1)[0], 1))
		geo_d = tf.math.acos(tf.matmul(points, circle_center_v1))
		return tf.math.divide(tf.math.sin(angle_r1)*points + tf.matmul(tf.math.sin(geo_d - angle_r1), tf.transpose(circle_center_v1)), tf.math.sin(geo_d))

	@tf.function
	def GetFkPoints(self, points, angle_r1, rot_mat):
		return 1./tf.math.sin(angle_r1) * tf.matmul(points, tf.transpose(rot_mat[0:-1,:]))

	def Fit(self, points):

		print("Processing sphere with points shape: ", points.shape)

		dim = tf.shape(points)[-1]

		north = np.zeros(dim)
		north[-1] = 1
		north = tf.constant(north, dtype=tf.float32)

		mean_points, variance_points = tf.nn.moments(points, axes=[0])
		mean_points = self.NormalizeVectors(mean_points, 0)
		# x0 = np.append(north, np.pi/4)

		print("Fitting best circle ...")

		points = self.NormalizeVectors(points)
		dataset = tf.data.Dataset.from_tensor_slices((points,))
		batch_size = 50
		dataset = dataset.shuffle(10).batch(batch_size)


		self.angle_r1 = tf.Variable(self.GeodesicDistance(mean_points, north).numpy())

		lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-2, tf.shape(points)[0]/batch_size, 0.96, 1)
		self.optimizer = tf.keras.optimizers.SGD(lr)

		self.model = tf.keras.Sequential([
			layers.Dense(1, use_bias=False, input_shape=(dim,), activation=tf.math.acos, kernel_initializer=tf.constant_initializer(north.numpy()), kernel_constraint=tf.keras.constraints.UnitNorm())
			])
		
		start = time.time()

		for epoch in range(50):
			step = 0
			avg_loss = 0
			for points_ds in dataset:
				avg_loss += self.train_step(points_ds)
				step += 1
			avg_loss /= step

			print("loss:", avg_loss.numpy())

		print("Sphere fitting took", time.time() - start)

		circle_center_v1 = tf.cast(tf.reshape(self.model.layers[0].get_weights()[0], [-1]), dtype=tf.float32)
		
		rot_mat = self.GetRMatrix(circle_center_v1, north)

		tf.keras.backend.clear_session()

		projected_points = self.GetProjectedPoints(points, self.angle_r1, circle_center_v1)
		return self.NormalizeVectors(self.GetFkPoints(projected_points, self.angle_r1, rot_mat)), circle_center_v1, self.angle_r1

	@tf.function
	def train_step(self, points):

		mae = tf.keras.losses.MeanAbsoluteError()
		# mse = tf.keras.losses.MeanSquaredError()

		with tf.GradientTape() as tape:

			logits = self.model(points)
			loss = mae(logits, self.angle_r1)
			# loss = mse(logits, self.angle_r1)

			var_list = self.trainable_variables

			gradients = tape.gradient(loss, var_list)
			self.optimizer.apply_gradients(zip(gradients, var_list))

			return loss

class PNS(tf.keras.Model):

	def __init__(self):
		super(PNS, self).__init__()

		self.output_dir = "./out"
		self.sub_sphere = SubSphere()

	def GeodesicDistance(self, x, y):
		return self.sub_sphere.GeodesicDistance(x, y)

	def GetRMatrix(self, a, b):
		return self.sub_sphere.GetRMatrix(a, b)

	def NormalizeVectors(self, x, axis=1):
		return self.sub_sphere.NormalizeVectors(x, axis)

	def GetProjectedPoints(self, points):

		dim = tf.shape(points)[-1].numpy()
		circle_center_v1, angle_r1, rot_mat = self.GetSubSphereFit(dim)
		return self.sub_sphere.GetProjectedPoints(points, angle_r1, circle_center_v1)

	def GetFkPoints(self, last_dim = 2):

		points = self.points
		dimension = tf.shape(points)[-1]

		for d in range(dimension, last_dim, -1):
			circle_center_v1, angle_r1, rot_mat = self.GetSubSphereFit(d)
			projected_points = self.sub_sphere.GetProjectedPoints(points, angle_r1, circle_center_v1)
			points = self.NormalizeVectors(self.sub_sphere.GetFkPoints(projected_points, angle_r1, rot_mat))

		return points

	def GetFk_1Points(self, points):

		dim = tf.shape(points)[-1].numpy()
		circle_center_v1, angle_r1, rot_mat = self.GetSubSphereFit(dim + 1)

		x_t = tf.math.multiply(points, tf.math.sin(angle_r1))
		x_t = tf.concat([x_t, tf.ones([tf.shape(points)[0], 1])], axis=-1)

		x_t_cos = np.zeros([tf.shape(points)[0].numpy()])
		x_t_cos[-1] = tf.math.cos(angle_r1)
		x_t_cos = tf.cast(tf.reshape(tf.constant(x_t_cos), [tf.shape(points)[0], 1]), dtype=tf.float32)

		return self.NormalizeVectors(tf.matmul(x_t, rot_mat) + x_t_cos)

	def SetOutputDirectory(self, out):
		self.output_dir = out	

	def Fit(self, points, continue_training=False):

		self.points = points
		dimension = tf.shape(points)[-1]

		if(continue_training):
			min_dim = dimension
			for d in range(dimension.numpy(), 2, -1):
				fit_path = os.path.join(self.output_dir, "sphere_fit_" + str(d) + ".json") 
				if(os.path.exists(fit_path)):
					min_dim -= 1

			dimension = min_dim
			if(dimension != min_dim):
				points = self.GetFkPoints(dimension + 1)

		for dim in range(dimension, 2, -1):
			sub_sphere = SubSphere()
			points, circle_center_v1, angle_r1 = sub_sphere.Fit(points)
			self.SaveFit(circle_center_v1, angle_r1, dim)

	def GetSubSphereFit(self, dimension):
		fit_path = os.path.join(self.output_dir, "sphere_fit_" + str(dimension) + ".json") 
		with open(fit_path, 'r') as f:
		    fit_obj = json.loads(f.read())

		    circle_center_v1 = np.array(fit_obj["circle_center_v1"])
		    angle_r1 = np.array(fit_obj["angle_r1"])

		    dim = tf.shape(circle_center_v1)[-1]

		    north = np.zeros(dim)
		    north[-1] = 1
		    north = tf.constant(north, dtype=tf.float32)

		    rot_mat = self.GetRMatrix(tf.constant(circle_center_v1, dtype=tf.float32), north)

		    return tf.cast(circle_center_v1, dtype=tf.float32), tf.cast(angle_r1, dtype=tf.float32), tf.cast(rot_mat, dtype=tf.float32)
		
	def SaveFit(self, circle_center_v1, angle_r1, dimension):

		if not os.path.isdir(self.output_dir):
			os.makedirs(self.output_dir)
		
		out_dict = {}
		out_dict["circle_center_v1"] = circle_center_v1.numpy().tolist()
		out_dict["angle_r1"] = float(angle_r1.numpy())

		with open(os.path.join(self.output_dir, "sphere_fit_" + str(dimension) + ".json"), 'w') as f:
		    json.dump(out_dict, f)