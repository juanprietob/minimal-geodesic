
import numpy as np
import tensorflow as tf
from scipy.optimize import least_squares

class PNS:

	def __init__(self):
		self.sub_sphere = None
		self.dimension = 2

	# def GetRMatrix(self, a, b):

	# 	dim = a.shape[-1]
	# 	a = self.NormalizeVector(a)
	# 	a = np.reshape(a, [dim, 1])
	# 	b = self.NormalizeVector(b)
	# 	b = np.reshape(b, [dim, 1])
		
	# 	c = (b - np.matmul(a, np.matmul(a.T, b)))/np.linalg.norm(b - np.matmul(a, np.matmul(a.T, b)))
		
	# 	A = np.matmul(a, c.T) - np.matmul(c, a.T)

	# 	theta = np.arccos(np.matmul(a.T, b))
		
	# 	return np.identity(a.size) + np.sin(theta)*A + (np.cos(theta) - 1.)*(np.matmul(a, a.T) + np.matmul(c, c.T))

	# def NormalizeVector(self, x):
	# 	return np.array(x/np.linalg.norm(x))

	# def NormalizeVector(self, x):
	# 	return np.array(x/np.linalg.norm(x))

	# def GeodesicDistance(self, x, y):
	# 	x = self.NormalizeVector(x)
	# 	y = self.NormalizeVector(y)
	# 	return np.arccos(np.dot(x, y))

	def NormalizeVectors(self, x, axis=1):
		return tf.linalg.normalize(x, axis=axis)[0]

	def GeodesicDistance(self, x, y):
		x = tf.reshape(x, [tf.shape(x)[-1], 1])
		y = tf.reshape(y, [tf.shape(y)[-1], 1])
		return tf.math.acos(tf.matmul(tf.transpose(x),y)).numpy()

	def GetRMatrix(self, a, b):

		dim = tf.shape(a)[-1]
		a = tf.reshape(a, [dim, 1])
		b = tf.reshape(b, [dim, 1])
		
		c = (b - tf.matmul(a, tf.matmul(tf.transpose(a), b)))/tf.linalg.normalize(b - tf.matmul(a, tf.matmul(tf.transpose(a), b)))[1]
		
		A = tf.matmul(a, tf.transpose(c)) - tf.matmul(c, tf.transpose(a))

		theta = tf.math.acos(tf.matmul(tf.transpose(a), b))
		
		return tf.Variable(np.identity(dim)) + tf.math.sin(theta)*A + (tf.math.cos(theta) - 1.)*(tf.matmul(a, tf.transpose(a)) + tf.matmul(c, tf.transpose(c)))

	def BestCircle(self, x, points):
		circle_center = self.NormalizeVectors(x[0:-1], axis=0)
		angle_r = x[-1]
		return tf.reshape(tf.map_fn(lambda p: (self.GeodesicDistance(p, circle_center) - angle_r), points), [-1]).numpy()

	def CircleMean(self, x, points):
		return [(self.GeodesicDistance(p, x)) for p in points]

	def GetSubSphere(self, dimension):
		if(self.dimension == dimension):
			return self
		if(self.sub_sphere):
			return self.sub_sphere.GetSubSphere(dimension)
		return None

	def GetCircleCenter(self, dimension):
		if(self.dimension == dimension):
			return self.circle_center_v1
		if(self.sub_sphere):
			return self.sub_sphere.GetCircleCenter(dimension)
		return None

	def GetAngleR1(self, dimension):
		if(self.dimension == dimension):
			return self.angle_r1
		if(self.sub_sphere):
			return self.sub_sphere.GetAngleR1(dimension)
		return None

	def GetRotationMatrix(self, dimension):
		if(self.dimension == dimension):
			return self.rot_mat
		if(self.sub_sphere):
			return self.sub_sphere.GetRotationMatrix(dimension)
		return None

	def GetProjectedPointsAndResiduals(self, points):
		
		angle_r1 = self.angle_r1
		circle_center_v1 = self.circle_center_v1

		projected_points = tf.map_fn(lambda p: ((tf.math.sin(angle_r1)*p + tf.math.sin(self.GeodesicDistance(p, circle_center_v1) - angle_r1)*circle_center_v1)/tf.math.sin(self.GeodesicDistance(p, circle_center_v1))), points)
		projected_points = tf.reshape(projected_points, tf.shape(points))
		
		residuals = tf.map_fn(lambda p_pr: (self.GeodesicDistance(p_pr[0], p_pr[1])), tf.stack([points, projected_points], axis=1))
		residuals = tf.reshape(residuals, [tf.shape(points)[0], -1])

		return points, projected_points, residuals
		

	def GetFkPoints(self, points):
		angle_r1 = self.angle_r1
		rot_mat = self.rot_mat

		return self.NormalizeVectors(1./tf.math.sin(angle_r1) * tf.matmul(points, tf.transpose(rot_mat[0:-1,:])))

	def GetScores_r(self, dimension, points):
		if(self.dimension == dimension):
			return self.GetProjectedPointsAndResiduals(points)
		else:
			return self.sub_sphere.GetScores_r(dimension, self.GetFkPoints(points))

	def GetScores(self, dimension):
		return self.GetScores_r(dimension, self.points)

	def Fit(self, points):
		self.points = self.NormalizeVectors(points)
		self.Fit_r(self.points)

	def Fit_r(self, points):

		print("Processing sphere with points shape: ", points.shape)
		north = np.zeros(points.shape[-1])
		north[-1] = 1
		x0 = np.append(north, np.pi/4)

		print("Fitting best circle ...")
		res_lsq = least_squares(self.BestCircle, x0, args=(points,))
		circle_center_v1 = self.NormalizeVectors(res_lsq.x[0:-1], axis=0)
		angle_r1 = res_lsq.x[-1]

		rot_mat = self.GetRMatrix(circle_center_v1, north)
		
		self.dimension = points.shape[-1]
		self.circle_center_v1 = circle_center_v1
		self.angle_r1 = angle_r1
		self.rot_mat = rot_mat

		if(points.shape[-1] > 3):
			self.sub_sphere = PNS()
			self.sub_sphere.Fit_r(self.GetFkPoints(points))