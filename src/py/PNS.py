
import numpy as np
from scipy.optimize import least_squares

class PNS:

	def __init__(self):
		self.sub_sphere = None
		self.dimension = 2

	def GetRMatrix(self, a, b):

		dim = a.shape[-1]
		a = self.NormalizeVector(a)
		a = np.reshape(a, [dim, 1])
		b = self.NormalizeVector(b)
		b = np.reshape(b, [dim, 1])
		
		c = (b - np.matmul(a, np.matmul(a.T, b)))/np.linalg.norm(b - np.matmul(a, np.matmul(a.T, b)))
		
		A = np.matmul(a, c.T) - np.matmul(c, a.T)

		theta = np.arccos(np.matmul(a.T, b))
		
		return np.identity(a.size) + np.sin(theta)*A + (np.cos(theta) - 1.)*(np.matmul(a, a.T) + np.matmul(c, c.T))

	def GeodesicDistance(self, x, y):
		x = self.NormalizeVector(x)
		y = self.NormalizeVector(y)
		return np.arccos(np.dot(x, y))

	def BestCircle(self, x, points):
		circle_center = x[0:-1]
		angle_r = x[-1]
		return [(self.GeodesicDistance(p, circle_center) - angle_r) for p in points]

	def CircleMean(self, x, points):
		return [(self.GeodesicDistance(p, x)) for p in points]

	def NormalizeVector(self, x):
		return np.array(x/np.linalg.norm(x))

	def GetSubSphere(self, dimension):
		if(self.dimension == dimension):
			return self
		if(self.sub_sphere):
			return self.sub_sphere.GetSubSphere(dimension)
		return None

	def GetPoints(self, dimension):
		if(self.dimension == dimension):
			return self.points
		if(self.sub_sphere):
			return self.sub_sphere.GetPoints(dimension)
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

	def GetProjectedPoints(self, dimension):
		if(self.dimension == dimension):
			return self.projected_points
		if(self.sub_sphere):
			return self.sub_sphere.GetProjectedPoints(dimension)
		return None

	def Fit(self, points):

		
		print("Processing sphere with points shape: ", points.shape)

		north = np.zeros(points.shape[-1])
		north[-1] = 1
		x0 = np.append(north, np.pi)

		print("Fitting best circle ...")
		res_lsq = least_squares(self.BestCircle, x0, args=(points,))
		circle_center_v1 = self.NormalizeVector(res_lsq.x[0:-1])
		angle_r1 = res_lsq.x[-1]

		rot_mat = self.GetRMatrix(np.array(circle_center_v1), north)

		projected_points = []
		residuals = []
		fk_points = []
		fk_points_magnitude = []

		print("Projecting...")
		for i, p in enumerate(points):
			pr_p = (np.sin(angle_r1)*p + np.sin(self.GeodesicDistance(p, circle_center_v1) - angle_r1)*circle_center_v1)/np.sin(self.GeodesicDistance(p, circle_center_v1))

			projected_points.append(pr_p)
			residuals.append(self.GeodesicDistance(p, pr_p))

			fk_p = (1./np.sin(angle_r1)) * np.matmul(rot_mat[0:-1,:], p)

			fk_mag = np.linalg.norm(fk_p)
			fk_points_magnitude.append(fk_mag)
			fk_points.append(fk_p/fk_mag)

		fk_points = np.array(fk_points)
		residuals = np.array(residuals)
		projected_points = np.array(projected_points)

		self.points = points
		self.dimension = points.shape[-1]
		self.circle_center_v1 = circle_center_v1
		self.angle_r1 = angle_r1
		self.rot_mat = rot_mat
		self.residuals = residuals
		self.projected_points = projected_points

		if(points.shape[-1] > 3):
			self.sub_sphere = PNS()
			self.sub_sphere.Fit(fk_points)

		
		

