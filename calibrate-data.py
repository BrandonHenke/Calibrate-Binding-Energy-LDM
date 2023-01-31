import numpy as np
import pandas as pd
import scipy.sparse as sps
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.optimize import leastsq,fmin

def BLDM(N,Z):
	A = N+Z
	B = np.array([A,-A**(2/3),-(N-Z)**2/A,-Z**2/A**(1/3)])[:,:,0].T
	return B

def hessian(x):
	"""
	Calculate the hessian matrix with finite differences
	Parameters:
		- x : ndarray
	Returns:
		an array of shape (x.dim, x.ndim) + x.shape
		where the array[i, j, ...] corresponds to the second derivative x_ij
	"""
	x_grad = np.gradient(x) 
	hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
	for k, grad_k in enumerate(x_grad):
		# iterate over dimensions
		# apply gradient again to every component of the first derivative.
		tmp_grad = np.gradient(grad_k) 
		for l, grad_kl in enumerate(tmp_grad):
			hessian[k, l, :, :] = grad_kl
	return hessian


def main():
	col_names=[
		"N",
		"Z",
		"B/A"
	]
	data = pd.read_csv("Masses2016.txt",delim_whitespace=True,skiprows=1,usecols=[0,1,2],names=col_names)
	N = np.array(data["N"]).reshape(len(data["N"]),1)
	Z = np.array(data["Z"]).reshape(len(data["Z"]),1)
	Y = np.array(data["B/A"]).reshape(len(data["B/A"]),1)
	# print(N.shape,Z.shape,Y.shape)

	inds = np.argwhere(np.logical_and(N%2==0,Z%2==0))[:,0]
	N,Z,Y = N[inds],Z[inds],Y[inds]

	A = N+Z
	Y *= A

	B = BLDM(N,Z)

	p = np.linalg.inv(B.T@B)@B.T@Y
	print("Best fit parameters:")
	print(p.T)

	H = hessian(B@p)

	datmat = sps.csr_array(
		(
			(Y-B@p)[:,0],
			(Z[:,0],N[:,0])
		),
		# shape=(
		# 	max(Z)+min(Z),
		# 	max(N)+min(N)
		# 	)
		)
	cRange=[
		min(data["B/A"]),
		max(data["B/A"])
	]
	# print(cRange)
	fig  = plt.figure(figsize=(16,9))
	# norm = mpl.colors.CenteredNorm(vcenter=cRange[1],halfrange=cRange[0])
	plt.pcolormesh(datmat.toarray(),cmap="PRGn",norm=mpl.colors.CenteredNorm())
	plt.colorbar()
	plt.xlabel("Number of neutrons (N)")
	plt.ylabel("Number of protons (Z)")
	plt.savefig("figure.png",dpi=600)

	fig2 = plt.figure(figsize=(16,9))
	plt.scatter(A,Y/A)
	plt.plot(A,B@p/A)
	plt.xlabel("Number of nucleons (A)")
	plt.ylabel("Binding energy per nucleon (B/A)")
	plt.savefig("figure2.png",dpi=600)


if __name__=="__main__":
	main()