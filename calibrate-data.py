import numpy as np
import pandas as pd
import scipy.sparse as sps
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.optimize import least_squares

def BLDM(N,Z):
	A = N+Z
	B = np.array([A,-A**(2/3),-(N-Z)**2/A,-Z**2/A**(1/3)])[:,:,0].T
	# B /= A
	return B

def residuals(params,B,Y):
	β = np.array(params).reshape(len(params),1)
	ret = Y-B@β
	return ret

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

	β	= np.linalg.inv(B.T@B)@B.T@Y
	H	= 2*B.T@B
	r	= residuals(β,B,Y)
	s_2	= (r.T@r)/(len(Y)-len(β))
	C	= s_2*np.linalg.inv(B.T@B)
	cH	= H
	for a in range(H.shape[0]):
		for b in range(H.shape[1]):
			cH *= H[a,a]**(-1/2)*H[b,b]**(-1/2)

	print("Best fit parameters (manual):")
	print(β.T)
	print("Reduced χ^2:")
	print(s_2)
	print("Hessian:")
	print(H)
	print("Covariance Matrix:")
	print(C)
	print("Conditioned Hessian:")
	print(cH)
	print("Conditioned Hessian Eigenvalues:")
	print(np.linalg.eig(cH)[0])
	
	N	 = np.array(data["N"]).reshape(len(data["N"]),1)
	Z	 = np.array(data["Z"]).reshape(len(data["Z"]),1)
	Y	 = np.array(data["B/A"]).reshape(len(data["B/A"]),1)
	A	 = N+Z
	Y	*= A
	B	 = BLDM(N,Z)
	r	 = residuals(β,B,Y)
	
	datmat = sps.csr_array(
		(
			r[:,0],
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
	plt.plot(A,B@β/A)
	plt.xlabel("Number of nucleons (A)")
	plt.ylabel("Binding energy per nucleon (B/A)")
	plt.savefig("figure2.png",dpi=600)


if __name__=="__main__":
	main()