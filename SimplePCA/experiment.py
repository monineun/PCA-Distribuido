import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np
import random


# Calcula la media y la matriz de covarianza de X
def infodata(X, dbg=True):
	
	x_mean = X.mean(0)
	n = len(X)

	X1 = (X - x_mean)
	X2 =  np.transpose(X - x_mean)

	Q = np.dot(X2, X1)/(n-1) 

	# Informacion del dataset
	if dbg:
		print('\nINFO\n' + '·'*50)
		print(' Tamaño del dataset completo: ' + str(len(X)) + ' observaciones con ' + str(len(X[0])) + ' variables')
		print('\n Media de cada variable:\n ' + str(X.mean(0)))
		print('\n Matriz de covarianza:')
		print(Q)

	return x_mean, Q


# Calcula los autovalores y autovectores de la matriz de covarianza
def eigenv(Q, dbg=True):

	eig_vals, eig_vecs = np.linalg.eig(Q)

	#  Hacemos una lista de parejas (autovector, autovalor) 
	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
	eig_pairs.sort(key=lambda x: x[0], reverse=True)


	# A partir de los autovalores, calculamos la varianza explicada
	tot = sum(eig_vals)
	var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
	cum_var_exp = np.cumsum(var_exp)


	if dbg:

		print('\n\nEIGENVALUES & EIGENVECTORS\n' + '·'*50)
		
		print('\n Autovalores:')
		for i in eig_pairs:
		    print(' ' + str(i[0]))

		print('\n Autovectores:')
		for i in eig_pairs:
		    print(' ' + str(i[1]))
		
		print('\n Variabilidad explicada (%): ' + str(cum_var_exp))

	return eig_pairs, cum_var_exp


# Divide los datos de X en N grupos de tamaño aleatorio
def dividedata(X, N, dbg=True):
	
	nodes = []

	splt = sorted(random.sample(range(1, len(X)), N-1))
	splt.insert(0,0)
	splt.append(len(X))
	
	for i in range(N):

		init = splt[i]
		end  = splt[i+1]

		aux = X[init:end, :]
		nodes.append(aux)

	if dbg:
		print(' Repartimos los datos en ' + str(N) + ' grupos')

		for i in range(N):
			print(' · Grupo ' + str(i) + ': ' + str(len(nodes[i])) + ' muestras')


	return nodes


# Calcula la media de manera federada
def federatedmean(nodes, dbg=True):

	N = 0
	fedmean = 0

	for x in nodes:
		m = np.sum(x, axis=0)
		n = len(x)
		N += n

		fedmean = fedmean + (m)

	fedmean = fedmean/N

	if dbg:
		print('\n Calculamos la media federada')
		print(' ' + str(fedmean))

	return fedmean


# Calcula la covarianza de manera federada
def federatedcov(nodes, fedmean, dbg=True):

	N = 0
	fedQ = 0

	for x in nodes:

		n = len(x)
		N += n

		X1 = (x - fedmean)
		X2 =  np.transpose(x - fedmean)

		fedQ = fedQ + np.dot(X2, X1)

	fedQ = fedQ/(N-1)

	if dbg:
		print('\n Calculamos la covarianza federada')
		print(fedQ)

	return fedQ




print('\n\n' + ' '*30 + 'I R I S    D A T A S E T')
print('\n' + ' '*20 + 'F E D E R A T E D    E X P E R I M E N T')
print(' '*18 + '---------------------------------------------')


# Leemos los datos
df = pd.read_csv('iris.csv', names=['lng sepalo','anch sepalo','lng petalo','anch petalo','especie'])
X = df.iloc[:, 0:4].values


# Calculamos la media, covarianza y autovalores y autovectores de los datos
mean, Q = infodata(X)
eig_pairs, cum_var_exp = eigenv(Q)


# Caso 1. Media conocida
print('\n'*3 + '·'*50 + '\n· CASO 1. MEDIA CONOCIDA\n' + '·'*50)
clientes = dividedata(X, 5)
federatedcov(clientes, mean)


# Caso 2. Media desconocida
print('\n'*2 + '·'*50 + '\n· CASO 2. MEDIA DESCONOCIDA (?)\n' + '·'*50)
clientes = dividedata(X, 5)
fm = federatedmean(clientes)
federatedcov(clientes, fm)






