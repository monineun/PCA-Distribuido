import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np


def infodata(X):
	# Calculamos x_mean y Q
	x_mean = X.mean(0)
	n = len(X)

	X1 = (X - x_mean)
	X2 =  np.transpose(X - x_mean)

	Q = np.dot(X2, X1)/(n-1) 

	# Informacion del dataset
	print('\nINFO\n' + '·'*50)
	print(' Tamaño del dataset: ' + str(len(X)) + ' observaciones con ' + str(len(X[0])) + ' variables')
	print('\n Datos:')
	print(X)
	print('\n Media de cada variable:\n ' + str(np.round(X.mean(0), 2)))
	#print('\n Matriz de covarianza:')
	#print(np.cov(X.T))
	print('\n Matriz de covarianza:')
	print(Q)

	return n, x_mean, Q



def eigenv(Q):
	# Calculo de autovectores (direcciones con mayor varianza) y autovalores
	print('\n\nEIGENVALUES & EIGENVECTORS\n' + '·'*50)
	eig_vals, eig_vecs = np.linalg.eig(Q)


	#  Hacemos una lista de parejas (autovector, autovalor) 
	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
	eig_pairs.sort(key=lambda x: x[0], reverse=True)


	# Visualizamos la lista de autovalores en orden desdenciente
	print('\n Autovalores:')
	for i in eig_pairs:
	    print(' ' + str(i[0]))

	print('\n Autovectores:')
	for i in eig_pairs:
	    print(' ' + str(i[1]))


	# A partir de los autovalores, calculamos la varianza explicada
	tot = sum(eig_vals)
	var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
	cum_var_exp = np.cumsum(var_exp)
	print('\n Variabilidad explicada (%): ' + str(np.round(cum_var_exp, 2)))

	return eig_pairs


def dividedata(X, n):
	print(' Repartimos los datos en ' + str(n) + ' grupos')
	nodes = []
	t = int(len(X)/n)
	print(' Cogemos ' + str(t) + ' muestras por grupo')
	
	for i in range(3):
		init = i*t
		end  = i*t + t

		aux = X[init:end, :]
		nodes.append(aux)

		print('\n Grupo ' + str(i))
		print(aux)

	return nodes

def mediafederada(nodes):

	print('\n Calculamos la media federada')

	N = 0
	fedmean = 0

	for x in nodes:
		m = np.sum(x)
		n = len(x)
		N += n

		fedmean = fedmean + (m)

		print('  Sumatorio ' + str(m) + ', ' + str(n) + ' elementos')

	fedmean = fedmean/N
	print('\n La media federada es ' + str(fedmean))


def covarianzafederada(nodes, fedmean):

	print('\n Calculamos la covarianza federada')

	N = 0
	fedQ = 0

	for x in nodes:

		n = len(x)
		N += n

		X1 = (x - fedmean)
		X2 =  np.transpose(x - fedmean)

		fedQ = fedQ + np.dot(X2, X1)

	print('\n')
	print(fedQ/(N-1))




print('\n\n' + ' '*30 + 'I R I S    D A T A S E T')
print('\n' + ' '*20 + 'F E D E R A T E D    E X P E R I M E N T')
print(' '*18 + '---------------------------------------------')


# Leemos los datos
df = pd.read_csv('iris.csv', names=['lng sepalo','anch sepalo','lng petalo','anch petalo','especie'])


# Separamos la variables predictoras de las de clase
X = df.iloc[0:30, 0:1].values
# X = df.iloc[:, 0:4].values
# Y = df.iloc[:, 4].values


n, x_mean, Q = infodata(X)
eigenv(Q)


print('\n'*3 + '·'*50 + '\n· CASO 1. MEDIA CONOCIDA (' + str(float(x_mean)) + ')\n' + '·'*50)
clientes = dividedata(X, 3)
covarianzafederada(clientes, x_mean)

'''
print('\n'*3 + '·'*50 + '\n· CASO 2. MEDIA DESCONOCIDA (?)\n' + '·'*50)
clientes = dividedata(X, 3)
mediafederada(clientes)


print(' \n\n' + ':'*30)

x1 = X[0:10, :]
x2 = X[10:20, :]
x3 = X[20:30, :]

a = []
a.append(x1)
a.append(x2)
a.append(x3)

for x in a:
	print('\n Grupo ')
	print(x)
	print(' con media ' + str(x.mean(0)) + ' y tamaño ' + str(len(x)))

x1m =x1.mean(0)
x2m =x2.mean(0)
x3m =x3.mean(0)

x1n = len(x1)
x2n = len(x2)
x3n = len(x3)

mediaf = ((x1m*x1n)+(x2m*x2n)+(x3m*x3n))/(x1n+x2n+x3n)

print('\n\n La media tiene que ser: ' + str(x_mean))
print(' y sale ' + str(mediaf))


print('\n'*3 + '·'*50 + '\n· CASO 2. MEDIA DESCONOCIDA\n' + '·'*50)
clientes = dividedata(X, 3)
'''




