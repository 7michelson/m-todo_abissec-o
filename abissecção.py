import numpy as np


def bissec(a, b, erro):
    # numero de interações
    n = (np.log(b-a) - np.log(erro))/ np.log(2)
    n = np.ceil(n) # pega o próximo número inteiro
    i = 0
    print('*'*30)
    while i< n:
        # verificando a condição de bolzano
        if f(a)*f(b) > 0:
            print('Não pode afirma se existe raiz nesse intervalo')
        else:
            # método da abissecção
            m = (a+b)/2 # ponto médio
            
            if f(a)*f(m) < 0:
                b = m
            else:
                a = m
        print('Valor de x_{} = {}'.format(i+1, m))
        i = i+1
        print('*'*30)
    return print(f'\nO valor aproximado da raiz é : {m}')


#Definindo a função 
def f(x):
    y = x**2-7
    return y

bissec(2, 3, 0.00001)


print('Valor da raiz: ',7**(1/2))

