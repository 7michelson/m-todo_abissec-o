# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:19:55 2024

@author: 7michelson
"""
import numpy as np
import sys

# OPERAÇÕES COM VETORES

def scalar_vec_real(a, x, check_input=True):
    '''
    Compute the product of a scalar a and vector x, where
    a is real and x is in R^N. The imaginary parts are ignored.

    The code uses a simple "for" to iterate on the array.

    Parameters
    ----------
    a : scalar
        Real number.

    x : array 1D
        Vector with N elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array
        Product of a and x.
    '''
    if check_input is True:
        assert isinstance(a, (float, int)), 'a deve ser um escalar'
        assert type(x) == np.ndarray, 'x deve ser um numpy array'
        assert x.ndim == 1, 'x deve ter ndim = 1'

    result = np.empty_like(x)
    for i in range(x.size):
        # the '.real' forces the code to use
        # only the real part of the arrays
        result[i] = a.real*x.real[i]

    return result.real

def scalar_vec_complex(a, x, check_input=True):
    '''
    Compute the dot product of a is a complex number and x
    is a complex vector.

    Parameters
    ----------
    a : scalar
        Complex number.

    x : array 1D
        Complex vector with N elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : scalar
        Product of a and x.
    '''
    if check_input is True:
        assert isinstance(a, (complex, float, int)), 'a may be complex or scalar'
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert x.ndim == 1, 'x must have ndim = 1'

    result_real = scalar_vec_real(a.real, x.real, check_input=False)
    result_real -= scalar_vec_real(a.imag, x.imag, check_input=False)
    result_imag = scalar_vec_real(a.real, x.imag, check_input=False)
    result_imag += scalar_vec_real(a.imag, x.real, check_input=False)

    result = result_real + 1j*result_imag

    return result

def dot_real(x, y, check_input=True):
    '''
    Calcular o produto escalar de x e y, em que
    x, y são elementos de R^N. As partes imaginárias são ignoradas.

    O código usa um simples "for" para iterar nos vetores.

    Parâmetros
    ----------
    x, y : vetores 1D
        Vetores 1D com N elementos.

    check_input : booleano
        Se for Verdadeiro, verifica se a entrada é válida. A predefinição é True.

    Retorna
    -------
    result : escalar
        Produto escalar de x e y.
    '''
    def check(x, y):
        assert len(x) == len(y), 'Numero de elementos em x é diferente de numero de elementos em y'
        assert isinstance(x, np.ndarray), 'x deve ser um numpy array, ex: numpy.array([])'
        assert isinstance(y, np.ndarray), 'y deve ser um numpy array, ex: numpy.array([])'
        assert x.ndim == 1, 'x deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'
        assert y.ndim == 1, 'y deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'
        # Testando se os valores são números reais (não pode ser complexo ou str)
        assert all(not isinstance(value, (str)) and 
                   not isinstance(value, complex) for value in x), "O array x contém valores que não são números reais"
        assert all(not isinstance(value, (str)) and 
                   not isinstance(value, complex) for value in y), "O array y contém valores que não são números reais"

    if check_input == True:
        check(x, y)
    elif check_input == False:
        pass
    
    N = len(x) # lembrar que o N de x e y deve ser igual, fazer o acert
    result = 0
    
    for i in range(0, N):
        result += x[i]*y[i]
    return result

def dot_complex(x, y, check_input=True):
    '''
    Calcula o produto escalar de x e y, onde
    x, y são elementos de C^N.

    O código usa um simples "for" para iterar nas matrizes.

    Parâmetros
    ----------
    x, y : arrays 1D
        Vectores com N elementos.

    check_input : booleano
        Se for Verdadeiro, verifica se a entrada é válida. A predefinição é True.

    Retorna
    -------
    result : escalar
        Produto escalar de x e y.
    '''
    # testando erros
    if check_input==True:
        assert np.size(x) == np.size(y), "Número de elementos de x diferente de y!"
        if dim(x) != 1 and dim(y) != 1:
            raise ArithmeticError('A matrix x e y deve ser 1D.')
    else:
        pass
    assert all(isinstance(val, (complex, int, float)) for val in x), "Vetor x não pertence ao cunjunto dos complexos."
    assert all(not isinstance(val, str) for val in x), "Não deve conter string."
    assert all(isinstance(val, (complex, int, float)) for val in y), "Vetor y não pertence ao cunjunto dos complexos."
    # Extraindo partes reais e imaginárias
    
    x_real = np.real(x)
    x_imag = np.imag(x)
    y_real = np.real(y)
    y_imag = np.imag(y)
    
    #calculando o pruto escalar
    c1_real = dot_real(x_real, y_real) 
    c2_real = dot_real(x_imag, y_imag) #termo que subtrai do real
    c1_imag = dot_real(x_real, y_imag)
    c2_imag = dot_real(x_imag, y_real) #termo que soma do imaginario
    result = (c1_real - c2_real) + 1j*(c1_imag + c2_imag)
    return result

def dim(obj):
    '''
    Essa função recebe uma lsita e retorna a dimenssão dela. 
    '''
    if isinstance(obj, (list, np.ndarray)):
        if isinstance(obj, np.ndarray):
            return obj.ndim
        elif isinstance(obj, list):
            return 1 + max(dim(item) for item in obj) if obj else 1
    else:
        return TypeError('A estrutura de dados não é uma lista ou um numpy array.')

def hadamard_real_matrix(x, y, check_input=True):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be real vectors or matrices having the same shape.
    The imaginary parts are ignored.

    The code uses a simple doubly nested loop to iterate on the arrays.

    Parameters
    ----------
    x, y : arrays
        Real vectors or matrices having the same shape.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''
    N, M = np.shape(x) # formato da matrix
    
    if check_input == True: 
        # Verificar se x e y têm o mesmo formato
        assert np.shape(x) == np.shape(y), 'As matrizes x e y devem ter o mesmo formato.'
        assert not np.iscomplexobj(x), 'A matriz x contém valores complexos. Use a função hadamard_complex.'
        assert not np.iscomplexobj(y), 'A matriz y contém valores complexos. Use a função hadamard_complex.'
        assert not any(isinstance(i, str) for i in x.flat), 'A matriz x contém strings.'
        assert not any(isinstance(i, str) for i in y.flat), 'A matriz y contém strings.'
       
    else:
        pass  
    
    result = np.zeros(shape=(N, M))
    
    for i in range (0, N): 
        for j in range(0, M):
            result[i, j] = x[i, j] * y[i, j]
    return result

def hadamard_real_vector(x, y, check_input=True):
    """
    Calcula o produto de Hadamard (ou multiplicação element-wise) entre dois vetores reais.

    Parâmetros:
    x (list of floats): Vetor 1D com valores reais.
    y (list of floats): Vetor 1D com valores reais do mesmo tamanho que x.
    check_input (bool): Se True, verifica se os vetores têm o mesmo tamanho. Caso contrário, não realiza verificação.

    Retorna:
    list of floats: Vetor resultante da multiplicação element-wise entre x e y.

    Levanta:
    AssertionError: Se check_input é True e os vetores x e y têm tamanhos diferentes.
    """
    if check_input:
        # Verifica se x e y têm o mesmo tamanho
        assert len(x) == len(y), "Os vetores devem ter o mesmo tamanho."
        assert type(x) == np.ndarray or type(y) == np.ndarray, 'Os vetores devem ser um numpy array'
        assert not np.iscomplexobj(x), 'O vetor x devem conter números reais'
        assert not np.iscomplexobj(y), 'O vetor y devem conter números reais'
    # Inicializa o vetor resultante
    z = np.zeros_like(x)
    
    # Calcula o produto de Hadamard
    for i in range(len(x)):
        z[i] = x[i] * y[i]
    
    return z

def hadamard_complex_vector(x, y, check_input=True):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be complex vectors having the same shape.

    Parameters
    ----------
    x, y : arrays
        Complex vectors having the same shape.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''
    
    if check_input == True:
        assert x.ndim == 1 and y.ndim == 1, 'Os vetores x e y devem ser 1D.'
        assert np.shape(x) == np.shape(y), 'As matrizes x e y devem ter o mesmo formato'
        assert x.dtype == complex or y.dype == complex, 'A matriz não é do conjuto do complexo, por favor. verifique os dados!'
    else:
        pass
    N, M  = np.shape(x)
    result = np.zeros(shape=(N,M))
    #separar a matriz em parte real e imaginária.  Porém a função imag() retorno os numeros reais
    Ar = np.real(x) # parte real
    Ai = np.imag(x) # parte imaginaria
    Br = np.real(y) # parte real
    Bi = np.imag(y) # parte imaginaria

    y_R  = hadamard_real_vector(Ar, Br) # real
    y_R -= hadamard_real_vector(Ai, Bi) # real
    y_I  = hadamard_real_vector(Ar, Bi) # complex
    y_I += hadamard_real_vector(Ai, Br) # complex
    result = y_R + 1j*y_I

    return result

def hadamard_complex_matrix(x, y, check_input=True):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be complex matrix having the same shape.

    Parameters
    ----------
    x, y : arrays
        Complex matrix having the same shape.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''
    
    if check_input == True:
        assert x.ndim > 1 and y.ndim > 1, 'As dimenções das matrizes x e y devem ser maior que 1D. dim >'
        assert np.shape(x) == np.shape(y), 'As matrizes x e y devem ter o mesmo formato'
        assert np.iscomplexobj(x) or np.iscomplexobj(y), 'A matriz não é do conjuto do complexo, por favor. verifique os dados!'
    else:
        pass
    N, M  = np.shape(x)
    result = np.zeros(shape=(N,M))
    #separar a matriz em parte real e imaginária.  Porém a função imag() retorno os numeros reais
    Ar = np.real(x) # parte real
    Ai = np.imag(x) # parte imaginaria
    Br = np.real(y) # parte real
    Bi = np.imag(y) # parte imaginaria

    y_R  = hadamard_real_matrix(Ar, Br) # real
    y_R -= hadamard_real_matrix(Ai, Bi) # real
    y_I  = hadamard_real_matrix(Ar, Bi) # complex
    y_I += hadamard_real_matrix(Ai, Br) # complex
    result = y_R + 1j*y_I

    return result

def outer_real(x, y, check_input=True):
    '''
    Outer real simples
    Compute the outer product of x and y, where
    x in R^N and y in R^M. The imaginary parts are ignored.

    The code uses a simple "for" to iterate on the arrays.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with real elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 2d
        Outer product of x and y.
    '''
    N = np.size(x)
    M = np.size(y)
    result = np.zeros((N, M))

    if check_input == True:
        assert not(x.dtype == complex or y.dtype == complex), TypeError('Somente valores reais. Caso queira operar com valores complexo, utilize a função outer complex!')
        
        if dim(x) != 1 or dim(y) != 1:
            raise ArithmeticError('A matrix x e y deve ser 1D.')
        
    for i in range(0, N):
        for j in range(0, M):
            result[i, j] = x[i]*y[j]

    return result.real

def outer_complex(x, y, check_input=True):
    '''
    outer complexo simples
    Calcula o produto externo de x e y, onde x e y são vetores complexos.

    Parâmetros
    ----------
    x, y : vetores 1D
        Vetores complexos.

    check_input : booleano
        Se True, verifica se a entrada é válida. O padrão é True.

    Retornos
    -------
    resultado : matriz 2D
        Produto externo de x e y.
    '''
    if check_input == True:
        # Verifica se x e y são vetores 1D
        assert isinstance(x, np.ndarray) and x.ndim == 1, 'x deve ser um vetor 1D'
        assert isinstance(y, np.ndarray) and y.ndim == 1, 'y deve ser um vetor 1D'
        # Verifica se x e y contêm números complexos
        assert np.iscomplexobj(x), 'x deve conter números complexos'
        assert np.iscomplexobj(y), 'y deve conter números complexos'
    else:
        pass
    # Calcula as partes reais e imaginárias do produto externo
    M_R = outer_real(np.real(x), np.real(y))
    M_R -= outer_real(np.imag(x), np.imag(y))
    M_I = outer_real(np.real(x), np.imag(y))
    M_I += outer_real(np.imag(x), np.real(y))
    # Combina as partes reais e imaginárias
    M = M_R + 1j * M_I

    return M

def outer_real_row(x, y, check_input=True):
    """
    Calcula o produto externo de x e y, onde x está em R^N e y em R^M.
    As partes imaginárias são ignoradas.

    O código usa um único loop para calcular as linhas da matriz resultante
    como um produto escalar-vetor.

    Parâmetros
    ----------
    x, y : listas 1D ou arrays
        Vetores com elementos reais.
    check_input : boolean
        Se True, verifica se a entrada é válida. Padrão é True.

    Retorna
    -------
    M : array 2D
        Produto externo de x e y.
    """
    if isinstance(x, (list, tuple)):
        x = np.asarray(x)  # Transforma x em um array se não for
    elif isinstance(y, (list, tuple)):
        y = np.asarray(y)  # Transforma y em um array se não for
    if check_input:
        # Verifica se x e y são listas, tuplas ou arrays NumPy, e se contêm apenas números reais
        assert isinstance(x, (np.ndarray)) and isinstance(y, (np.ndarray)), "x e y devem ser numpy arrays."
        assert not np.iscomplexobj(x), "x deve conter apenas valores reais."
        assert not np.iscomplexobj(y), "y deve conter apenas valores reais."
    else:
        pass
    N = len(x)
    M = len(y)
    
    # Inicializa a matriz resultante com zeros usando np.zeros
    result = np.zeros((N, M))

    # Calcula as linhas da matriz resultante como produto escalar-vetor
    for i in range(N):
        result[i, :] = scalar_vec_real(x[i], y)
    
    return result

def outer_real_column(x, y, check_input=True):
    """
    Calcula o produto externo de x e y, onde x está em R^N e y em R^M.
    As partes imaginárias são ignoradas.

    O código usa um único loop para calcular as colunas da matriz resultante
    como um produto escalar-vetor.

    Parâmetros
    ----------
    x, y : listas 1D ou arrays
        Vetores com elementos reais.
    check_input : boolean
        Se True, verifica se a entrada é válida. Padrão é True.

    Retorna
    -------
    M : array 2D
        Produto externo de x e y.
    """
    if isinstance(x, (list, tuple)):
        x = np.asarray(x)  # Transforma x em um array se não for
    elif isinstance(y, (list, tuple)):
        y = np.asarray(y)  # Transforma y em um array se não for
    if check_input:
        # Verifica se x e y são listas, tuplas ou arrays NumPy, e se contêm apenas números reais
        assert isinstance(x, (np.ndarray)) and isinstance(y, (np.ndarray)), "x e y devem ser numpy arrays."
        assert not np.iscomplexobj(x), "x deve conter apenas valores reais."
        assert not np.iscomplexobj(y), "y deve conter apenas valores reais."

    N = len(x)
    M = len(y)
    
    # Inicializa a matriz resultante com zeros usando np.zeros
    result = np.zeros((N, M))

    # Calcula as colunas da matriz resultante como produto escalar-vetor
    for j in range(M):
        result[:, j] = scalar_vec_real(y[j], x)
    
    return result





def vec_norm(x, p, check_input=True):
    '''
    x deve ser um vetor. 
    Enquanto p diz a ordem da normalização do vetor.
    p = 0,1,2.
    '''
    if check_input == True:
        assert p == 0 or p==1 or p ==2, 'A ordem só deve ser 0, 1 ou  2.'
        assert np.ndim(x) == 1, 'O vetor deve ter uma dimenção!'
    else:
        pass    
    result = 0
    if p == 0:
        for i in range(0, len(x)):
            x[i] = abs(x[i].real)
        result = np.max(x)
    elif p==1:
        for i in range(0, len(x)):
            result += abs(x[i].real)
    elif p == 2:
        for i in range(0,len(x)):
            result += x[i].real*x[i].real
        result = result**(1/2)

    return float(result)

# OPERAÇÕES COM MATRIZES - VETORES

def mat_norm(A, norm_type='fro', check_input=True):
    """
    Calcula a norma de uma matriz de acordo com o tipo especificado.
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz 2D (NxM) para a qual será calculada a norma.
    
    norm_type : str
        Tipo de norma a ser calculada. Pode ser:
        - 'fro' : Norma de Frobenius (padrão)
        - '1'   : 1-norma (máxima soma de colunas)
        - '2'   : 2-norma (norma espectral)
        - 'inf' : Infinito-norma (máxima soma de linhas)
        
    check_input : bool, opcional
        Se True, verifica a validade da entrada, assegurando que A é uma matriz (não um vetor).
        O padrão é True.
    
    Retorna:
    --------
    norm : float
        O valor da norma calculada.
    
    Exceções:
    ---------
    ValueError:
        Lança erro se A não for uma matriz 2D ou se um tipo de norma inválido for especificado.
    """
    
    # Verificação da entrada
    if check_input:
        # Verifica se a entrada é uma matriz (2D)
        if not isinstance(A, np.ndarray):
            raise ValueError("A deve ser um array numpy.")
        if A.ndim != 2:
            raise ValueError("A deve ser uma matriz 2D. Vetores não são permitidos.")
        assert norm_type in ['fro', '1', '2', 'inf'], "A norma p deve ser 'fro', 1, 2, ou 'inf'."
    
    AtA = matmat_real_dot(A.T, A)
    if norm_type == 'fro':
        # Norma de Frobenius usando a raiz quadrada do traço de A^T A
        return np.sqrt(np.trace(AtA))
    elif norm_type == '1':
        # 1-norma (máxima soma de colunas)
        return np.max(np.sum(np.abs(A), axis=0))
    elif norm_type == '2':
        # 2-norma (norma espectral)
        # Calcula os autovalores de A^T A (matriz de Gram)
        eigenvalues = np.linalg.eigvals(AtA)
        # A 2-norma é a raiz quadrada do maior autovalor
        norm_2 = np.sqrt(np.max(eigenvalues))
        return norm_2
    elif norm_type == 'inf':
        # Infinito-norma (máxima soma de linhas)
        return np.max(np.sum(np.abs(A), axis=1))
    
def matvec_real(A, x, check_input=True):
    '''
    Calcula o produto matriz-vetor de A e x, onde
    A está em R^NxM e x está em R^M. As partes imaginárias são ignoradas.

    O código usa um "for" duplamente aninhado para iterar nas matrizes.

    Parâmetros
    ----------
    A : matriz 2D
        Matriz NxM com elementos reais.

    x : vetor 1D
        Vetor real com M elementos.

    check_input : booleano
        Se True, verifica se a entrada é válida. O padrão é True.

    Retornos
    -------
    y : vetor 1D
        Produto de A e x.
    '''
    if check_input:
        # Verifica se A é uma matriz 2D e x é um vetor 1D
        assert isinstance(A, np.ndarray) and A.ndim == 2, 'A deve ser uma matriz 2D'
        assert isinstance(x, np.ndarray) and x.ndim == 1, 'x deve ser um vetor 1D'
        # Verifica se A e x contêm apenas números reais
        assert np.isrealobj(A), 'A deve conter apenas números reais'
        assert np.isrealobj(x), 'x deve conter apenas números reais'
        # Verifica se as dimensões são compatíveis
        N, M = np.shape(A)
        assert M == len(x), 'O número de colunas de A deve ser igual ao número de elementos em x'
    else:
        N, M = np.shape(A)
    
    # Inicializa o vetor de resultado
    y = np.zeros(N)
    # Realiza a multiplicação matriz-vetor usando laços duplamente aninhados
    for i in range(N):
        for j in range(M):
            y[i] += A[i, j] * x[j]
    
    return y

def matvec_dot(A, x, check_input=True):
    '''
    Calcular o produto matricial-vetorial de A e x, em que
    A em R^NxM e x em R^M. As partes imaginárias são ignoradas.

    O código substitui um for por um produto escalar.

    Parâmetros
    ----------
    A : matriz 2D
        Matriz NxM com elementos reais.

    x : matriz 1D
        Vetor real com M elementos.

    check_input : booleano
        Se for Verdadeiro, verifica se a entrada é válida. A predefinição é True.

    Retorna
    -------
    resultado : matriz 1D
        Produto de A e x.
    '''

    N, M = np.shape(A)
    L = len(x)
    if check_input == True:
        assert M == L, 'Devem ter o mesma quantidade de coluna. ex: 2x3 e 1x3' 
        assert dim(x) == 1, 'x precisa ser um vetor. Deve ser 1D.'
        assert not any(isinstance(i, complex) for i in x), 'O vetor deve ter valores no conjuto dos reais, não no complexo'
        assert not any(isinstance(i, complex) for i in A.flat), 'A matriz deve ter valores no conjuto dos reais, não no complexo'
    else:
        pass
    y = np.zeros(N)
    for i in range(0, N):
        y[i] = dot_real(A[i,:], x[:])
    return y

def matvec_real_simple(A, x, check_input=True):
    '''
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.

    The code uses a simple doubly nested "for" to iterate on the arrays.

    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.

    x : array 1D
        Real vector witn M elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 1D
        Product of A and x.
    '''
    N, M = np.shape(A)
    L = len(x)

    if check_input == True:
        assert M == L, 'Devem ter o mesma quantidade de coluna. ex: 2x3 e 1x3' 
        assert dim(x) == 1, 'x precisa ser um vetor. Deve ser 1D.'
        assert not any(isinstance(i, complex) for i in x), 'O vetor deve ter valores no conjuto dos reais, não no complexo'
        assert not any(isinstance(i, complex) for i in A.flat), 'A matriz deve ter valores no conjuto dos reais, não no complexo'
    else:
        pass
    y = np.zeros(N)

    for i in range(0, N):
        for j in range(0, M):
            y[i] += A[i,j]*x[j]
    
    return y

def matvec_real_columns(A, x, check_input=True):
    '''
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.

    The code replaces a for by a scalar-vector product.

    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.

    x : array 1D
        Real vector witn M elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 1D
        Product of A and x.
    '''
    N, M = np.shape(A)
    L = len(x)
    if check_input == True:
        assert M == L, 'Devem ter o mesma quantidade de coluna. ex: 2x3 e 1x3' 
        assert dim(x) == 1, 'x precisa ser um vetor. Deve ser 1D.'
        assert not any(isinstance(i, complex) for i in x), 'O vetor deve ter valores no conjuto dos reais, não no complexo'
        assert not any(isinstance(i, complex) for i in A.flat), 'A matriz deve ter valores no conjuto dos reais, não no complexo'
    else:
        pass
    y = np.zeros(N)
    for j in range(0, M):
        y[:] += scalar_vec_real(x[j], A[:,j])
    return y

def matvec_complex(B, x, check_input=True):
    '''
    Compute the matrix-vector product of an NxM matrix A and
    a Mx1 vector x.

    Parameters
    ----------
    A : array 2D
        NxM matrix.

    x : array 1D
        Mx1 vector.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 1D
        Product of A and x.
    '''
    if check_input:
        assert any(isinstance(i, complex) for i in B.flat) or any(isinstance(i, complex) for i in x), 'A matrix (B) ou o vetor (X) deve ter valores no conjunto dos complexo'
    # compute the real and imaginary parts of the product
    y_R  = matvec_dot(np.real(B), np.real(x))
    y_R -= matvec_dot(np.imag(B), np.imag(x))
    y_I  = matvec_dot(np.real(B), np.imag(x))
    y_I += matvec_dot(np.imag(B), np.real(x))
    y = y_R + 1j*y_I
    
    # return the result        
    return y

### Média movel e derivada 1D

import numpy as np

def mat_sma(data, window, check_input=True):
    '''
    Calculate the moving average filter by using the matrix-vector product.

    Parameters
    ----------
    data : numpy array 1d
        Vector containing the data.
    window : positive integer
        Positive integer defining the number of elements forming the window.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Vector containing the filtered data.
    '''
    
    if check_input:
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            raise ValueError("data must be a 1-dimensional numpy array")
        if not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer")
        if window >= len(data):
            raise ValueError("window size must be smaller than data size")
        if window % 2 == 0:
            raise ValueError("window size must be odd")
    
    N = len(data)
    ws = window
    i0 = ws // 2
    #matrix caracteristica da media móvel
    A = np.array(
        np.hstack(
            (
                (1./ws) * np.ones(ws), 
                np.zeros(N - ws + 1)
            )
        )
    )
    
    A = np.resize(A, (N - 2 * i0, N))
    A = np.vstack((np.zeros(N), A, np.zeros(N)))
    
    result = matvec_dot(A, data)
    
    return result

def deriv1d(data, spacing, check_input=True):
    '''
    Calculate the first derivative by using the matrix-vector product.

    Parameters
    ----------
    data : numpy array 1d
        Vector containing the data.
    spacing : positive scalar
        Positive scalar defining the constant data spacing.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Vector containing the computed derivative.
    '''
    
    if check_input:
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            raise ValueError("data must be a 1-dimensional numpy array")
        if not isinstance(spacing, (int, float)) or spacing <= 0:
            raise ValueError("spacing must be a positive scalar")
    
    N = len(data)
    ws = 3  # window size
    i0 = ws // 2
    h = spacing
    
    # Step 1: Create the initial matrix with -1, 0, 1 and zeros
    A = np.array(
        np.hstack(
            (np.array([-1, 0, 1]), np.zeros(N - ws + 1))
        )
    )
    
    # Step 2: Resize the matrix
    A = np.resize(A, (N - 2 * i0, N))
    
    # Step 3: Add rows of zeros at the top and bottom
    D = np.vstack((np.zeros(N), A, np.zeros(N)))
    
    # Step 4: Divide by 2h
    D = D / (2 * h)
    
    # Step 5: Multiply by the data vector
    result = matvec_dot(D, data)
    result = np.delete(result, [0, -1]) #elimina o primeiro e o último elemento
    
    return result

### Operações Matriz-Matriz

def matmat_real_simple(A, B, check_input=True):
    '''
    Calcula o produto matriz-matriz de A e B, onde
    A está em R^NxM e B está em R^MxP. As partes imaginárias são ignoradas.

    O código usa um "for" triplo simples para iterar nas matrizes.

    Parâmetros
    ----------
    A, B : matrizes 2D
        Matrizes reais.

    check_input : booleano
        Se True, verifica se a entrada é válida. O padrão é True.

    Retornos
    -------
    resultado : matriz 2D
        Produto de A e B.
    '''
    if check_input == True:
        # Verifica se A e B são matrizes 2D
        assert isinstance(A, np.ndarray) and A.ndim == 2, 'A deve ser uma matriz 2D'
        assert isinstance(B, np.ndarray) and B.ndim == 2, 'B deve ser uma matriz 2D'
        # Verifica se A e B contêm apenas números reais
        assert np.isrealobj(A), 'A deve conter apenas números reais'
        assert np.isrealobj(B), 'B deve conter apenas números reais'
        # Verifica se as dimensões são compatíveis
        N, M = np.shape(A)
        O, L = np.shape(B)
        assert M == O, 'O número de colunas de A deve ser igual ao número de linhas de B'
    else:
        N, M = np.shape(A)
        O, L = np.shape(B)

    # Inicializa a matriz de resultado
    C = np.zeros((N, L))
    # Executa a multiplicação de matrizes
    for i in range(N):
        for j in range(L):
            for k in range(M):
                C[i, j] += A[i, k] * B[k, j]
    return C

def matmat_real_dot(A, B, check_input=True):
    '''
    Calcula o produto matriz-matriz de A e B, onde
    A está em R^NxM e B está em R^MxP. As partes imaginárias são ignoradas.

    O código substitui um "for" por um produto escalar.

    Parâmetros
    ----------
    A, B : matrizes 2D
        Matrizes reais.

    check_input : booleano
        Se True, verifica se a entrada é válida. O padrão é True.

    Retornos
    -------
    resultado : matriz 2D
        Produto de A e B.
    '''
    if check_input == True:
        # Verifica se A e B são matrizes 2D
        assert isinstance(A, np.ndarray) and A.ndim == 2, 'A deve ser uma matriz 2D'
        assert isinstance(B, np.ndarray) and B.ndim == 2, 'B deve ser uma matriz 2D' 
        # Verifica se A e B contêm apenas números reais
        assert np.isrealobj(A), 'A deve conter apenas números reais'
        assert np.isrealobj(B), 'B deve conter apenas números reais'
        # Verifica se as dimensões são compatíveis
        N, M = np.shape(A)
        O, L = np.shape(B)
        assert M == O, 'O número de colunas de A deve ser igual ao número de linhas de B'
    else:
        N, M = np.shape(A)
        O, L = np.shape(B)
    
    # Inicializa a matriz de resultado
    C = np.zeros((N, L))
    # Realiza a multiplicação de matrizes usando produto escalar
    for i in range(N):
        for j in range(L):
            C[i, j] = dot_real(A[i, :], B[:, j])
    
    return C

def matmat_real_columns(A, B, check_input=True):
    '''
    Calcula o produto matriz-matriz de A e B, onde
    A está em R^NxM e B está em R^MxP. As partes imaginárias são ignoradas.

    O código substitui dois "fors" por um produto matriz-vetor definindo
    uma coluna da matriz resultante.

    Parâmetros
    ----------
    A, B : matrizes 2D
        Matrizes reais.

    check_input : booleano
        Se True, verifica se a entrada é válida. O padrão é True.

    Retornos
    -------
    resultado : matriz 2D
        Produto de A e B.
    '''
    if check_input == True:
        # Verifica se A e B são matrizes 2D
        assert isinstance(A, np.ndarray) and A.ndim == 2, 'A deve ser uma matriz 2D'
        assert isinstance(B, np.ndarray) and B.ndim == 2, 'B deve ser uma matriz 2D'
        # Verifica se A e B contêm apenas números reais
        assert np.isrealobj(A), 'A deve conter apenas números reais'
        assert np.isrealobj(B), 'B deve conter apenas números reais'
        # Verifica se as dimensões são compatíveis
        N, M = np.shape(A)
        O, L = np.shape(B)
        assert M == O, 'O número de colunas de A deve ser igual ao número de linhas de B'
    else:
        N, M = np.shape(A)
        O, L = np.shape(B)

    # Inicializa a matriz de resultado
    C = np.zeros((N, L))
    # Realiza a multiplicação de matrizes usando produto matriz-vetor
    for j in range(L):
        C[:, j] = matvec_real(A, B[:, j])
    
    return C

def matmat_real_matvec(A, B, check_input=True):
    '''
    Calcula o produto matriz-matriz de A e B, onde
    A está em R^NxM e B está em R^MxP. As partes imaginárias são ignoradas.

    O código substitui dois "fors" por um produto matriz-vetor definindo
    uma linha da matriz resultante.

    Parâmetros
    ----------
    A, B : matrizes 2D
        Matrizes reais.

    check_input : booleano
        Se True, verifica se a entrada é válida. O padrão é True.

    Retornos
    -------
    resultado : matriz 2D
        Produto de A e B.
    '''
    if check_input:
        # Verifica se A e B são matrizes 2D
        assert isinstance(A, np.ndarray) and A.ndim == 2, 'A deve ser uma matriz 2D'
        assert isinstance(B, np.ndarray) and B.ndim == 2, 'B deve ser uma matriz 2D'
        # Verifica se A e B contêm apenas números reais
        assert np.isrealobj(A), 'A deve conter apenas números reais'
        assert np.isrealobj(B), 'B deve conter apenas números reais'
        # Verifica se as dimensões são compatíveis
        N, M = np.shape(A)
        M2, L = np.shape(B)
        assert M == M2, 'O número de colunas de A deve ser igual ao número de linhas de B'
    else:
        N, M = np.shape(A)
        L = np.shape(B)[1]
    
    # Inicializa a matriz de resultado
    C = np.zeros((N, L))
    # Realiza a multiplicação de matrizes usando produto matriz-vetor
    for i in range(N):
        C[i, :] = matvec_real(B.T, A[i, :])
    
    return C

def matmat_real_outer(A, B, check_input=True):
    '''
    Calcula o produto matriz-matriz de A e B, onde
    A está em R^NxM e B está em R^MxP. As partes imaginárias são ignoradas.

    O código substitui dois "fors" por um produto externo.

    Parâmetros
    ----------
    A, B : matrizes 2D
        Matrizes reais.

    check_input : booleano
        Se True, verifica se a entrada é válida. O padrão é True.

    Retornos
    -------
    resultado : matriz 2D
        Produto de A e B.
    '''
    if check_input:
        # Verifica se A e B são matrizes 2D
        assert isinstance(A, np.ndarray) and A.ndim == 2, 'A deve ser uma matriz 2D'
        assert isinstance(B, np.ndarray) and B.ndim == 2, 'B deve ser uma matriz 2D'
        
        # Verifica se A e B contêm apenas números reais
        assert np.isrealobj(A), 'A deve conter apenas números reais'
        assert np.isrealobj(B), 'B deve conter apenas números reais'
        
        # Verifica se as dimensões são compatíveis
        N, M = np.shape(A)
        O, L = np.shape(B)
        assert M == O, 'O número de colunas de A deve ser igual ao número de linhas de B'
    else:
        N, M = np.shape(A)
    # Inicializa a matriz de resultado
    C = np.zeros((N, L))
    # Realiza a multiplicação de matrizes usando produto externo
    for k in range(M):
        C += outer_real(A[:, k], B[k, :])
    
    return C

def matmat_complex(A, B, check_input=True):
    '''
    Calcula o produto matriz-matriz de A e B, onde
    A está em C^NxM e B está em C^MxP.

    Parâmetros
    ----------
    A, B : matrizes 2D
        Matrizes complexas.

    check_input : booleano
        Se True, verifica se a entrada é válida. O padrão é True.

    Retornos
    -------
    resultado : matriz 2D
        Produto de A e B.
    '''
    if check_input:
        # Verifica se A e B são matrizes 2D
        assert isinstance(A, np.ndarray) and A.ndim == 2, 'A deve ser uma matriz 2D'
        assert isinstance(B, np.ndarray) and B.ndim == 2, 'B deve ser uma matriz 2D'
        # Verifica se A e B contêm números complexos
        assert np.iscomplexobj(A) or np.iscomplexobj(B), 'A ou B deve conter números complexos'
        # Verifica se as dimensões são compatíveis
        N, M = np.shape(A)
        M2, L = np.shape(B)
        assert M == M2, 'O número de colunas de A deve ser igual ao número de linhas de B'
    else:
        N, M = np.shape(A)
    
    # Calcula as partes reais e imaginárias do produto
    C_R = matmat_real_simple(np.real(A), np.real(B))
    C_R -= matmat_real_simple(np.imag(A), np.imag(B))
    C_I = matmat_real_simple(np.real(A), np.imag(B))
    C_I += matmat_real_simple(np.imag(A), np.real(B))
    # Combina as partes reais e imaginárias
    C = C_R + 1j * C_I
    return C


### Resulução sistema ax = d

#Matriz de Permutação
def permut(C, k):
        '''Função para encontrar o pivô e realizar a permutação de linhas'''
        # Encontra o índice do maior elemento em módulo na coluna k a partir da linha k
        max_index = np.argmax(abs(C[k:, k])) + k
        if max_index != k:
            # Realiza a troca de linhas, se necessário
            C[[k, max_index]] = C[[max_index, k]]
        return max_index, C

#Eliminação de Gauss 
  
def Gauss_elim(A, x, check_input=True, Mat_L = False):
    """
    Executa a eliminação Gaussiana em uma matriz quadrada A e vetor x para resolver o sistema linear Ax = b.
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz quadrada 2D (NxN) representando os coeficientes do sistema linear.
        
    x : np.ndarray
        Vetor 1D (N) representando o vetor do lado direito do sistema Ax = b.
        
    check_input : bool, opcional
        Se True, verifica se as entradas são válidas, incluindo a verificação de que A é uma matriz 2D 
        e x é um vetor 1D, além de garantir que ambos contenham apenas números reais. 
        O padrão é True.
    Mat_L : bool, opcional
        Se True, Retrona a matriz L (triangular inferior).
        O padrão é False.
    Retornos:
    ---------
    Uper : np.ndarray
        Matriz 2D (NxN) resultante da eliminação de Gauss, onde Uper é a matriz triangular superior.
    
    b : np.ndarray
        Vetor 1D (N) modificado correspondente ao vetor do lado direito após a eliminação Gaussiana.

    L : np.ndarray
        Matriz 2D (NxN) reultante do processo da eliminação de Gauss, onde L é uma matriz triangular inferior
    
    Exceções:
    ---------
    AssertionError
        Se `check_input` for True e `A` não for uma matriz 2D, ou `x` não for um vetor 1D, ou se contiverem
        valores não reais.
    
    Exemplo de uso:
    --------------
    A = np.array([[2, -1, 1],
                  [3, 3, 9],
                  [3, 3, 5]], dtype=float)
    
    x = np.array([8, 0, -6], dtype=float)
    
    Uper, b = Gaussian_elimination(A, x)
    print("Matriz escalonada (Uper):", Uper)
    print("Vetor modificado (b):", b)
    """
    N = A.shape[0]  # Número de linhas
    Uper = np.copy(A)
    b = np.copy(x)
    I = np.identity(N)
    L = np.identity(N) # L inicia como identidade

    if check_input == True:
        assert isinstance(A, np.ndarray) and A.ndim == 2, 'A deve ser uma matriz 2D'
        assert isinstance(x, np.ndarray) and b.ndim == 1, 'x deve ser uma vetor 1D'
        # Verifica se A e b contêm apenas números reais
        assert np.isrealobj(A), 'A deve conter apenas números reais'
        assert np.isrealobj(x), 'x deve conter apenas números reais'
    else:
        pass
    
    for k in range(N - 1):
        # Criação do vetor u_k
        u_k = np.zeros(N)
        u_k[k] = 1
        #C = np.zeros(A.shape)
        # Criação do vetor t_k
        t_k = np.zeros(N)
      
        for i in range(k + 1, N):
            t_k[i] = Uper[i, k] / Uper[k, k]
        # Cria a matriz de permutação
    
        # Atualização da matriz Uper e do vetor b usando matmat_real_dot e matvec_dot
        M = outer_real(t_k, u_k)
        Uper = matmat_real_dot(I - M, Uper)
        b = matvec_dot(I - outer_real(t_k, u_k), b)

        # Acumula as operações de eliminação para construir a matriz L
        L = matmat_real_dot(L, I + M)
    
    if Mat_L == False:
        return Uper, b
    else:
        return Uper, b, L

# Ax = B resolução de sistema, usando matriz triangulçar superior e inferior
def triangular_superior(a, d, check_input=True):
    '''
    Resolva o sistema linear Ax = y para x usando a substituição reversa.

    Os elementos de x são calculados usando um 'ponto' dentro de um único for.

    Parâmetros
    ----------
    A : numpy array 2d
        Matriz triangular superior.
    y : matriz numpy 1d
        Vetor independente do sistema linear.
    check_input : booleano
        Se True, verifique se a entrada é válida. O padrão é True.

    Retorna
    -------
    Resultado: numpy array 1d
        Solução x do sistema linear.
    '''

    N = len(d)

    if check_input == True:
        # Verifica se A é uma matriz 2D
        if not (isinstance(a, np.ndarray) and a.ndim == 2):
            raise ValueError('A deve ser uma matriz 2D (NxN)')
        # Verifica se d é um vetor 1D
        if not (isinstance(d, np.ndarray) and d.ndim == 1):
            raise ValueError('d deve ser um vetor 1D')
        # Verifica se A é uma matriz quadrada
        if a.shape[0] != a.shape[1]:
            raise ValueError('A deve ser uma matriz quadrada (NxN)')
        # Verifica se o comprimento de d corresponde ao número de linhas de A
        if a.shape[0] != d.size:
            raise ValueError('O comprimento de d deve ser igual ao número de linhas de A')
        # Verifica se A é triangular superior
        if not np.allclose(a, np.triu(a)):
            raise ValueError('A deve ser uma matriz triangular superior')
        # Verifica se não há zeros na diagonal principal de A
        if np.any(np.diag(a) == 0):
            raise ValueError('Nenhum elemento da diagonal principal de A deve ser zero')
    else:
        pass
    
    p = np.zeros(N)
    for i in range(N-1, -1, -1):
        soma = 0
        p[i] = d[i]
        for j in range(i+1, N):
            soma += a[i, j]*p[j]
        p[i] = (d[i]- soma)/a[i,i]
    return p

def triangular_inferior(a, d, check_input=True):
    '''
    Resolva o sistema linear Ax = y para x usando a substituição direta.
    Os elementos de x são calculados usando um 'ponto' dentro de um único for.

    Parâmetros
    ----------
    A : numpy array 2d
        Matriz triangular inferior.
    y : matriz numpy 1d
        Vetor independente do sistema linear.
    check_input : booleano
        Se True, verifique se a entrada é válida. O padrão é True.

    Retorna
    -------
    Resultado: numpy array 1d
        Solução x do sistema linear.
    '''

    if check_input == True:
        # Verifica se A é uma matriz 2D
        if not (isinstance(a, np.ndarray) and a.ndim == 2):
            raise ValueError('A deve ser uma matriz 2D (NxN)')
        
        # Verifica se d é um vetor 1D
        if not (isinstance(d, np.ndarray) and d.ndim == 1):
            raise ValueError('d deve ser um vetor 1D')
        
        # Verifica se A é uma matriz quadrada
        if a.shape[0] != a.shape[1]:
            raise ValueError('A deve ser uma matriz quadrada (NxN)')
        
        # Verifica se o comprimento de d corresponde ao número de linhas de A
        if a.shape[0] != d.size:
            raise ValueError('O comprimento de d deve ser igual ao número de linhas de A')
        
        # Verifica se A é triangular inferior
        if not np.allclose(a, np.tril(a)):
            raise ValueError('A deve ser uma matriz triangular inferior')
        
        # Verifica se não há zeros na diagonal principal de A
        if np.any(np.diag(a) == 0):
            raise ValueError('Nenhum elemento da diagonal principal de A deve ser zero')
    else:
        pass

    N = len(d)
    p = np.zeros(N)
    for i in range(0, N):
        soma = 0
        p[i] = d[i]
        for j in range(0, i):
            soma += a[i, j]*p[j]
        p[i] = (d[i] - soma )/a[i,i]
    return p
#Função de minimos quadrados
def minimos_quadrados(A, d, check_input=True, inc = False):
    """
    Calcula a solução do sistema Ax = d utilizando o método dos mínimos quadrados.
    
    Seja o sistema Ax = d, onde A é a matriz de sensitividade e d é o vetor de dados observados.
    A solução por mínimos quadrados é obtida resolvendo o sistema linear aproximado que minimiza
    o erro quadrático entre os dados observados e os dados preditos.
    
    A solução dos mínimos quadrados, x_min, é dada por:
    
    x_min = (A.T @ A)^(-1) @ A.T @ d
    
    Onde:
    - A.T é a transposta de A.
    - (A.T @ A)^(-1) é a inversa da matriz de Gram.
    - d é o vetor de dados observados.

    Parâmetros:
    -----------
    A : np.ndarray
        Matriz 2D (NxM) representando a matriz de sensitividade dos dados.
    
    d : np.ndarray
        Vetor 1D (N) representando os dados observados.
    
    Retornos:
    ---------
    x_min : np.ndarray
        Vetor 1D (M) que representa a solução dos mínimos quadrados, minimizando o erro quadrático.
    
    Exemplo de uso:
    --------------
    A = np.array([[1, 2], [3, 4], [5, 6]])
    d = np.array([7, 8, 9])
    
    x_min = minimos_quadrados(A, d)
    print("Solução dos mínimos quadrados:", x_min)
    """

    if check_input == True:
        assert isinstance(A, np.ndarray) and A.ndim == 2, 'A deve ser uma matriz 2D'
        assert isinstance(d, np.ndarray) and d.ndim == 1, 'd deve ser um vetor 1D'
    
        # Verifica se A e d têm dimensões compatíveis
        assert A.shape[0] == d.size, 'O número de linhas de A deve ser igual ao tamanho de d'
    else:
        pass
        
    y = matvec_real_simple(A.T, d)
    G = matmat_real_simple(A.T, A)
    G_inv = np.linalg.inv(G)
    x_min = matvec_real_simple(G_inv, y)

    if inc:
        # Calcula a matriz de covariância dos parâmetros
        incerteza = covariancia_parametros(A, d=d)
        return x_min, incerteza

    return x_min

def minimos_quadrados_ponderado(A, d, w, inc = False, check_input=True):
    """
    Estima os valores absolutos de gravidade nos nós de uma rede sintética, 
    usando o método dos mínimos quadrados ponderados, e calcula as incertezas associadas.

    Parâmetros:
    A (np.ndarray): Matriz de coeficientes que relaciona as observações com os nós da rede.
    d (np.ndarray): Vetor de observações de gravidade.
    W (np.ndarray): Matriz de pesos das observações.
    sigma_d (float): Desvio padrão das observações.

    Retorna:
    tuple: Um tuplo contendo:
        - p_hat (np.ndarray): Vetor dos valores estimados de gravidade nos nós.
        - uncertainties (np.ndarray): Vetor das incertezas associadas às estimativas.

    Lança:
    AssertionError: Se as entradas forem inválidas.
    """
    # Verificar as entradas
    W = np.diag(w) # matriz diagonal de w (pesos de ponderamento)
    if check_input == True:
        assert isinstance(A, np.ndarray) and A.ndim == 2, "A deve ser uma matriz 2D (np.ndarray)"
        assert isinstance(d, np.ndarray) and d.ndim == 1, "d deve ser um vetor 1D (np.ndarray)"
        assert isinstance(W, np.ndarray) and W.ndim == 2, "W deve ser uma matriz 2D (np.ndarray)"
        assert A.shape[0] == d.shape[0], "A e d devem ter o mesmo número de linhas"
        assert W.shape == (A.shape[0], A.shape[0]), "W deve ser uma matriz quadrada com o mesmo número de linhas que A"
    else:
        pass
        
    L = matmat_real_simple(matmat_real_dot(A.T, W) , A) # At *W* A
    wd = matvec_dot(W, d) # W *d
    t = matvec_dot(A.T, wd) # (A.t * W *d)
    L_inv = np.linalg.inv(L)
    p_hat = matvec_dot(L_inv, t)

    if inc == True:
        N = len(w) 
        W_half = np.diag([1/np.sqrt(w[0])]*5 + [1/np.sqrt(w[N-1])]*2)
        sigma_d = np.sqrt(variancia(d)) # Desvio padrão dos dados observados
        assert isinstance(sigma_d, (float, int)) and sigma_d > 0, "sigma_d deve ser um número positivo"
        covariance_matrix = L_inv @ A.T @ W_half @ np.linalg.inv(W) @ W_half @ A @ L_inv
        std = np.sqrt(np.diag(covariance_matrix))
        return p_hat, covariance_matrix, std
    else:
        return p_hat

def residuo(A, x_min, d, check_input=True):
    """
    Calcula o resíduo r = d - Ax_min e a norma do resíduo (R2).

    Parâmetros
    ----------
    A : np.ndarray
        Matriz 2D (NxM) representando a matriz de sensitividade dos dados.
    x_min : np.ndarray
        Vetor 1D (M) representando a solução dos mínimos quadrados.
    d : np.ndarray
        Vetor 1D (N) representando os dados observados.
    check_input : bool, opcional
        Se True, verifica se as entradas são válidas. O padrão é True.

    Retornos
    --------
    residuo : np.ndarray
        Vetor 1D (N) representando o resíduo r = d - Ax_min.
    R2 : float
        Norma do resíduo, calculada como ||d - Ax_min||.

    Exceções
    ---------
    AssertionError
        Se `check_input` for True e `A` não for uma matriz 2D, `x_min` e `d` não forem vetores 1D,
        ou se `A`, `x_min`, e `d` tiverem dimensões incompatíveis.

    Exemplo de uso
    --------------
    A = np.array([[1, 2], [3, 4], [5, 6]])
    x_min = np.array([0.5, 1.5])
    d = np.array([7, 8, 9])
    
    residuo_valor, R2 = residuo(A, x_min, d)
    print("Resíduo:", residuo_valor)
    print("Norma do resíduo (R2):", R2)
    """
    
    if check_input == True:
        # Verifica se A é uma matriz 2D
        assert isinstance(A, np.ndarray) and A.ndim == 2, 'A deve ser uma matriz 2D'
        # Verifica se x_min e d são vetores 1D
        assert isinstance(x_min, np.ndarray) and x_min.ndim == 1, 'x_min deve ser um vetor 1D'
        assert isinstance(d, np.ndarray) and d.ndim == 1, 'd deve ser um vetor 1D'
        # Verifica se A, x_min e d têm dimensões compatíveis
        assert A.shape[1] == x_min.size, 'O número de colunas de A deve ser igual ao tamanho de x_min'
        assert A.shape[0] == d.size, 'O número de linhas de A deve ser igual ao tamanho de d'
    else:
        pass

    d_pred = matvec_real_simple(A, x_min)
    residuo = d - d_pred
    R2 = vec_norm(residuo, 2)
    return residuo, R2

# Decomposição LU 
def lu_decomp(A, f, check_input=True):
    """
    Realiza a decomposição LU da matriz A para o sistema linear Ax = f.
    
    A decomposição LU decompõe a matriz A em duas matrizes:
        - L: matriz triangular inferior
        - U: matriz triangular superior
    
    Parâmetros:
    -----------
    A : numpy.ndarray
        Matriz quadrada de coeficientes (n x n) do sistema linear.
    f : numpy.ndarray
        Vetor de termos independentes do sistema (n x 1).
    check_input : bool, opcional
        Se True, verifica as dimensões de entrada de A e f. O padrão é True.
    
    Retorna:
    --------
    L : numpy.ndarray
        Matriz triangular inferior resultante da decomposição LU.
    U : numpy.ndarray
        Matriz triangular superior resultante da decomposição LU.
    """
    if check_input:
        # Verificações de entrada usando assert
        assert isinstance(A, np.ndarray), "A matriz A deve ser um numpy.ndarray."
        assert A.ndim == 2, "A matriz A deve ser 2D."
        assert A.shape[0] == A.shape[1], "A matriz A deve ser quadrada."
        assert isinstance(f, np.ndarray), "O vetor f deve ser um numpy.ndarray."
        assert f.ndim == 1, "O vetor f deve ser 1D."
        assert A.shape[0] == f.shape[0], "O número de linhas de A deve ser igual ao tamanho de f."
    
    # Executa a eliminação de Gauss para obter U, L e b modificados
    U, b, L = Gauss_elim(A, f, Mat_L=True)
    
    return [L, U]

def lu_decomp_pivoting(A, retornaLU = False, check_input=True):
    '''
    Computa a decomposição LU para uma matriz A aplicando pivotamento parcial.
    
    Parâmetros
    ----------
    A : numpy ndarray 2D
        Matriz quadrada do sistema linear.
    retornaLU : booleano
        Se True, decompõe a Matriz C em L U.
        Padrão é False, Retornando somente Matriz C.
    check_input : booleano
        Se True, verifica se a entrada é válida. O padrão é True.
    
    Retornos
    -------
    Se retornaLU == Falso:
        P : lista de inteiros
            Lista contendo as permutações.
        C : numpy array
            Matriz composta  da L (elementos abaixo da diagonal + identidade),
            e U (elementos acima da diagonal incluindo a diagonal).
    Se retornaLU == True:
        C : numpy array 2D
            Matriz Composta de L e U.
        L : numpy array 2D
            Matriz triangular inferior com elementos de L.
        U : numpy array 2D
            Matriz triangular superior com elementos de U.
        '''
    
    N = A.shape[0]
    if check_input:
        assert A.ndim == 2, 'A deve ser uma matriz'
        assert A.shape[1] == N, 'A deve ser quadrada'

    # Cria a matriz C como uma cópia de A
    C = A.copy()

    # Lista inicial de permutações
    P = list(range(N))

            
    # Decomposição LU com pivotamento parcial
    for k in range(N - 1):
        # Etapa de permutação
        p, C = permut(C, k)
        
        # Atualiza a lista de permutações
        P[k], P[p] = P[p], P[k]
        
        # Verifica se o pivô é diferente de zero
        assert C[k, k] != 0., 'pivô nulo!'
        
        # Calcula os multiplicadores de Gauss e armazena na parte inferior de C
        C[k+1:, k] = C[k+1:, k] / C[k, k]
        
        # Zera os elementos na k-ésima coluna
        C[k+1:, k+1:] = C[k+1:, k+1:] - np.outer(C[k+1:, k], C[k, k+1:])
    
    if retornaLU == True:
        # Separando L e U de C
        L = np.tril(C, -1) + np.eye(N)  # (-1) pega o Elementos abaixo da diagonal de C + diagonal de 1s
        U = np.triu(C)  # Elementos acima da diagonal de C (incluindo a diagonal)
        return P, L, U
    else:
        return P, C

def lu_solve(C, y, check_input=True):
    """
    Resolve o sistema linear LUx = y, onde C contém as matrizes L e U 
    resultantes da decomposição LU.

    Parâmetros:
    -----------
    C : list
        Lista contendo duas matrizes [L, U], onde:
        - L é a matriz triangular inferior (decomposição LU).
        - U é a matriz triangular superior (decomposição LU).
    
    y : np.ndarray
        Vetor independente do sistema linear LUx = y.
    
    check_input : bool, opcional
        Se True, verifica se as entradas são válidas. O padrão é True.

    Retorna:
    --------
    x : np.ndarray
        Vetor solução do sistema linear LUx = y.
    """
    # Verificação de entrada se o check_input estiver habilitado
    if check_input ==True:
        assert isinstance(C, list) and len(C) == 2, "C deve ser uma lista contendo [L, U]."
        assert isinstance(y, np.ndarray), "y deve ser um vetor (np.ndarray)."
        # Extrair L e U da lista C
        L, U = C
        assert L.shape[0] == L.shape[1] == U.shape[0] == U.shape[1], "Matrizes L e U devem ser quadradas."
        assert y.shape[0] == L.shape[0], "Dimensão de y deve ser compatível com L e U."
    else:
        # Extrair L e U da lista C
        L, U = C
    
    # Resolver o sistema Ly = y com substituição direta
    y_sol = triangular_inferior(L, y)
    # Resolver o sistema Ux = y_sol com substituição reversa
    x = triangular_superior(U, y_sol)
    return x

def lu_solve_pivoting(P, C, y, check_input=True):
    '''
    Resolve o sistema linear Ax = y utilizando a decomposição LU da matriz A 
    com pivotamento parcial.
    
    Parâmetros
    ----------
    P : lista de inteiros
        Lista contendo todas as permutações definidas para computar a decomposição LU 
        com pivotamento parcial (saída da função 'lu_decomp_pivoting').
    C : numpy ndarray 2D
        Matriz quadrada contendo os elementos de L abaixo da diagonal principal e 
        os elementos de U na parte superior (incluindo os elementos da diagonal principal).
        (Saída da função 'lu_decomp_pivoting').
    y : numpy ndarray 1D
        Vetor independente do sistema linear.
    check_input : booleano
        Se True, verifica se a entrada é válida. O padrão é True.
    
    Retornos
    -------
    x : numpy ndarray 1D
        Solução do sistema linear Ax=y.
    '''
    N = C.shape[0]
    
    if check_input:
        assert C.ndim == 2, 'C deve ser uma matriz'
        assert C.shape[1] == N, 'C deve ser quadrada'
        assert isinstance(P, list), 'P deve ser uma lista'
        assert len(P) == N, 'P deve ter N elementos'
        assert y.ndim == 1, 'y deve ser um vetor'
        assert y.size == N, 'O número de colunas de C deve ser igual ao tamanho de y'
    
    # Aplicar as permutações no vetor y
    y_permuted = y[P]

    # Separar L e U da matriz C
    L = np.tril(C, -1) + np.eye(N)  # Matriz L com diagonal de 1s
    U = np.triu(C)  # Matriz U

    # Passo 1: Resolver Lz = y_permuted (substituição direta)
    w = triangular_inferior(L, y_permuted)

    # Passo 2: Resolver Ux = z (substituição retroativa)
    x = triangular_superior(U, w)
    return x

# Decomposição de Cholesky

def cho_decomp(A, check_input=True):
    '''
    Compute the Cholesky decomposition of a symmetric and 
    positive definite matrix A. Matrix A is not modified.
    
    Parameters
    ----------
    A : numpy narray 2d
        Full square matrix of the linear system.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    G : numpy array 2d
        Lower triangular matrix representing the Cholesky factor of matrix A.
    '''
    N = A.shape[0]
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[1] == N, 'A must be square'
        assert np.all(A.T == A), 'A must be symmetric'
    
    G = np.zeros((N, N))  # Initialize G as an NxN matrix of zeros
    for j in range(N):
        # Compute the diagonal element of G
        G[j, j] = A[j, j] - dot_real(G[j, :j], G[j, :j])
        if G[j, j] <= 0:
            raise ValueError("A is not positive definite")
        G[j, j] = np.sqrt(G[j, j])
        
        # Compute the off-diagonal elements of G
        if j < N - 1:
            G[j+1:, j] = (A[j+1:, j] - matvec_dot(G[j+1:, :j], G[j, :j])) / G[j, j]
    
    return G

def cho_decomp_overwrite(A, check_input=True):
    '''
    Compute the Cholesky decomposition of a symmetric and 
    positive definite matrix A. The lower triangle of A, including its main
    diagonal, is overwritten by its Cholesky factor.
    
    Parameters
    ----------
    A : numpy narray 2d
        Full square matrix of the linear system.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    A : numpy array 2d
        Modified matrix A with its lower triangle, including its main diagonal, overwritten 
        by its corresponding Cholesky factor.
    '''
    N = A.shape[0]
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[1] == N, 'A must be square'
        assert np.all(A.T == A), 'A must be symmetric'
    
    for j in range(N):
        # Atualizar o elemento diagonal
        A[j, j] = np.sqrt(A[j, j] - np.dot(A[j, :j], A[j, :j]))
        
        if A[j, j] <= 0:
            raise ValueError("A matriz não é definida positiva")
        
        # Atualizar os elementos abaixo da diagonal
        for i in range(j+1, N):
            A[i, j] = (A[i, j] - np.dot(A[i, :j], A[j, :j])) / A[j, j]
        
        # Zerando a parte superior (opcional, mas pode ajudar na depuração)
        A[j, j+1:] = 0.0
    
    return A

def cho_inverse(G, check_input=True):
    '''
    Compute the inverse of a symmetric and positive definite matrix A 
    by using its Cholesky factor.
    
    Parameters
    ----------
    G : numpy narray 2d
        Cholesky factor of matrix A (output of function 'cho_decomp' or 'cho_decomp_overwrite').
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    Ainv : numpy array 2d
        Inverse of A.
    '''
    N = G.shape[0]
    if check_input is True:
        assert G.ndim == 2, 'G must be a matrix'
        assert G.shape[1] == N, 'G must be square'
    
    # Inicializar Ainv como uma matriz identidade
    Ainv = np.zeros((N, N))
    
    # Resolver para cada coluna de Ainv
    for i in range(N):
        e_i = np.zeros(N)
        e_i[i] = 1  # vetor unitário
        # Resolver G @ z = e_i
        z = np.linalg.solve(G, e_i)
        # Resolver G.T @ Ainv[:, i] = z
        Ainv[:, i] = np.linalg.solve(G.T, z)
    
    return Ainv

# Transformada discrta de fourier
def DFT_matrix(N):
    """Cria a matriz de transformada discreta de Fourier (DFT)."""
    W = np.exp(-2j * np.pi / N)
    F = np.array([[W ** (n * k) for k in range(N)] for n in range(N)])
    return F

def DFT(x):
    """Calcula a Transformada Discreta de Fourier (DFT) de um vetor x."""
    N = len(x)
    F = DFT_matrix(N)
    return matvec_complex(F, x)

def IDFT_matrix(N):
    """Cria a matriz de transformada discreta de Fourier inversa (IDFT)."""
    W = np.exp(2j * np.pi / N)
    F_inv = np.array([[W ** (n * k) for k in range(N)] for n in range(N)])
    return F_inv

def IDFT(X):
    """Calcula a Transformada Discreta de Fourier inversa (IDFT) de um vetor X."""
    N = len(X)
    F_inv = IDFT_matrix(N)
    return np.dot(F_inv, X) / N


# Extra 
def mat_covariancia(H, v):
    N = v.shape[0]
    var = variancia(v)
    # Defina a matriz de covariância Σ_v (diagonal para elementos não correlacionados)
    sigma_v = var  # Variância de v
    Sigma_v = sigma_v ** 2 * np.eye(N)  # Matriz de covariância diagonal

    #Matriz de Covariancia
    Sigma_t = Sigma_v @ matmat_real_dot(H, H.T)
    return Sigma_t

def covariancia_parametros(A, d):
    """
    Calcula a matriz de covariância dos parâmetros estimados.
    
    Parameters:
    A (numpy.ndarray): Matriz de design.
    Sigma_d (numpy.ndarray): Matriz de covariância dos dados.
    
    Returns:
    numpy.ndarray: Matriz de covariância dos parâmetros estimados.
    """
    N = d.shape[0]
    var = variancia(dados=d)
    # Defina a matriz de covariância Σ_v (diagonal para elementos não correlacionados)
    sigma_d = var  # Variância de v
    Sigma_d = sigma_d ** 2 * np.eye(N)  # Matriz de covariância diagonal
    # Calcula a matriz de covariância dos parâmetros
    return np.linalg.inv(A.T @ np.linalg.inv(Sigma_d) @ A)

def variancia(dados):
    N = len(dados)
    media = np.mean(dados)
    soma = 0
    for i in range(0, N):
        soma = soma + (dados[i] - media)**2
    var = soma/(N-1)
    return var

def coeficiente_determinacao(y_true, y_pred):
    """
    Calcula o coeficiente de determinação R².
    
    Parâmetros:
    -----------
    y_true : array-like
        Valores observados (reais).
    y_pred : array-like
        Valores previstos pelo modelo.
    
    Retorna:
    --------
    r2 : float
        O coeficiente de determinação R².
    """
    # Converte para numpy arrays se não forem
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calcula a soma dos quadrados dos resíduos (SSR)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Calcula a soma total dos quadrados (SST)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Calcula o coeficiente de determinação (R²)
    r2 = 1 - (ss_res / ss_tot)
    
    return r2

def erro_padrao_e_intervalo(param_cov_matrix, alfa=0.05):
    """
    Calcula os erros padrão dos parâmetros e seus intervalos de confiança.
    
    Parâmetros:
    -----------
    param_cov_matrix : array-like
        Matriz de covariância dos parâmetros estimados.
    alfa : float, opcional
        Nível de significância para o intervalo de confiança (padrão: 0.05 para 95% de confiança).
        
    Retorna:
    --------
    erros_padrao : np.ndarray
        Erros padrão dos parâmetros.
    intervalos_conf : np.ndarray
        Intervalos de confiança para os parâmetros (inferior, superior).
    """
    # Número de parâmetros
    num_params = param_cov_matrix.shape[0]
    
    # Erros padrão (raiz quadrada da variância, que está na diagonal da matriz de covariância)
    erros_padrao = np.sqrt(np.diag(param_cov_matrix))
    
    # Valor crítico para a distribuição normal (para o intervalo de confiança)
    z = 1.96  # Aproximadamente para 95% de confiança
    
    # Calcula os intervalos de confiança
    intervalos_conf = np.zeros((num_params, 2))
    for i in range(num_params):
        intervalo = z * erros_padrao[i]
        intervalos_conf[i, 0] = -intervalo
        intervalos_conf[i, 1] = intervalo
    
    return erros_padrao, intervalos_conf

# SQ line
def straight_line_matrix(x):
    """
    Cria a matriz de design para ajuste de linha reta.
    
    Parameters:
    x (numpy.ndarray): Vetor de pontos x.
    
    Returns:
    numpy.ndarray: Matriz de design para linha reta.
    """
    # Adiciona uma coluna de 1s para o termo de interceptação
    return np.vstack([x, np.ones_like(x)]).T

def straight_line(x, y):
    """
    Ajusta uma linha reta aos dados x e y e retorna os parâmetros da linha.
    
    Parameters:
    x (numpy.ndarray): Vetor de pontos x.
    y (numpy.ndarray): Vetor de pontos y.
    
    Returns:
    numpy.ndarray: Vetores de parâmetros [inclinação, interceptação].
    """
    A = straight_line_matrix(x)
    # Resolve o sistema de equações normais para obter os parâmetros
    return np.linalg.lstsq(A, y, rcond=None)[0]


#Somente para os residuos 
class Estatisticas:
    def __init__(self, residuo):
        """
        Inicializa o objeto com o vetor de resíduos.
        
        :param residuo: Lista ou vetor contendo os resíduos.
        """
        self.residuo = residuo
        self.media = self.calc_media()
        self.variancia = self.calc_variancia()
        self.desvio_padrao = self.calc_desvio_padrao()

    def calc_media(self):
        """Calcula a média dos resíduos."""
        return np.mean(self.residuo)

    def calc_variancia(self):
        """Calcula a variância dos resíduos."""
        return np.var(self.residuo, ddof=1)  # ddof=1 para variância amostral

    def calc_desvio_padrao(self):
        """Calcula o desvio padrão dos resíduos."""
        return np.std(self.residuo, ddof=1)  # ddof=1 para desvio padrão amostral