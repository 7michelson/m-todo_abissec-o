import nbformat

def merge_notebooks(notebook_paths, output_path):
    # Cria um notebook vazio no formato v4
    merged_notebook = nbformat.v4.new_notebook()

    # Itera sobre os notebooks que serão mesclados
    for path in notebook_paths:
        with open(path, 'r', encoding='utf-8') as file:
            # Lê o notebook atual
            notebook = nbformat.read(file, as_version=4)
            # Adiciona as células desse notebook ao notebook mesclado
            merged_notebook.cells.extend(notebook.cells)

    # Escreve o notebook mesclado no arquivo de saída
    with open(output_path, 'w', encoding='utf-8') as file:
        nbformat.write(merged_notebook, file)

# Lista de caminhos dos notebooks a serem mesclados
notebooks =  ['1 test_dot real and complex.ipynb' , '2 hadamard_testes.ipynb' , '3 teste outer.ipynb' , '4 Test outler logica.ipynb', '6 matrix-vetor.ipynb' , '7 Testando SMA e Derivada1D.ipynb' , '8 Teste matrix - matrix.ipynb' , '9 mat mat teste.ipynb' , '10 Teste_mat_norm.ipynb' , '11 Matrizes triangulares test.ipynb', '12 test eliminação de gauss.ipynb', '13 decomposição e solução LU.ipynb', '14 Testing the function lu_decomp_pivoting.ipynb', '15 Testing the function cho_decomp.ipynb', '16 test least_squares.ipynb' , '17 grav_net_Bruno.ipynb' , '18 Minimos quadrados .ipynb', '19 teste matriz de covariancia.ipynb' , '20  curva de contorno minimos quadrados .ipynb' , '21 Teste DFT e IDFT.ipynb']

# Nome do notebook de saída
output_notebook = 'notebook_teste_bruno.ipynb'

# Chama a função para mesclar os notebooks
merge_notebooks(notebooks, output_notebook)
