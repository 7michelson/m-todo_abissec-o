# Bibliotecas de Álgebra Linear e Cálculo Numérico Aplicadas à Geofísica

Este repositório contém implementações de bibliotecas desenvolvidas para resolver problemas de álgebra linear e cálculo numérico com aplicações voltadas para geofísica. As funções e algoritmos foram construídos do zero, sem o uso de bibliotecas externas, com o objetivo de aumentar o entendimento e controle sobre os métodos utilizados, além de permitir personalizações específicas para contextos geofísicos.

## Motivação

A geofísica, em especial a sismologia, requer o uso intensivo de ferramentas matemáticas para lidar com grandes volumes de dados e realizar cálculos precisos, como a inversão de dados sísmicos, análise de ondas e modelagem de estruturas subterrâneas. O desenvolvimento dessas bibliotecas visa oferecer uma solução customizada e eficiente para:

- Resolver sistemas lineares grandes;
- Executar decomposições matriciais (LU, Cholesky);
- Aplicar transformadas de Fourier discretas (DFT);
- Realizar operações com vetores e matrizes reais e complexos;
- Implementar algoritmos de eliminação e escalonamento;
- E muito mais, tudo de forma escalável e adaptada às necessidades geofísicas.

## Funcionalidades

Algumas das principais funcionalidades incluem:

1. **Decomposição LU e Cholesky:** Resolução eficiente de sistemas lineares utilizando decomposição LU e fatoração de Cholesky, essenciais para resolver equações de ondas e problemas de modelagem geofísica.

2. **Transformada Discreta de Fourier (DFT) e IDFT:** Implementação manual da DFT e sua inversa (IDFT), amplamente utilizada na análise de sinais sísmicos.

3. **Produto de Hadamard:** Operações de produto de Hadamard, aplicadas a vetores e matrizes reais e complexos, úteis em diversas operações de processamento de dados sísmicos.

4. **Eliminação Gaussiana:** Solução de sistemas lineares por eliminação Gaussiana com suporte para qualquer dimensão de matriz, aplicável em ajustes de modelos sísmicos.

5. **Normas de Matrizes:** Cálculo de várias normas de matrizes (Frobenius, 1-norma, 2-norma e infinito-norma), usadas para medir erros em soluções de problemas de inversão geofísica.

## Aplicações em Geofísica

- **Sismologia:** As funções desenvolvidas permitem processar dados sísmicos de maneira eficiente, como correção NMO, empilhamento de dados e inversão de sismogramas.
  
- **Modelagem da Crosta Terrestre:** Através da resolução de sistemas lineares e decomposições matriciais, é possível construir modelos subterrâneos detalhados e determinar propriedades geofísicas, como a localização do Moho (discontinuidade de Mohorovičić).

- **Análise de Ondas:** Ferramentas como a DFT são fundamentais para análise de frequências e comportamento de ondas sísmicas em diferentes camadas da Terra.

## Como Usar

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git







## Tem se como acréssimo a técnica da  Abissecção e Minimos Quadradrados

##### script na linguagem python 

### O script faz uso do método abissecção para uma determinada função onde, a **f(x)** definida no arquivo abissecção.py segue um exemplo **f(x) = x² - 7** e o intervalo é calculado entre os pontos [2,3] e um erro de 0.00001.

### Nesse casso a função do python passa como paramêtro os valores [a,b] e C o erro de propragação para a função na qual pode ser definida. 
 
```python
bissec(a, b, erro)
```
---
---

# Abisection 

###### script in python language

### The script makes use of the abisection method for a given function, where the **f(x)** defined in the file abisection.py follows an example **f(x) = x² - 7** and the interval is calculated between the points [2,3] and an error of 0.00001.

### In this case the python function passes as parameter the values [a,b] and 'erro' the propagation error for the function in which it can be defined.

```python
bissec(a, b, erro)
```
