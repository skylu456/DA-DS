
"""
Implementación del algoritmo ID3 en el arbol de decisiones.
Este algoritmo utiliza mayormente ID3 con una parte pequeña en CART,
CART es muy similar a ID3 , con la unica diferencia que soporta mas arboles que
solo binarios. Por ello por conveniencia a la hora de decisiones se utilizo
una combinación de CART e ID3.

"""

training_dataset = [                        #Empezamos definiendo el dataset.
    ['Verde', 9, 'Mango'],
    ['Rojo', 8, 'Uva'],
    ['Amarillo', 4, 'Manzana'],
    ['Verde', 5, 'Naranja'],
    ['Rojo', 2, 'Limon'],


   ]
headers = ['color', 'Tamaño', 'Etiqueta'] #Etiquetas de Columnas, es para la impresión..

def unique_vals(rows, col):
    return set([row[col] for row in rows])

def conteo_clases_unico(rows): # Aqui se cuenta cada atributo unico del dataset.
    counts ={}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def es_numero(value):
    return isinstance(value, int) or isinstance(value, float)  # Aqui a funcion determina que cosa es un numero para el
                                                              # dataset.

class Pregunta:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if es_numero(val):
            return val >= self.value
        else:
            return val == self.value
    def __repr__(self):
        condition = "=="
        if es_numero(self.value):
            condition = ">="
        return "Es esto  %s %s %s ?" % (
            headers[self.column], condition, str(self.value)
        )

def partition(rows, pregunta):

    true_rows, false_rows = [], []
    for row in rows:
        if pregunta.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def gini(rows):

    counts =conteo_clases_unico(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain(left, right, current_uncertainty):    # Aqui se calcula la ganancia de Información

    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p*gini(left) - (1-p) * gini(right)

def find_best_split(rows):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) -1

    for col in range(n_features):

        values = set([row[col] for row in rows ])

        for val in values:
            pregunta = Pregunta(col, val)
            true_rows, false_rows = partition(rows, pregunta)
        if len(true_rows)    == 0 or len(false_rows) == 0:
            continue
        gain = info_gain(true_rows, false_rows, current_uncertainty)
        if gain >= best_gain:
            best_gain, best_question = gain, pregunta
    return best_gain, best_question

class Leaf:

    def __init__(self, rows):
        self.predictions = conteo_clases_unico(rows)

class Desicion_node:

    def __init__(self, pregunta, true_branch, false_branch):

        self.pregunta = pregunta
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows):

    gain, pregunta = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, pregunta)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return Desicion_node(pregunta, true_branch, false_branch)

def print_tree(node, spacing = ""):

    if isinstance(node, Leaf):
        print (spacing + "Prediccion", node.predictions)
        return

    print(spacing + str(node.pregunta))
    print(spacing + '- - > Verdadero: ')
    print_tree(node.true_branch, spacing+ " ")
    print( spacing+ ' - - > Falso:')
    print_tree(node.false_branch, spacing + " ")

def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions

    if node.pregunta.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_leaf(counts):
    total  =sum(counts.values()) * 1.0
    probs ={}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total*100)) + "%"
    return probs

if __name__ == '__main__':

    my_tree = build_tree(training_dataset)
    print_tree(my_tree)
    testing_data =[
    ['Verde', 9, 'Mango'],
    ['Rojo', 8, 'Uva'],
    ['Amarillo', 4, 'Manzana'],
    ['Verde', 5, 'Naranja'],
    ['Rojo', 2, 'Limon'],

   ]
    for row in testing_data:
        print("Actual: %s. Predicción: %s" %
              (row[-1], print_leaf(classify(row, my_tree))))



