#!/usr/bin/env python
# coding: utf-8

# En los primeros artículos de este blog nos propusimos desarrollar un corrector ortográfico para la detección de errores del tipo non-word con el objetivo de iniciarnos en el estudio del **procesamiento del lenuaje natural**. Escribimos parte del código necesario para el cumplimiento de las principales tareas que nos fuimos planteando como: la **carga de datos** desde un **CORPUS** para construir un **modelo del lenguaje** y emplear métodos estadísticos para la generación de sugerencias.
# 
# Después de recibir feedback de varios de los lectores y profundizar un poco más (solo un poco) en las técnicas empleadas para la creación de correctores ortográficos, finalmente tenemos toda la implementación completa de nuestro **corrector ortográfico estadístico**.
# 
# Para esta implementación nos proponemos construir dos modelos del lenguaje, un modelo basado en trigramas de palabras y otro modelo basado en trigramas de etiquetas morfológicas. El objetivo que se persigue con esta nueva implementación (además del objetivo de aprender) es dotar a nuestro corrector ortográfico de mayor exactitud en el momento de mostrar las sugerencias de corrección, ya que un modelo que utilice etiquetas morfológicas nos permitirá calcular la probabilidad de ocurrencia de una palabra dadas las anteriores a ella pero esta vez basado en las características de dicha palabra, ej: si es un sustantivo, adjetivo, verbo, etc..; o si se encuentra en modo **plural/singular**, **masculino/femenino**. Entre otras características que son extraídas por los analizadores morfológicos.
# 
# ### Análisis morfológico
# 
# Por definición el **análisis morfológico** es el conjunto de técnicas que permiten procesar una unidad léxica y establecer una relación entre ésta y el conjunto de rasgos morfológicos que la caracterizan.
# 
# Ejemplo, rasgos morfológicos de los sustantivos comunes (español):
# 
# * Lexema (raíz) y/o forma canónica (lema)
# * Género: masculino, femenino
# * Número: singular, plural
# * Registro: normal, diminutivo, aumentativo, despectivo
# 
# Por ejemplo para la palabra **camino** tenemos los siguientes rasgos morfológicos:
# 
# * Categoría gramatical: sustantivo
# * Lexema: camin-
# * Lema: camino
# * Género: masculino
# * Número: singular
# * Registro: normal
# 
# El conjunto de rasgos morfológicos dependen de la categoría gramatical, por ejemplo la palabra **camino** también puede analizarse tomándola como un verbo en lugar de un sustantivo:
# 
# * Categoría gramatical: verbo
# * Lexema: camin-
# * Lema: caminar
# * Forma: personal
# * Modo: indicativo
# * Tiempo: presente
# * Persona: primera
# * Número: singular
# 
# Para la realización de este proyecto no profundizaremos en como se implementan estos analizadores, en lugar de eso nos valdremos de **spaCy**.

# #### spaCy
# 
# **spaCy** es una biblioteca (librería) de código abierto para realizar tareas de procesamiento del lenguaje natural utilizando Python como lenguaje de programación. Está diseñada específicamente para entornos de producción permitiéndonos construir aplicaciones que procesen grandes volúmenes de texto de manera rápida y eficiente. 
# 
# **Nota**: Inicialmente se había escogido **StanfordNLP** para realizar esta tarea pero demoraba mucho tiempo realizar el etiquetado morfológico de todo el CORPUS; buscando una manera más eficiente de realizar esto fué como llegamos a **spaCy** el cuál mostró un rendimiento considerablemente mayor en cuanto a tiempo de etiquetado. No se pretende establecer una comparación entre ambas bibliotecas pero es bueno señalar que **spaCy** ha sido diseñada especialmente para tener un excelente rendimiento siendo utilizada con Python.
# 
# ##### Instalando spaCy
# 
# ```bash
# pip install spacy
# python -m spacy download es
# ```
# 
# Con estas instrucciones instalamos spaCy y descargamos los archivos necesarios para trabajar con el idioma Español.

# In[1]:


get_ipython().run_cell_magic('time', '', 'import spacy\nfrom spacy.tokens import Doc\nfrom spacy.vocab import Vocab')


# Una vez instalado **spacy** comenzaremos con la carga de nuestros datos.

# In[2]:


def preclean_text(text):
    text = text.replace('\n', '')
    text = '_ _ ' + text.replace('.', '. _ _ ')
    text = text.replace('  ', ' ')
    
    return text


# In[3]:


get_ipython().run_cell_magic('time', '', "textcorpus = ''\n\nfor i in range(1, 203):\n    textcorpus += (open('CORPUS/{}.txt'.format(i)).read().split('Contenido:')[-1])\n    \ntextcorpus = preclean_text(textcorpus)\n\ntextcorpus[:1000]")


# Con el código anterior hemos cargado y pre-procesado todo el contenido de nuestro CORPUS el cuál recordemos que está compuesto por artículos de noticias, esta vez como una característica extra para nuestro CORPUS se decidió incluir además un clásico de la lengua española: **DON QUIJOTE DE LA MANCHA** de Miguel de Cervantes. Comprobamos que el CORPUS que teníamos inicialmente no dotaba a nuestros modelos del lenguaje de suficiente "conocimiento de la lengua" por lo que se decidió agregar un texto que abarcara un vocabulario más amplio.

# ### Etiquetado morfológico del texto utilizando spaCy
# 
# Comenzaremos cargando los modelos de spaCy para el procesamiento de español

# In[4]:


get_ipython().run_cell_magic('time', '', "nlp = spacy.load('es_core_news_md')\nnlp.max_length = 2000000")


# En la instrucción anteriór además hemos incrementado el número máximo de palabras que pueden ser procesadas por **spaCy** debido a que con la cantidad que viene definida por defecto no es suficiente para procesar todo nuestro CORPUS.

# A continuación procesamos el texto realizando el etiquetado morfológico de las palabras.

# In[5]:


get_ipython().run_cell_magic('time', '', "doc = nlp(textcorpus)\ndoc.vocab.to_disk('corpus_vocab')\ndoc.to_disk('corpus_doc')\ndel doc")


# Como se puede apreciar el tiempo que demora realizar el etiquetado morfológico es considerablemente alto por esta razón es que empleamos la función `to_disk` que nos permite almacenar en disco el vocabulario y todo el documento generado por **spaCy** con el etiquetado morfológio de las palabras. Esto nos permite cargar estos datos ya procesados desde el disco en un tiempo mucho menor.

# In[6]:


get_ipython().run_cell_magic('time', '', "vocab = Vocab().from_disk('corpus_vocab')\ndoc = Doc(vocab).from_disk('corpus_doc')")


# Ahora veamos que nos devuelve **spaCy**. Comencemos por leer la primera oración de nuestro CORPUS.

# In[7]:


[sent for sent in doc.sents][0]


# En `doc.sents` tenemos una lista de todas nuestras oraciones. Sin embargo spaCy no nos devuelve en esta lista la oración como tal sino un iterador, para obtener la lista de oraciones podemos crear nuestra propia lista de la siguiente manera:

# In[8]:


sentences = [sent for sent in doc.sents]


# Y acceder a cada oración utilizando el método de indexado tradicional.

# In[9]:


sentences[0]


# Igualmente para las palabras

# In[10]:


sentences[0][3]


# Ahora cada una de las palabras de nuestro texto además están etiquetadas morfológicamente:

# In[11]:


sentences[0][3].lemma_


# In[12]:


sentences[0][3].tag_


# El uso del `_` al final del nombre del atributo que queremos consultar nos permite leer el valor del atributo, si omitimos esto nos devolverá un **id** que es asignado internamente por la biblioteca.

# ### Extrayendo trigramas para la construcción de los modelos

# In[13]:


doc[0]


# In[14]:


offset = doc[0]


# La instruccion anterior nos permiten obtener el caracter `_` con su etiquetado correspondiete para ser utilizado como caracter de relleno en la extracción de nuestros trigramas. En caso que la cantidad de palabras de una oración no se un múltiplo de tres se completarán los caracteres que falten con `_`.

# In[15]:


def get_trigrams(sentences):
    # Lista para almacenar los trigramas
    trigrams = []
    tag_trigrams = []
    
    # Recorrer la lista de palabras
    for sent in sentences:
        for i, _ in enumerate(sent):
            try:
                w1, w2, w3 = sent[i:i+3]
            except:
                try:
                    w1, w2 = sent[i:i+2]
                    w3 = offset
                except:
                    w1 = sent[i]
                    w2 = w3 = offset
            
            # Agregar trigrama a la lista
            trigrams.append((w1.text, w2.text, w3.text))
            
            tag_trigrams.append((
                w1.tag_,
                w2.tag_,
                w3.tag_
            ))
    
    return trigrams, tag_trigrams


# Esta función hace un recorrido por todo el documento extrayendo y almacenando en una lista tanto los trigramas de palabras como los trigramas de etiquetas.
# 
# La siguiente función realiza la misma tarea pero esta vez solo extrayendo los trigramas de palabras, esta función nos será útil posteriormente para la tarea de corrección.

# In[16]:


def get_word_trigrams(sentences):
    # Lista para almacenar los trigramas
    trigrams = []
    
    # Recorrer la lista de palabras
    for sent in sentences:
        for i, _ in enumerate(sent):
            try:
                w1, w2, w3 = sent[i:i+3]
            except:
                try:
                    w1, w2 = sent[i:i+2]
                    w3 = offset
                except:
                    w1 = sent[i]
                    w2 = w3 = offset
            
            # Agregar trigrama a la lista
            trigrams.append((w1.text, w2.text, w3.text))
    
    return trigrams


# Y a continuación procedemos con obtener nuestras listas de trigramas.

# In[17]:


get_ipython().run_cell_magic('time', '', 'word_trigrams, tag_trigrams = get_trigrams(doc.sents)')


# In[18]:


word_trigrams[:10]


# In[19]:


tag_trigrams[:10]


# ### Construccón del modelo

# In[20]:


def build_model(trigrams):
    model = {}
    
    # Contamos la frecuencia de co-ocurrencia
    for i, _ in enumerate(trigrams):
        w1, w2, w3 = trigrams[i]
    
        # El control de excepciones se encarga de manejar los distintos casos 
        # en que un trigrama aún no ha sido registrado.
        try:
            model[w1, w2][w3] += 1
        except: # Aqui se asume que w3 lanza la excepcion
            try:
                model[w1, w2][w3] = 1
            except: # Aqui se asume que el par (w1, w2) lanza la excepcion
                model[w1, w2] = {w3:1}
            
    # Ahora transformamos el conteo en probabilidades
    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
    
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count
            
    return model


# Las instrucciones anteriores lo que hacen es recorrer nuestra lista de trigramas e ir extrayendo cada uno de los que lo conforman. El diccionario que utilizamos como estructura de datos almacenará como llave los dos primeros elementos del trigrama y como valor un nuevo diccionario que tendrá como llave el tercer elemento del trigrama y como valor el número de veces que se repite el trigrama.
# 
# La primera instrucción `model[w1, w2][w3] += 1` asume que ya el trigrama completo ha sido registrado en cuyo caso solo es necesario aumentar el contador, en caso contrario se lanza una excepción y se procede a ejecutar la siguiente instrucción: `model[w1, w2][w3] = 1`, esta instrucción asume que se ha registrado ya una llave `(w1,w2)` para el primer diccionario, en este caso se procede a registrar la llave `w3` para el segundo diccionario y se inicializa su valor 1 indicando que es primera vez que se registra el trigrama. En caso de que no se hallan registrado ninguna de las llaves para ninguno de los dos diccionarios se lanzará nuevamente una excepción y se procede a registrar ambas llaves y a inicializar su valor igualmente con 1 utilizando la instrucción: `model[w1, w2] = {w3:1}`

# Con el último ciclo lo que hacemos es obtener las probabilidades de ocurrencia de cada trigrama. Recorando la hipotesis markoviana $P(w_n|w_1^{n-1})$ se puede estimar como $P(w_n|w_{n-2}^{n-1})$.
# 
# Nuestro objetivo con la construcción de este modelo es poder decir dados dos elementos ya sean palabras o etiquetas gramaticales cuál es la probabilidad de ocurrencia de un tercero. Probabilidad que será utlizada por nuestro corrector para ordenar las sugerencias generadas.

# In[21]:


get_ipython().run_cell_magic('time', '', 'word_model = build_model(word_trigrams)')


# In[22]:


get_ipython().run_cell_magic('time', '', 'tag_model = build_model(tag_trigrams)')


# In[23]:


len(word_model)


# In[24]:


len(tag_model)


# Y así finalmente tenemos construido nuestros modelos del lenguaje.
# 
# Ahora por ejemplo determinemos cual es la probabilidad del siguiente trigrama extraído del corpus:
# 
#     aquellos caballeros andantes

# In[25]:


sentence = "aquellos caballeros andantes"

w1, w2, w3 = sentence.split()

word_model[w1, w2][w3]


# Igualmente podemos realizar esta operación pero para las etiquetas morfológicas, más adelante veremos como realizar esta tarea pero definiremos formalmente que la probabilidad de ocurrencia de un trigrama será la multiplicación de la probabilidad devuelta por `word_model` y `tag_model`. De esta manera estamos tomando tanto en cuenta la probabilidad de ocurrencia por palabras como por etiquetas morfológicas.

# Con la siguientes instrucciones obtendremos las probabilidades más bajas de nuestros modelos para utilizarlas como proababilidad por defecto para el caso en que se intente buscar en nuestro modelo un trigrama que no se encuentra en nuestro CORPUS y por tanto no está contemplado en los modelos (*Problema de las probabilidades nulas*)

# In[26]:


get_ipython().run_cell_magic('time', '', 'word_min = 1\n\nfor i in word_model.values():\n    for ii in i.values():\n        if ii < word_min:\n            word_min = ii\n            \nword_min')


# In[27]:


get_ipython().run_cell_magic('time', '', 'tag_min = 1\n\nfor i in tag_model.values():\n    for ii in i.values():\n        if ii < tag_min:\n            tag_min = ii\n            \ntag_min')


# ### Generación de sugerencias por transformaciones de caracteres
# 
# Una **edición simple** de una palabra es una eliminación (eliminar una letra), una transposición (intercambiar dos letras adyacentes), un reemplazo (cambiar una letra por otra), o un inserción (agregar una letra). La siguiente función nos devuelve un conjunto de todas las cadenas editadas (ya sean palabras del lenguaje o no) que se pueden hacer con una edición simple:

# In[28]:


def edits1(word):
    "Distancia de edición 1"
    letters = 'abcdefghijklmnñopqrstuvwxyzáéíóú'
    
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    
    deletes = [L + R[1:] for L, R in splits if R]
    
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    
    replaces = [L +  C + R[1:] for L, R in splits if R for C in letters]
    
    inserts = [L + C + R for L, R in splits for C in letters]
    
    return set(deletes + transposes + replaces + inserts)


# La siguiente función lo que hace es realizar ediciones simples sobre las ediciones ya realizadas por la función anterior, o sea nos devuelve las palabras que se pueden obtener a partir de dos ediciones simples.

# In[29]:


def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


# Estas funciones nos devuelven un conjunto inmenso de palabras sin embargo nuestro objetivo es obtener palabras que pertenezcan al lenguaje, para esto nos valdremos de un diccionario, la siguiente función nos devuelve el conjunto de todas las palabras existentes en nuestro CORPUS.

# In[30]:


def get_dictionary():
    return set([w.text for w in doc])


# In[31]:


get_ipython().run_cell_magic('time', '', 'dictionary = get_dictionary()')


# In[32]:


len(dictionary)


# Ahora procederemos a implementar una función a la que le daremos como parámetro la lista de palabras generada por nuestras funciones de edición y nos devolvera un conjunto con las palabras de esa lista que sí pertenecen al lenguaje.

# In[33]:


def known(words):    
    return set(w for w in words if w in dictionary)


# La siguiente función es la encargada de generar sugerencias a partir de una palabra que le pasamos como parámetro, pero antes mencionaremos algo muy utilizado en la construcción de correctores ortográficos **modelo del error**.
# 
# Un modelo del error basicamente es un modelo que nos dice con que frecuencia (o probabilidad) se comete un error determinado. Por ejemplo hay errores que son comunes en el español como cambio de `c` por `s` o `v` por `b` y viceversa, además de errores que pueden ocurrir por desconocimiento de algunas reglas ortográficas.
# 
# En nuestro caso no tenemos datos sobre errores ortográficos para poder construir un modelo del error pero definiremos un modelo del error trivial que dice que todas las palabras conocidas de distancia de edición 1 son infinitamente más probables que las palabras conocidas de distancia de edición 2, e infinitamente menos probables que una palabra conocida de distancia de edición 0. Obteniendo el siguiente orden de prioridad:
# 
# 1. Palabra original, si se conoce.
# 2. Lista de palabras conocidas con distancia de edición 1.
# 3. Lista de palabras conocidas con distancia de edición 2.
# 4. Palabra original, aunque no sea conocida.

# In[34]:


def suggestions(word):    
    word = word.lower()
    
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])


# Así hemos definido nuestra función para generar sugerencias.

# In[35]:


suggestions('canbio')


# In[36]:


suggestions('kaza')


# ### Calculo de probabilidades

# Las siguientes funciones son funciones auxiliares utilizadas para obtner la probabilidad de ocurrencia de un trigrama, primeramente se busca el trigrama en el modelo y en caso de que no exista se le asigna la probabilidad mínima.

# In[37]:


def P_Word(trigram):
    n_2, n_1, n = trigram
    
    try:
        return word_model[(n_2, n_1)][n]
    except:
        return word_min


# In[38]:


def P_Tag(trigram):
    n_2, n_1, n = trigram
    
    try:
        return tag_model[(n_2, n_1)][n]
    except:
        return tag_min


# ### Obteniendo sugerencias de trigramas

# In[39]:


def getsugg(trigrams):
    
    trigramlist = []
    
    # Recorrer lista de trigramas extrayendo cada trigrama
    for t in trigrams:
        # Extracción de las palabras de cada trigrama
        w1, w2, w3 = t
        
        # Generación de sugerencias para cada palabra del trigrama
        w1s = suggestions(w1)
        w2s = suggestions(w2)
        w3s = suggestions(w3)
        
        sugglist = []
        
        # Se crean nuevos trigramas con todas las posibles combinaciones 
        # de palabras que han sido sugeridas
        # Se almacenan los trigramas en una lista de sugerencias con la 
        # probabilidad de ocurrencia del trigrama
        for i in w1s:
            for ii in w2s:
                for iii in w3s:
                    w_1, w_2, w_3 = nlp(i + ' ' + ii + ' ' + iii)
                    ptag = P_Tag((w_1.tag_, w_2.tag_, w_3.tag_))
                    pword = P_Word((i, ii, iii))
                    
                    sugglist.append({(i,ii,iii): (pword*ptag)})
                    
        trigramlist.append(sugglist)
                
    return trigramlist


# La función anterior lo que hace es generarnos sugerencias pero esta vez para toda una lista de trigramas obtenemos sugerencias para cada uno de los trigramas y vamos calculando la probabilidad de ocurrencia de cada uno de estos trigramas. Esta información será enviada a la función definida a continuación que será la encargada de obtener de la lista de sugerencias para un trigrama la sugerencia con mayor probabilidad y así conformar una nueva lista con las sugerencias más probales.

# In[40]:


def gettermax(tsugg):
    maxlist = []
    
    for s in tsugg:
        maxlist.append(max(s, key = lambda x : max(x.values()) ))
        
    return maxlist


# Y finalmente implementamos una función a la cuál le enviaremos el texto que deseamos corregir y nos devolverá el texto con las correcciones realizadas.

# In[41]:


def get_correction(text):
    
    # Preprocesamos el texto de entrada
    text = preclean_text(text)
    
    # El texto se analiza con spaCy
    doctest = nlp(text)
    
    # Obtenemos trigramas del texto   
    trigramas_tests = get_word_trigrams(doctest.sents)
    
    # Obtenemos las sugerencias para los trigramas
    ss = getsugg(trigramas_tests)
    
    # Seleccionamos las sugerencias con mayor probabilidad
    common = gettermax(ss)
    
    # Construcción del texto corregido
    
    # Se selecciona el primer trigrama
    w1, w2, w3 = [k for k in common[0].keys()][0]
    ttfinal = [w1, w2, w3]

    # Se van recorriendo los trigramas restantes y haciendo merge (mezcla)
    # Se hace merge entre los trigramas buscando obtener la mayor probabilidad
    for i in range(1, len(common)):
        w1, w2, w3  = [k for k in common[i].keys()][0]
    
        ttfinal.append(w3)
    
        if [v for v in common[i].values()][0] > [v for v in common[i-1].values()][0]:
            ttfinal[i] = w1
            ttfinal[i+1] = w2
    
    # Transformar la lista de palabras en una cadena de texto
    result = ''
    last_w = ''
    
    for w in ttfinal:
        if last_w == '_':
            result += w.capitalize() + ' '
        else:
            result += w + ' '
            
        last_w = w
            
    # Limpieza del resultado final
    
    result = result.replace('. _ _ ', '.')
    result = result.replace(' .', '. ')
    result = result.replace(' :', ': ')
    result = result.replace(' ,', ', ')
    result = result.replace(' ;', '; ')
    result = result.replace('¿ ', '¿')
    result = result.replace('¡ ', '¡')
    result = result.replace(' ?', '?')
    result = result.replace(' !', '!')
    result = result.replace('_ _ ', ' ')
    result = result.replace('  ', ' ')
    
    return result


# En la función anterior incluimos algunos comentarios para tratar de explicar lo que va realizando, sin embargo es difícil ver esto sin conocer exactamente cuáles son las salidas de las funciones que generan las sugerencias de trigramas y cuál es el objetivo que se persigue con realizar la implemetación de esta forma.
# 
# Vamos a mostrar un pequeño test de nuestro corrector ya funcionando para luego comenzar a descomponer las funciones anteriores e ir explicando todo el código.

# In[42]:


get_ipython().run_cell_magic('time', '', 'test_text = """\nHay canbio de planes.\nLa probinsia orienttal.\nEl intrnet de las cosas.\nLa onra del artizta.\nPara unrar a su amigo.\nEl camno del merkado.\nLa defenza de nuestros ideales.\nTod0 está en saver conbibir.\n4lgunos alg0r1tmos pu3d3n s3r muy interes4nt3s.\n"""')


# In[43]:


get_ipython().run_cell_magic('time', '', 'aa = get_correction(test_text)')


# In[44]:


aa


# Como se puede apreciar nuestro corrector es capaz de detectar cuáles son las palabras que no pertenecen al lenguaje y por tanto están escritas de manera incorrecta y generar una corrección para estas.

# ### Proceso de corrección paso a paso.

# Primeramente tenemos un texto con varios errores ortográficos.

# In[45]:


get_ipython().run_cell_magic('time', '', 'test_text = """\nHay canbio de planes.\nLa probinsia orienttal.\nEl intrnet de las cosas.\nLa onra del artizta.\nPara unrar a su amigo.\nEl camno del merkado.\nLa defenza de nuestros ideales.\nTod0 está en saver conbibir.\n4lgunos alg0r1tmos pu3d3n s3r muy interes4nt3s.\n"""')


# Lo primero que haremos será preprocesar este texto

# In[46]:


# Preprocesamos el texto de entrada
text_cleaned = preclean_text(test_text)


# In[47]:


text_cleaned


# Como se puede observar hemos aplicado el mismo preprocesamiento que hemos empleado con nuestro CORPUS. El siguiente paso es etiquetar morfologicamente el texto.

# In[48]:


# El texto se analiza con spaCy
doctest = nlp(text_cleaned)


# Con esto tenemos en `doctest` el texto etiquetado de la misma manera que habíamos hecho con el CORPUS. Realmente el objetivo que perseguimos con esto no es obtener las etiquetas morfológicas sino separar el texto por oraciones para así llamar a una función similar a la que hemos empleado con el CORPUS para obtener los trigramas.

# In[49]:


# Obtenemos trigramas del texto   
trigramas_tests = get_word_trigrams(doctest.sents)


# In[50]:


trigramas_tests[:20] # Primeros 20 trigramas


# Ahora pasaremos a obtener sugerencias para cada uno de los trigramas

# In[51]:


# Obtenemos las sugerencias para los trigramas
ss = getsugg(trigramas_tests)


# Ahora veamos que nos devuelve esta función

# In[52]:


ss[0]


# In[53]:


ss[1]


# In[54]:


ss[2]


# In[55]:


ss[22]


# In[56]:


ss[23]


# In[57]:


ss[24]


# Como podemos apreciar cada elemento de esta lista es otra lista de sugerencias para el trigrama correspondiente donde además cada sugerencia contiene su probabilidad de ocurrencia. Para algunos trigramas solo se obtiene una sugerencia como en el caso de los tres primeros mostrados y para otros obtenemos más de una sugerencia, ahora lo que queremos hacer es quedarnos con los trigramas que mayor probabilidad tienen de esas listas de sugerencias.

# In[58]:


# Seleccionamos las sugerencias con mayor probabilidad
common = gettermax(ss)


# In[59]:


common


# Y aquí podemos observar la lista de trigramas con las posibles correcciones. Ahora lo que deseamos es construir una lista de palabras que se utilizará para construir luego el texto final, esto lo haremos haciendo merge (mezcla) entre los trigramas segun vayan apareciendo. 
# 
# El algortimo sería el siguiente:
# 
# 0. Tenemos una lista vacía donde iremos agregando las palabras
# 
# 1. Las palabras del primer trigrama se agregan directamente a la lista.
# 
# 2. Para todos los trigramas siguientes:
# 
#     2.1 Se agrega la última palabra del trigrama a la lista.
#     
#     2.2 Si la probabilidad de ocurrencia del trigrama actual es mayor que la probabilidad del anterior:
#         
#         2.2.2 Se sustituyen las dos ultimas palabras agregadas por el trigrama anterior por las dos primeras palabras del trigrama actual.

# In[60]:


# Se selecciona el primer trigrama
w1, w2, w3 = [k for k in common[0].keys()][0]
ttfinal = [w1, w2, w3]

# Se van recorriendo los trigramas restantes y haciendo merge (mezcla)
# Se hace merge entre los trigramas buscando obtener la mayor probabilidad
for i in range(1, len(common)):
    w1, w2, w3  = [k for k in common[i].keys()][0]
    
    ttfinal.append(w3)
    
    if [v for v in common[i].values()][0] > [v for v in common[i-1].values()][0]:
        ttfinal[i] = w1
        ttfinal[i+1] = w2


# In[61]:


print(ttfinal)


# Con esto ya tenemos nuestro texto corregido solo es cuestión de trasnformar esta lista de palabras en un texto pero primero aclaremos estas instrucciones que han sido utilizadas para hacer la sustitución de las palabras en el paso `2.2.2` del algoritmo descrito para hacer el *merge*.
# 
# ```python
# ttfinal[i] = w1
# ttfinal[i+1] = w2
# ```
# 
# Recordemos que inicialmente habíamos agregado tres palabras a la lista que son las tres palabras del primer trigrama, por lo tanto nuestra lista como mínimo tendría tres palabras `len(ttfinal) = 3`. Ahora supongamos que nos encontramos analizando el segundo trigrama por tanto `i = 1` y que el segundo trigrama tiene mayor probabilidad que el primero, por lo que corresponde hacer la sustitución. Si ttfinal solamente tiene las tres palabras del trigrama anterior ¿Cómo obtenemos la segunda palabra del trigrama anterior? `ttfinal[1]` pero sabemos que `i = 1` por tanto `ttfinal[1] == ttfinal[i]` y para obtener la tercera palabra sería `ttfinal[2]` o lo que es lo mismo la siguiente palabra a `ttfinal[i]` que sería `ttfinal[i+1]`. 
# 
# A medida que se continúa el recorrido por la lista de trigramas siempre que se debe hacer una sustitución podemos obtener la segunda palabra del trigrama anterior `ttfinal[i]` y la siguiente palabra (la tercera) `ttfinal[i+1]`

# Y finalmente sólo nos queda convertir esta lista en una cadena y eliminar el caracter auxiliar `_` además nos valemos de este caracter auxiliar para saber cuando corresponde colocar una letra mayúscula indicando el inicio de una oración.

# In[62]:


# Transformar la lista de palabras en una cadena de texto
result = ''
last_w = ''
    
for w in ttfinal:
    if last_w == '_':
        result += w.capitalize() + ' '
    else:
        result += w + ' '
            
    last_w = w
            
# Limpieza del resultado final
    
result = result.replace('. _ _ ', '.')
result = result.replace(' .', '. ')
result = result.replace(' :', ': ')
result = result.replace(' ,', ', ')
result = result.replace(' ;', '; ')
result = result.replace('¿ ', '¿')
result = result.replace('¡ ', '¡')
result = result.replace(' ?', '?')
result = result.replace(' !', '!')
result = result.replace('_ _ ', ' ')
result = result.replace('  ', ' ')
    
result


# Finalmente creo que podemos dar por concluida la tarea de implementar este **Corrector Ortográfico Estadístico**. Cuálquier duda o sugerencia por favor pueden dejarlas en los comentarios del post.
# 
# Como nota final para quienes decidan copiar el código y probarlo en sus PC's debo señalar que la tarea del etiquetado morfológico consume una buena cantidad de memoria RAM llegando a tener ocupados casi 4GB de memoria, acá al final les dejaré las propiedades la PC en la que hice las pruebas y si ven que no tienen recursos suficientes para correr el código en sus PC's pues les recomiendo utilizar Google Colab.

# In[63]:


get_ipython().system('neofetch')


# Además todo el código relacionado con este proyecto está disponible en Github:
