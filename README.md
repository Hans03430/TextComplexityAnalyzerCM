# Idiomas soportados
Para procesar el texto se utilizó la librería "Spacy". Es necesario instalar los paquetes de idiomas siguientes: "Español", el único idioma soportado hasta el momento. Para ello, usar los siguientes comandos:

python -m spacy download es_core_news_lg

# Métricas implementadas.
De las 45 métricas a implementar originalmente, se implementarán 48. Aquí están listadas aquellas que han sido ya implementadas.

## Índices descriptivos (Descriptive)

    * DESPC: Cantidad de párrafos totales en el texto. Un parrafo es separado por los caractéres '\n\n'.
    * DESSC: Cantidad de oraciones totales en el texto.
    * DESWC: Cantidad de palabras totales en el texto. Una palabra es aquel token identificado por Spacy que solo posee letras.
    * DESPL: Promedio de oraciones por párrafo.
    * DESPLd: Desviación estándar de la cantidad de oraciones por párrafo.
    * DESSL: Promedio de palabras por oración.
    * DESSLd: Desviación estándar de la cantidad de palabras por oración.
    * DESWLsy: Promedio de sílabas por palabra.
    * DESWLsyd: Desviación estándar de la cantidad de sílabas por palabra.
    * DESWLlt: Promedio de letras por palabra.
    * DESWLltd: Desviación estándar de la cantidad de letras por palabra.

## Ińdices de información de palabras (Word information)

    * WRDNOUN: Incidencia de sustantivos.
    * WRDVERB: Incidencia de verbos.
    * WRDADJ: Incidencia de adjetivos.
    * WRDADV: Incidencia de adverbios.
    * WRDPRO: Incidencia de pronombres.
    * WRDPRP1s: Incidencia de pronombres personales en primera persona en singular.
    * WRDPRP1p: Incidencia de pronombres personales en primera persona en plural.
    * WRDPRP2s: Incidencia de pronombres personales en segunda persona en singular.
    * WRDPRP2p: Incidencia de pronombres personales en segunda persona en plural.
    * WRDPRP3s: Incidencia de pronombres personales en tercera persona en singular.
    * WRDPRP3p: Incidencia de pronombres personales en tercera persona en plural.

## Índices de densidad de patrones sintácticos (Syntactic pattern density)

    * DRNP: Incidencia de frases nominales.
    * DRVP: Incidencia de frases verbales.
    * DRNEG: Incidencia de expresiones negativas. Se identifican como expresiones negativas a las siguientes palabras: no, nunca, jamás, tampoco.

## Índices de complejidad sintáctica (Syntactic complexity)

    * SYNNP: Cantidad promedio de modificadores por frase nominal. Se consideran los adjetivos como modificadores.
    * SYNLE: Cantidad promedio de palabras antes del verbo principal.

## Índices de diversidad léxica (Lexical diversity)

    * LDTTRa: Radio entre la cantidad de tokens únicos y la cantidad de palabras totales.
    * LDTTRcw: Radio entre la cantidad de palabras de contenido únicos (Sustantivos, verbos, adjetivos y adveribios) y la cantidad total de estos.

## Índices de legibilidad (Readability)

    * RDFHGL: Índice de Fernández-Huerta para español.

## Índices de cohesión referencial (Referential cohesion)

    * CRFNO: Promedio de oraciones contiguas que poseen solapamiento entre sustantivos. Estos deben coincidir completamente.
    * CRFNOa: Promedio del total de todas las oraciones que poseen solapamiento entre sustantivos. Estos deben coincidir completamente.
    * CRFAO1: Promedio de oraciones contiguas que poseen solapamiento entre pronombres personales o lemmas de sustantivos.
    * CRFAOa: Promedio del total de todas las oraciones que poseen solapamiento entre pronombres personales o lemmas de sustantivos.
    * CRFSO1: Promedio de oraciones contiguas que poseen solapamiento entre lemmas de palabras de contenido.
    * CRFSOa: Promedio del total de todas las oraciones que poseen solapamiento entre lemmas de palabras de contenido.
    * CRFCWO1: Promedio de oraciones contiguas que poseen solapamiento entre palabras de contenido.
    * CRFCWO1d: Desviación estándar de la cantidad de oraciones contiguas que poseen solapamiento entre palabras de contenido.
    * CRFCWOa: Promedio del total de oraciones que poseen solapamiento entre palabras de contenido.
    * CRFCWOad: Desviación estándar de la cantidad de oraciones contiguas que poseen solapamiento entre palabras de contenido.
    * CRFANP1: Promedio de oracioneos contiguas que poseen solapamiento entre pronombres.
    * CRFANPa: Promedio del total de todas las oraciones que poseen solapamiento entre pronombres.

## Índices de connectivos (Connectives)

    * CNCCaus: Incidencia de conectivos causales. Se consideran conectivos causales a los siguientes: a causa de, por ese motivo, dado que, por eso, de ahí que, por esto, de modo que, por lo cual, debido a, por lo dicho, en consecuencia, por lo tanto, por, porque, por ello, pues, por esa razón, ya que.
    * CNCLogic: Incidencia de conectivos lógicos. Se consideran conectivos lógicos a los siguientes: y, o.
    * CNCADC: Incidencia de conectivos adversativos. Se consideran conectivos adversativos a los siguientes: además, entre otros, al fin, es esa época, asumiendo que, futuramente, así también, igualmente, como se expuso anteriormente, incluso, de forma cercana, junto a, de hecho, más aún, de igual forma, otra vez, en consecuencia de lo comentado, para ser preciso, en el pasado, por todo lo dicho, en pocas palabras, posteriormente, en primer lugar, sin embargo, en principio, también, entonces, tan pronto.
    * CNCTemp: Incidencia de conectivos temporales. Se consideran conectivos temporales a los siguientes: a mitad de mañana, al amanecer, al anochecer, al atardecer, al caer la tarde, al mediodía, comenzando la mañana, después del mediodía, en la mañana, en la noche, en la tarde, por la mañana, por la noche, seguida la tarde, al principio, algún tiempo atrás, anteriormente, antes, con anterioridad, desde el principio, en primer momento, hace tiempo, inicialmente, previamente, previo, tiempo antes, tiempo atrás, a la vez, actualmente, al mismo tiempo, al tiempo, en este preciso instante, en este preciso momento, mientras, mientras tanto, paralelamente, simultaneamente.
    * CNCAdd: Incidencia de conectivos aditivos. Se consideran conectivos aditivos a los siguientes: a decir verdad, en primer lugar, aparte, en segundo lugar, asimismo, en tercer lugar, de hecho, en último lugar, de igual forma, por otro lado, de igual manera, por su parte, de igual modo, igualmente, del mismo modo, también.
    * CNCAll: Incidencia de todos los conectivos.
    
# Índices no implementados
Se implementaron todos los 48 indices que se iban a implementar.
