from text_complexity_analyzer_cm.constants import BASE_DIRECTORY
from text_complexity_analyzer_cm.text_complexity_analyzer import TextComplexityAnalyzer

import pickle

print(BASE_DIRECTORY)
tca = TextComplexityAnalyzer('es')
print(tca.predict_text_category(text='''
Hola a todos, como están?
Hoy es un buen día.

No lo creen? Me gusta este lugar.
Es muy bueno.

Estoy aburrido y no se que escribir.
Entonces necesito ayuda.''', workers=-1))
