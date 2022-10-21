from typing import Text
from text_complexity_analyzer_cm.text_complexity_analyzer import TextComplexityAnalyzer
from text_complexity_analyzer_cm.utils.utils import preprocess_text_spanish


texts = ['''Ellos jugaron todo el día. Asimismo, ellas participaron en el juego.


Yo corro con el hermoso gato. A nosotros no nos gusta el gato.
Ella tiene mascotas. Aunque, ella tiene plantas y él no.

Tú jamás dijiste que no, porque ustedes debieron salir temprano al mismo tiempo.''']

tca = TextComplexityAnalyzer('es', preprocessing_func=preprocess_text_spanish)
metrics = tca.calculate_all_indices_for_texts(texts, workers=16, batch_size=1)
categories = tca.predict_text_category(texts, workers=16, batch_size=1)
print(categories)
