from typing import Text
from text_complexity_analyzer_cm.text_complexity_analyzer import TextComplexityAnalyzer

if __name__ == '__main__':
    texts = ['''Ellos jugaron todo el día. Asimismo, ellas participaron en el juego.

Yo corro con el hermoso gato. A nosotros no nos gusta el gato.
Ella tiene mascotas. Aunque, ella tiene plantas y él no.

Tú jamás dijiste que no, porque ustedes debieron salir temprano al mismo tiempo.''']
    tca = TextComplexityAnalyzer('es', False)
    tca.calculate_all_indices_for_texts(texts, 1)
