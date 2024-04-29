import os
import sys
from PIL import Image
import numpy as np

os.makedirs('data', exist_ok=True)

def main():
    if 1 == len(sys.argv):
        print(
            """Использовать:
            app [путь к изображению капчи] [[путь: куда сохранить результат]]
            app [путь с меткам датасета для оценки] [[путь к датасету]]"""
        )
        return

    path = sys.argv[1]
    if os.path.isfile(path):
        try:
            data = Image.open(path)
        except:
            print(f'Ошибка чтения файла: "{path}"')
            return
        from encoder import Encoder
        encoder = Encoder()
        img = np.asarray(data)
        out = encoder.inference(img)
        print(out)
        if len(sys.argv) > 2:
            path = sys.argv[2]
            if path[-4].lower() != '.png':
                 path += '.png'
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.scatter(*out, color='r')
            plt.savefig(path)
        return

    if os.path.isdir(path):
        try:
            from score import Calculator
            calc = Calculator() 
            score = calc(path, sys.argv[2] if len(sys.argv) > 2 else None)
            print(f'Точность модели {score*100:2f}% датасета "{path}"')
        except Exception as e:
            print(f'Ошибка при расчете точности: "{e}"')
        finally: return

    print(f'Путь "{path}" не найден')

if "__main__" == __name__:
    main()
