from deepmultilingualpunctuation import PunctuationModel
import time

model = PunctuationModel()
text = "and so my fellow americans ask not what your country can do for you ask what you can do for your country and so my fellow americans ask not what your country can do for you ask what you can do for your country and so my fellow americans ask not what your country can do for you ask what you can do for your country"

start = time.time()
result = model.restore_punctuation(text)
end = time.time()
print(result)
print("Time: ", end - start, "s")

