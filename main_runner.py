import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import webbrowser
import document_similarity
import os


# Loading the trained model
model = load_model('./model_sign_classi.h5')


# Input the image to predict
test_image = image.load_img('./Dataset/input/PES1PG22CS020.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)

y_prob = model.predict(test_image)

# Getting the predicted class label
srn = np.argmax(y_prob[0])
srn = srn + 1  #offset 1

# fetching the document and printing the document_similarity
print()
print("The input signature matches with SRN ",srn)
print()
if srn <10:
    os.system('start ./Dataset/summarypdfs/PES1PG22CS00'+str(srn)+'.pdf')
else:
    os.system('start ./Dataset/summarypdfs/PES1PG22CS0'+str(srn)+'.pdf')

document_similarity.print_similarity(srn)
