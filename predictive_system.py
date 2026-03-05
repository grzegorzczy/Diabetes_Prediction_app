import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/Grzesiek/Desktop/KURSY/Generative_AI_Hindus/Kurs_Yt/5. Programy/Supervised_Classification/1.Diabetes_Prediction/trained_model.sav', 'rb'))

### czyli bierzemy jakąś próbkę danych jedną i sprawdzamy czy nam dobrze przewidzi
input_data = (4,110,92,0,0,37.6,0.191,30) # tylko oczywiście bez wartości dla Y - cyzli ma dać 0

#changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we ware predicting for one instance - bo trenowaliśy model na 786 probkach
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshape)
print(prediction)

if prediction == 0:
    print("Out patient is no diabetic")
else:
    print("Our patient is diabetic")