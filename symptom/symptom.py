import wave
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.utils.vis_utils import plot_model
#from IPython.display import Image
from sklearn.metrics import classification_report, confusion_matrix

#Keras implementation
from keras import Sequential
from keras import optimizers
from keras import backend as K
from keras.layers import Conv2D, Dense, Activation, Dropout, MaxPool2D, Flatten, LeakyReLU
import tensorflow as tf


#Import helper functions:
import mfcc
import data_pipeline as dp


def main():    
    # Load in input data
    # TODO: replace path to data
    df_no_diagnosis = pd.read_csv('../data/demographic_info.txt', names = 
                     ['Patient number', 'Age', 'Sex' , 'Adult BMI (kg/m2)', 'Child Weight (kg)' , 'Child Height (cm)'],
                     delimiter = ' ')
    # Load in outputs
    diagnosis = pd.read_csv('../data/patient_diagnosis.csv', names = ['Patient number', 'Diagnosis'])
 
    print('collecting files')
    root = '../data/data/audio_and_txt_files/'
    filenames = [s.split('.')[0] for s in os.listdir(path = root) if '.txt' in s]

    # Get annotations
    i_list = []
    rec_annotations = []
    rec_annotations_dict = {}
    for s in filenames:
        (i,a) = Extract_Annotation_Data(s, root)
        i_list.append(i)
        rec_annotations.append(a)
        rec_annotations_dict[s] = a
        break
    recording_info = pd.concat(i_list, axis = 0)
    recording_info.head()


    no_label_list = []
    crack_list = []
    wheeze_list = []
    both_sym_list = []
    filename_list = []
    for f in filenames:
        d = rec_annotations_dict[f]
        no_labels = len(d[(d['Crackles'] == 0) & (d['Wheezes'] == 0)].index)
        n_crackles = len(d[(d['Crackles'] == 1) & (d['Wheezes'] == 0)].index)
        n_wheezes = len(d[(d['Crackles'] == 0) & (d['Wheezes'] == 1)].index)
        both_sym = len(d[(d['Crackles'] == 1) & (d['Wheezes'] == 1)].index)
        no_label_list.append(no_labels)
        crack_list.append(n_crackles)
        wheeze_list.append(n_wheezes)
        both_sym_list.append(both_sym)
        filename_list.append(f)

    file_label_df = pd.DataFrame(data = {'filename':filename_list, 'no label':no_label_list, 'crackles only':crack_list, 'wheezes only':wheeze_list, 'crackles and wheezees':both_sym_list})

    print('extracting files')
    target_sample_rate = 22000 
    sample_length_seconds = 5
    sample_dict = dp.extract_all_training_samples(filenames, rec_annotations_dict, root, target_sample_rate, sample_length_seconds) #sample rate lowered to meet memory constraints
    training_clips = sample_dict[0]
    test_clips = sample_dict[1]


    [none_train, c_train, w_train, c_w_train] = [training_clips['none'], training_clips['crackles'], training_clips['wheezes'], training_clips['both']]
    [none_test, c_test, w_test,c_w_test] =  [test_clips['none'], test_clips['crackles'], test_clips['wheezes'], test_clips['both']]


    np.random.shuffle(none_train)
    np.random.shuffle(c_train)
    np.random.shuffle(w_train)
    np.random.shuffle(c_w_train)

    #Data pipeline objects
    print('preparing data')
    train_gen = data_generator([none_train, c_train, w_train, c_w_train], [1,1,1,1])
    test_gen = feed_all([none_test, c_test, w_test,c_w_test])

    print('preparing model')
    model = None
    path = 'symptom_model.h5'
    if os.path.exists(path)
        print('loading model')
        model = keras.models.load_model(path)
    else:
        print('training model')
        model = get_model(sample_height, sample_width)
        # plot_model(model, show_shapes=True, show_layer_names = True)
        # Image(filename='model.png')
        stats = model.fit(x = train_gen.generate_keras(batch_size), 
                            steps_per_epoch = train_gen.n_available_samples() // batch_size,
                            validation_data = test_gen.generate_keras(batch_size),
                            validation_steps = test_gen.n_available_samples() // batch_size, 
                            epochs = n_epochs)
        print('saving model')
        model.save(path)

    assert model

    plt.figure(figsize = (15,5))
    plt.subplot(1,2,1)
    plt.title('Accuracy')
    plt.plot(stats.history['acc'], label = 'training acc')
    plt.plot(stats.history['val_acc'], label = 'validation acc')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(stats.history['loss'], label = 'training loss')
    plt.plot(stats.history['val_loss'], label = 'validation loss')
    plt.legend()
    plt.title('Loss')

    test_set = test_gen.generate_keras(test_gen.n_available_samples()).__next__()
    predictions = model.predict(test_set[0])
    predictions = np.argmax(predictions, axis = 1)
    labels = np.argmax(test_set[1], axis = 1)

    print(classification_report(labels, predictions, target_names = ['none','crackles','wheezes','both']))
    print(confusion_matrix(labels, predictions))

    model.save('symptom_model.h5')


def Extract_Annotation_Data(file_name, root):
    tokens = file_name.split('_')
    recording_info = pd.DataFrame(data = [tokens], columns = ['Patient number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(root, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')
    return (recording_info, recording_annotations)



def get_model(sample_height, sample_width, batch_size=128, n_epochs=15):

    K.clear_session()

    model = Sequential()
    model.add(Conv2D(128, [7,11], strides = [2,2], padding = 'SAME', input_shape = (sample_height, sample_width, 1)))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPool2D(padding = 'SAME'))

    model.add(Conv2D(256, [5,5], padding = 'SAME'))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPool2D(padding = 'SAME'))

    model.add(Conv2D(256, [1,1], padding = 'SAME'))
    model.add(Conv2D(256, [3,3], padding = 'SAME'))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPool2D(padding = 'SAME'))

    model.add(Conv2D(512, [1,1], padding = 'SAME'))
    model.add(Conv2D(512, [3,3], padding = 'SAME',activation = 'relu'))
    model.add(Conv2D(512, [1,1], padding = 'SAME'))
    model.add(Conv2D(512, [3,3], padding = 'SAME', activation = 'relu'))
    model.add(MaxPool2D(padding = 'SAME'))
    model.add(Flatten())

    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(4, activation = 'softmax'))

    opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)

    model.compile(optimizer =  opt , loss = 'categorical_crossentropy', metrics = ['acc'])
    return model


if __name__=="__main__":
    main()

