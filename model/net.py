'''define the model'''

from keras.layers import Input, Dense, Flatten
from keras.models import Sequential, Model
from keras import backend as K
from keras.layers import LSTM, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, Dropout
import os
from keras.applications import vgg16
from keras import optimizers


class seq2class():
    def __init__(self):  # NEED TO PASS MORE PARAMS #
        self.WIDTH = 156
        self.HEIGHT = 48
        self.CHANNEL = 3
        self.FRAMES = 7
        self.NUM_EXAMPLES = 1134

        self.model = self.build_model()

    def build_model(self):
        '''
        '''
        # Fix parameters ####
        # Need to decided on model architecture#
        # should we use pre-trained model??

        vggmodel = self.build_vgg((self.HEIGHT, self.WIDTH, self.CHANNEL))
        # print(vggmodel.summary())
        x = Input(shape=(self.FRAMES, self.HEIGHT, self.WIDTH, self.CHANNEL))
        # print(x.shape)

        h2 = TimeDistributed(vggmodel)(x)
        # print(h2.shape)
        # Newly added 2d cnn
        h2 = TimeDistributed(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))(h2)

        h2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(h2)
        ####################
        h2 = TimeDistributed(Flatten())(h2)
        # This adds so many params
        # h2 = TimeDistributed(Dense(output_dim=4096, activation='relu'))(h2)

        h2 = TimeDistributed(Dense(output_dim=1024, activation='relu'))(h2)

        # h2 = TimeDistributed(Flatten())(h2)

        h2 = Dropout(0.5)(h2)

        h2 = LSTM(128, return_sequences=False, dropout=0.5)(h2)

        h2 = Dense(1, activation='sigmoid')(h2)

        model = Model(inputs=x, outputs=h2)

        # print(model.summary())

        # extract features and dropout
        # model.add(TimeDistributed(Flatten()))
        # model.add(Dropout(0.5))

        # # NOW feed to RNN

        # model.add(LSTM(256, return_sequences=False, dropout=0.5))
        # Only returns last output

    # classifier with sigmoid activation for multilabel
        # model.add(Dense(1, activation='sigmoid'))
        # # Add more ####################################################
        adam = optimizers.adam(lr=0.01)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        return model

    def build_vgg(self, in_shape):
        vgg = vgg16.VGG16(weights='imagenet', input_shape=in_shape, include_top=False)
        for layer in vgg.layers:
            layer.trainable = False

        outputs = vgg.layers[9].output

        return Model(vgg.input, outputs)
    # save function that saves the checkpoint in the path defined in the config file

    def save_model(self, checkpoint_path=os.path.join(os.getcwd(), '/check_point')):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load_model(self, checkpoint_path=None):
        if self.model is None:
            raise Exception("You have to build the model first.")

        try:
            print("Loading model checkpoint {} ...\n".format(checkpoint_path))
            self.model.load_weights(checkpoint_path)
            print("Model loaded")
        except ValueError:
            print('Not a valid check_point')
