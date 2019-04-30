
import logging
import os
import utils
from keras.callbacks import CSVLogger
from model.data_loader import DataLoader
from model.net import seq2class

##### SET HYPERPARAMETRS ###############
BATCH_SIZE = 64
DATA_PATH = os.path.join(os.getcwd(), 'Dataset')
LEARNING_RATE = 0.0001
EPOCHS = 3  # How many times you want to train the datasets

##########################################


def main():

    # Set the log file for debuging use
    utils.set_logger(os.path.join(os.getcwd(), 'train.log'))
    csv_logger = CSVLogger('log.csv', append=True, separator=';')
    logging.info('Loading datasets...')

    data_loader = DataLoader(DATA_PATH)

    X_train, Y_train, X_val, Y_val = data_loader.get_train_data()
    X_test, Y_test = data_loader.get_test_data()

    logging.info('Building the model...')
    my_model = seq2class()  # NEED TO PASS PARAMETERS SHIT

    print("Here is our model: ")
    print(my_model.model.summary())

    logging.info('Training....')
    history = my_model.model.fit(X_train, Y_train, epochs=EPOCHS, verbose=1, batch_size=BATCH_SIZE, validation_data=(X_val, Y_val), callbacks=[csv_logger])


    
    print('Testing...')
    loss, accuracy = my_model.model.evaluate(X_test, Y_test)
    logging.info('Testing loss', loss)
    logging.info("Test accuracy", accuracy)
    input("waiting for input")
    plot = utils.Plotting(history.history)
    plot.plot_loss()
    plot.plot_accuracy()


if __name__ == "__main__":
    main()
