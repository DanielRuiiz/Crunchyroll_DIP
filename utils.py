'''
Helper functions
'''
import logging
import matplotlib.pyplot as plt


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


class Plotting():
    def __init__(self, history):
        self.history = history
        self.loss = self.history['loss']
        self.val_loss = self.history['val_loss']

        self.acc = self.history['acc']
        self.val_acc = self.history['val_acc']

        self.epochs = range(1, len(self.loss) + 1)

    def plot_loss(self):
        plt.clf()

        plt.plot(self.epochs, self.loss, 'bo', label='Training loss')
        plt.plot(self.epochs, self.val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
        plt.savefig('loss_graph.png', bbox_inches='tight')

    def plot_accuracy(self):
        plt.clf()   # clear figure

        plt.plot(self.epochs, self.acc, label='Training acc')
        plt.plot(self.epochs, self.val_acc, label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
        plt.savefig('acc_graph.png', bbox_inches='tight')

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    plt.plot([1, 2, 3, 4], [10, 20, 25, 30], color='lightblue', linewidth=3)
    plt.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color='darkgreen', marker='^')
    plt.xlim(0.5, 4.5)
    plt.show()
    plt.savefig('all.png',bbox_inches='tight')
    # plt.show()
