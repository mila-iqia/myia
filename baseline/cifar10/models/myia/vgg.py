import numpy as np

class MiniVGG:
    """
    A mini version of VGG.
    """

    def __init__(self, lr=.003, dropout=0.5, name='mini_vgg'):
        self.lr = lr
        self.dropout = 0.5
        self.name = name

    def update(self, images, labels):
        """update the model with one mini-batch data
        """
        loss = 0.
        return loss

    
    def predict(self, images):
        """update the model with one mini-batch data
        """
        probs_pred = np.zeros((len(images), 10))
        return probs_pred


    def evaluate(self, images, labels):
        """evaluate the model with one mini-batch data
        """
        loss = 0.
        acc = 0.
        return loss, acc
 
    def save(self, model_dir='./saved_models/'):
        """save model
        """
        pass

    def load(self, model_dir='./saved_models/'):
        """load model
        """
        pass

    def get_var_count(self):
        """number of parameters
        """
        return int(0)
