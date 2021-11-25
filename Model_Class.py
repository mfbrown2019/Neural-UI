# Class for student account
class ModelEntry():
    # int to hold the student ids
    model_id = 1
    
    def __init__(self, model, activation, L1, L2, dropout, momentum, alpha, 
                 epochs, trainvalacc, train_accuracy):
        # Format the inital variabls
        ModelEntry.student_id += 1
        self.model = model
        self.activation = activation
        self.L1 = L1
        self.L2 = L2
        self.dropout = dropout
        self.momentum = momentum
        self.alpha = alpha
        self.epochs = epochs
        self.trainvalacc = trainvalacc
        self.train_accuracy = train_accuracy
        