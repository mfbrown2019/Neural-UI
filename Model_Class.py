# Class for student account
class ModelEntry():
    # int to hold the student ids
    model_id = 1
    
    def __init__(self, title, model, activation, L1, L2, dropout, momentum, alpha, 
                 epochs, note, training_photo, heatmap, train_loss, train_accuracy, date):
        # Format the inital variabls
        ModelEntry.student_id += 1
        self.title = title
        self.model = model
        self.activation = activation
        self.L1 = L1
        self.L2 = L2
        self.dropout = dropout
        self.momentum = momentum
        self.alpha = alpha
        self.epochs = epochs
        self.note = note
        self.training_photo = training_photo
        self.heatmap = heatmap
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy
        self.date = date
        