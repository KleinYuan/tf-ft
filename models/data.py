from copy import deepcopy


class DataSets:

    # In here, we wanna support multiple ways of organizing datasets, like raws images, hd5, tfrecords, ...
    # We only provide a model here and eventually,
    # you wanna override most functions in this model with your own loader ...

    def __init__(self):
        print 'Initializing a dataset ...'
        self.x_data = None
        self.y_data = None

    def load(self, mode, fn):
        print 'Loading data ...'
        self._load(mode=mode, fn=fn)
        print 'Data (%s) loaded !' % mode

    def _load(self, mode, fn):
        print 'To be overrided!'
