import tensorflow as tf

class Score_ET(tf.keras.metrics.Metric):
    def __init__(self, name='score', **kwargs):
        super(Score_ET, self).__init__(name=name, **kwargs)
        self.score = self.add_weight(name='tp', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        mean_y = tf.reduce_mean(y_true)
        y_sum = tf.reduce_sum(tf.math.square(y_true-mean_y))
        pred_sum = tf.reduce_sum(tf.math.square(y_true-y_pred))
        R2 = 1-pred_sum/y_sum
        self.score.assign_add(R2)
    def result(self):
        return self.score
    
class Score_MSPE(tf.keras.metrics.Metric):
    def __init__(self, name='score_mspe', **kwargs):
        super(Score_MSPE, self).__init__(name=name, **kwargs)
        self.score = self.add_weight(name='tp', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        relative = tf.math.square((y_true-y_pred))
        #relative = tf.divide(relative,y_true)
        _score = tf.reduce_mean(relative)
        #_score = tf.math.sqrt(tf.math.reduce_mean(pred_sum))
        self.score.assign_add(_score)
    def result(self):
        return self.score

