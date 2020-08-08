import caffe
import numpy as np

class SigmoidCrossEntropyWeightLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check for all inputs
        params = eval(self.param_str_)
        self.cls_weight = float(params["cls_weight"])
        if len(bottom) != 2:
            raise Exception("Need two inputs (scores and labels) to compute sigmoid crossentropy loss.")

    def reshape(self, bottom, top):
        # check input dimensions match between the scores and labels
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference would be the same shape as any input
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # layer output would be an averaged scalar loss
        top[0].reshape(1)

    def forward(self, bottom, top):
        score=bottom[0].data
        label=bottom[1].data

        first_term = -(label-1)*score
        second_term = -((1-self.cls_weight)*label - 1)*np.log(1+np.exp(-score))

        top[0].data[...] = np.sum(first_term + second_term)

        sig = 1.0/(1.0+np.exp(-score))
        self.diff = ((self.cls_weight-1)*label+1)*sig - self.cls_weight*label
        if np.isnan(top[0].data):
                exit()

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...]=self.diff
