import tensorflow as tf
import sonnet as snt 



from graph_nets import _base
from graph_nets import blocks

from graph_nets import modules
from graph_nets.demos_tf2 import models


# model = models.EncodeProcessDecode(edge_output_size=1)

class LimitEncodeProcessDecode(snt.Module):
    def __init__(self,
               edge_output_size=None,
               node_output_size=None,
               global_output_size=None,
               name="LimitEncodeProcessDecode"):
        super(LimitEncodeProcessDecode, self).__init__(name=name)

        self._model = models.EncodeProcessDecode(
            edge_output_size = edge_output_size,
            node_output_size=node_output_size,
            global_output_size=global_output_size)

        # edge_fn = lambda: snt.Linear(edge_output_size, name="edge_final_output")
        # self._output_transform = modules.GraphIndependent(edge_fn)


    def __call__(self, input_op, num_processing_steps):
        output_ops = self._model(input_op, num_processing_steps)        
        final_ops = [output_op.replace(edges = tf.math.scalar_mul(5,tf.nn.tanh(output_op.edges))) for output_op in output_ops]
        return final_ops
        # final_output = output_ops[-1]
        # return final_output.replace(edges = tf.nn.tanh(final_output.edges))