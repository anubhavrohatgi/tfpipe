import tensorflow as tf

from tfpipe.core.libs.tensorflow import resize
from tfpipe.pipeline.pipeline import Pipeline


class ConvertTFLite(Pipeline):
    """ The pipeline task for converting a TF-YOLOv4 model to tflite. """

    def __init__(self, weights, size, qm):
        model = tf.keras.models.load_model(weights, compile=False)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if qm == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT] # optimize latency
            converter.target_spec.supported_types = [tf.float16]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            converter.allow_custom_ops = True
        elif qm == 'int8':
            converter.target_spec.supported_ops = [tf.int8]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            converter.allow_custom_ops = True
            converter.representative_dataset = representative_data_gen
        
        self.tflite_model = converter.convert()
    
    def map(self, output):
        with open(output, "wb") as f:
            f.write(self.tflite_model)
        
        print(f"TFLite Model Written to: {output}")