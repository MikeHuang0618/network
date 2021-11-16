# !usr/bin/env python
# -*- coding:utf-8 _*-

import tensorflow as tf
import os
from tensorflow.python.keras import backend as K
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def train():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    checkpoint_path = "./models/checkpoint.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[cp_callback])
    model.evaluate(x_test, y_test, verbose=2)
    model.save('./models/model.h5')

def h52pb(h5_path):
    model = tf.keras.models.load_model(h5_path)
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./models",
                      name="frozen_graph.pb",
                      as_text=False)
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./models",
                      name="frozen_graph.pbtxt",
                      as_text=True)

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    print("-" * 50)
    print("Frozen model layers: ")
    layers = [op.name for op in import_graph.get_operations()]
    if print_graph == True:
        for layer in layers:
            print(layer)
    print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

def test_pb():
    with tf.io.gfile.GFile("./models/frozen_graph.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    wrap_frozen_graph(graph_def=graph_def,
                        inputs=["x:0"],
                        outputs=["Identity:0"],
                        print_graph=True)


if __name__ == "__main__":
    input_file = './models/model.h5'
    train()
    h52pb(input_file)
    test_pb()