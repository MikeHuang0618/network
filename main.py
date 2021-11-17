# !usr/bin/env python
# -*- coding:utf-8 _*-

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.optimizers import Adam

def train():
    cifar10 = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    model = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                     input_shape=(32, 32, 3))
    out = model.output
    out = Flatten()(out)
    out = Dropout(0.5)(out)
    out_layer = Dense(10, activation='softmax', name='softmax')(out)

    model_final = Model(inputs=model.input, outputs=out_layer)
    for layer in model_final.layers[:2]:
        layer.trainable = False
    for layer in model_final.layers[2:]:
        layer.trainable = True

    model_final.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model_final.summary()
    checkpoint_path = "./models/checkpoint.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    model_final.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[cp_callback])
    model_final.evaluate(x_test, y_test, verbose=2)
    model_final.save('./models/model.h5')

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
    # h52pb(input_file)
    # test_pb()