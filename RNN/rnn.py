import tensorflow as tf


def lstm(input_feature=1000, output_feature = 46):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_feature, 80),
        tf.keras.layers.LSTM(80),
        tf.keras.layers.Dense(output_feature, activation='sigmoid')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
