import tensorflow as tf
import rnn

def main():
  input_word = 1000
  (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=input_word, test_split=0.2)

  max_length = 100
  x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)
  x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)

  y_train = tf.keras.utils.to_categorical(y_train)
  y_test = tf.keras.utils.to_categorical(y_test)

  model = rnn.lstm(input_feature=input_word)

  hist = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_split=0.2)

  test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

  result = hist.history
  print('Test loss:', test_loss)
  print('Test accuracy:', test_acc)
  '''
  tr_loss = result['loss']
  accuracy = result['accuracy']
  val_loss = result['val_loss']
  val_accuracy = result['val_accuracy']

  new_plot('Train Loss & Validation Loss', 'epochs', 'Traing loss', tr_loss, val_loss, 'train', 'validation')
  new_plot('Train Accuracy & Validation Accuracy', 'epochs', 'Accuracy', accuracy, val_accuracy, 'train', 'validation')
  '''
