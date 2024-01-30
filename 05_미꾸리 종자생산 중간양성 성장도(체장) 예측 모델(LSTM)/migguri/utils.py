import numpy as np
import matplotlib.pyplot as plt

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, step_to_future=1, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    indices_target = range(i, i+target_size, step_to_future)
    data.append(dataset[indices])
    
    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[indices_target])

  return np.array(data), np.array(labels)


def multi_step_plot(history, true_future, prediction,
                    TARGET, TARGET_INDEX, STEP,
                    save_path=None,
                    rangeX=None, rangeY=None):
  
  def create_time_steps(length):
    return list(range(-length, 0))
  
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, TARGET_INDEX]), label='History')
  plt.plot(num_in, np.array(history[:, TARGET_INDEX]), 'bo', label='History dot')
  plt.plot(np.arange(num_out), np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out), np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.title(f'Target: {TARGET}')
  
  if rangeX:
    plt.xlim(rangeX[0], rangeX[1])
  
  if rangeY:
    plt.ylim(rangeY[0], rangeY[1])
  
  if save_path:
    plt.savefig(save_path)
  else:
    plt.show()
  
  plt.close()

def multi_step_plot_multi_pred(history, true_future, predictions,
                               TARGETS, TARGET_INDEX, STEP, save_path=None):
  
  def create_time_steps(length):
    return list(range(-length, 0))

  num_targets = len(TARGETS)
  plt.figure(figsize=(12, 6*num_targets))

  for i, target_index in enumerate(TARGET_INDEX):
    plt.subplot(num_targets, 1, i+1)
    plt.plot(create_time_steps(len(history)),
             np.array(history[:, target_index]), label='History')
    plt.plot(np.arange(len(true_future))/STEP, np.array(true_future), 'bo',
             label='True Future')
    if predictions.any():
      plt.plot(np.arange(len(predictions))/STEP, np.array(predictions)[:, i], 'ro',
               label='Predicted Future')
    plt.legend(loc='upper left')
    plt.title(f'Target: {TARGETS[i]}')

  if save_path:
    plt.savefig(save_path)
  else:
    plt.show()
  
  plt.close()
  
def multi_step_plot_test(history, true_future, prediction,
                    TARGET, TARGET_INDEX, STEP,
                    startX, rangeX, rangeY,
                    save_path=None):
  
  def create_time_steps(length):
    return list(range(-length, 0))
  
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, TARGET_INDEX]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.title(f'Target: {TARGET}')
  
  if save_path:
    plt.savefig(save_path)
  else:
    plt.show()
  
  plt.close()
  
  
def plot_train_history(history, title, save_path, validtation=True):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  if save_path:
    plt.savefig(save_path)
  else:
    plt.show()
  
  plt.close()
  
def class2dict(cls):
    return {a: getattr(cls, a)
            for a in dir(cls)
            if not a.startswith("__") and not callable(getattr(cls, a))}

def dict2class(input_dict):
    new_class = type('Config', (object,), input_dict)
    return new_class

def meatadata_display(data):
    """Display Configuration values."""
    print("\nConfigurations:")
    for key, val in data.to_dict().items():
        print(f"{key:30} {val}")
    print("\n")