from datetime import datetime

class Config(object):
  
  root = None
  
  features_considered = None
  features_considered_change = {}
  
  TARGET = None
  # TARGET_INDEX = features_considered.index(TARGET)
  
  INDEX_COLUMN = None

  BATCH_SIZE = 256
  BUFFER_SIZE = 10000

  STEP = None
  STEP_TO_FUTURE = 1

  past_history = None
  future_target = None

  EVALUATION_INTERVAL = 200
  EPOCHS = None
  
  date = datetime.now().strftime("%y%m%d_%H%M%S")
  save_root = None

  # model_file_name = TARGET + '_' + date
  # model_save_root = save_root + '/' + model_file_name
  # model_save_path = model_save_root + '/' + model_file_name + '.h5'
  
  def __init__(self):
    
    assert (self.future_target % self.STEP_TO_FUTURE) == 0
    
    self.csv_dataset_train_folder = self.root + '/Training'
    self.csv_dataset_valid_folder = self.root + '/Validation'
    self.csv_dataset_test_folder = self.root + '/Test'
    
    self.TARGET_INDEX = self.features_considered.index(self.TARGET)
    
    self.model_file_name = self.TARGET + '_' + self.date + '_' + str(self.past_history) + 'to' + str(self.future_target)
    self.model_save_root = self.save_root + '/' + self.model_file_name
    self.model_save_path = self.model_save_root + '/' + self.model_file_name + '.h5'
    
  def to_dict(self):
      return {a: getattr(self, a)
              for a in sorted(dir(self))
              if not a.startswith("__") and not callable(getattr(self, a))}

  def display(self):
      """Display Configuration values."""
      print("\nConfigurations:")
      for key, val in self.to_dict().items():
          print(f"{key:30} {val}")
      # for a in dir(self):
      #     if not a.startswith("__") and not callable(getattr(self, a)):
      #         print("{:30} {}".format(a, getattr(self, a)))
      print("\n")


class Config_BABY_MIDDLE(Config):
    features_considered = ['water_temp', 'water_do', 'water_orp', 'tank_lux', 'maturation_period', 'body_length', 'feed_type', 'feed_frequency']
    features_considered_change = {'feed_type': { 'C': 0, 'F': 1}}
  
    TARGET = 'body_length'
  
    INDEX_COLUMN = 'no'
    
class Config_MAMA(Config):
    features_considered = ['water_temp', 'water_do', 'water_orp', 'tank_lux', 'maturation_period', 'photoperiod', 'body_weight', 'gonads_weight', 'gsi']
    
    
    TARGET = 'gsi'
    
    INDEX_COLUMN = 'no'