import datetime

class Logger_LSTM:
    
    def __init__(self, test_log_path, print=True):
        self.test_log_path = test_log_path
        self.number = 0
        self.print = print
        
    def writemodelinfo(self, what, cmd, model, BATCH_SIZE, BUFFER_SIZE, EPOCHS, EVALUATION_INTERVAL, colums, target, past_history, future_target, step, step_to_future, dataset, save):
        
        with open(self.test_log_path, 'w', encoding='utf-8 sig') as f:
            f.write(f"{what} ")
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            
            f.write(f"\n{cmd}\n\n")
            
            f.write(f"model: {model}\n")
            f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
            f.write(f"BUFFER_SIZE: {BUFFER_SIZE}\n")
            f.write(f"EPOCHS: {EPOCHS}\n")
            f.write(f"EVALUATION_INTERVAL: {EVALUATION_INTERVAL}\n")
            f.write(f"colums: {colums}\n")
            f.write(f"target: {target}\n")
            f.write(f"past_history: {past_history}\n")
            f.write(f"future_target: {future_target}\n")
            f.write(f"step: {step}\n")
            f.write(f"step_to_future: {step_to_future}\n")
            f.write(f"dataset: {dataset}\n")
            f.write(f"save: {save}\n\n\n")
        
    def setallnumber(self, allnumber):
        self.allnumber = allnumber
    
    def setheader(self, header):
        self.number += 1
        text = f"\n■ {self.number} / {self.allnumber} ) {header}\n"
        
        with open(self.test_log_path, 'a', encoding='utf-8 sig') as f:
            f.write(text)
        
        if self.print:
            print(text)
            
    def write(self, content):
        text = f"\n{content}\n"
        
        with open(self.test_log_path, 'a', encoding='utf-8 sig') as f:
            f.write(text)
        
        if self.print:
            print(text)
            
    def end(self):
        with open(self.test_log_path, 'a', encoding='utf-8 sig') as f:
            f.write(f"\n{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        