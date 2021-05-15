import time

### util functions

def log_with_time(logmsg):
    logt = time.time()
    print("[{:.4f}] {}".format(logt, logmsg) )


class Timer():
  def __init__(self):
    self._start_time = time.time() 

  def reset(self):
    self._start_time = time.time()

  def elapsed(self):
    self._elapsed_time = time.time() - self._start_time
    return self._elapsed_time

  def check(self, check_time):
    if(self.elapsed() > check_time):
      self.reset()
      return True
    else:
      return False
