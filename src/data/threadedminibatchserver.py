'''
Created on Apr 27, 2015

@author: wohlhart
'''
import threading

class ThreadedMinibatchServer(threading.Thread):
    '''
    classdocs
    '''


    def __init__(self, numBatchesToStore):
        '''
        Constructor
        '''
        threading.Thread.__init__(self)
        
        self.batch_available_sem = threading.Semaphore(0)  # acquire/decrement this if you read a minibatch, release/increment it if you produced one
        #self.batches_needed_cond = threading.Condition() 
        self.empty_slots_available_sem = threading.Semaphore(numBatchesToStore) # acquire/decrement this it if you produced one, release/increment if you read one
        self.internal_manipulations_lock = threading.Lock()
        
        self.numBatchesOnStorage = 0
        self.numBatchesToStore = 0
        self.currentInsertPosition = 0
        self.currentExtractPosition = 0
        
    def run(self):
        '''
        Thread Main Loop
          generates mini_batches and  
        '''
        self.initMiniBatchStorage()
        
        while True:

            # check if there are empty slots to fill, if not -> wait
            self.empty_slots_available_sem.acquire()
            
            # otherwise create a batch
            mb = self.generateMinibatch()
            with self.internal_manipulations_lock:                    
                self.setMinibatch(mb,self.currentInsertPosition)
                self.currentInsertPosition = (self.currentInsertPosition + 1) % self.numBatchesToStore
                self.numBatchesOnStorage += 1
                
            self.batch_available_sem.release()  # signal to waiting consumers
        
        
    def getNextMinibatch(self):
        
        self.batch_available_sem.acquire()  # try to acquire one
        
        with self.internal_manipulations_lock:
            extPos = self.currentExtractPosition
            self.currentExtractPosition = (self.currentExtractPosition + 1) % self.numBatchesToStore
            self.numBatchesOnStorage -= 1

        # Should this also be in the lock ?
        #   I think it doesn't have to be. 
        #   The idx is ours and the position is not going to be overriden before we signal empty_slots_available_sem, or is ?
        data = self.getMinibatch(extPos)  
        
        self.empty_slots_available_sem.release()
        
        return data
     
            
    def initMiniBatchStorage(self):  
        '''
        setup storage space for minibatches
          (implement in derived classes)
        ''' 
        pass
        
    def generateMinibatch(self):
        '''
        generate a minibatch and return it
          (implement in derived classes)
        ''' 
        return 0
        
    def getMinibatch(self,idx):
        '''
        retrieve minibatch from position idx
          (implement in derived classes)
        ''' 
        return 0
        
    def setMinibatch(self,data,idx):
        '''
        set minibatch data at position idx
          (implement in derived classes)
        ''' 
        pass
            