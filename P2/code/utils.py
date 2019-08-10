
import numpy as np


class Evaluation():
    def __init__(self,predictions,y_True):
        self.allNum = len(predictions)*80
        self.TP=0
        self.TN=0
        self.FP=0
        self.FN=0
        self.calculate(predictions,y_True)
        self.printing()
    def printing(self):
        print(self.TP,
        self.TN,
        self.FP,
        self.FN)
        
        
        print('Accuracy:',(self.TP+self.TN)/(self.allNum))
        Recall = (self.TP)/(self.TP+self.FN)
        Precision = (self.TP)/(self.TP+self.FP)
        print('Recall:',Recall)
        print('Precision:',Precision)
        print('F1:',(2*Recall*Precision)/(Precision+Recall))
    def calculate(self,predictions,y_True):
        answer ={0:[1,0,0,0],1:[0,1,0,0],2:[0,0,1,0],3:[0,0,0,1]}
        predictions_binary = []
        for item in predictions:
            answer_temp = []
            for i in range(0,80,4):
                #print(item[i:i+4])
                #print(np.argmax(item[i:i+4]))
                answer_temp.append(answer[np.argmax(item[i:i+4])])
            l2=self.dig_lists(answer_temp)
            predictions_binary.append(l2)
            
        
        for i,item in enumerate(predictions_binary):
            for j,num in enumerate (item):
                if num==y_True[i][j]:
                    if num ==1:
                        self.TP+=1
                    elif num ==0:
                        self.TN+=1
                elif num!=y_True[i][j]:
                    if num ==1:
                        self.FP+=1
                    elif num ==0:
                        self.FN+=1
        print(counter) 
        
    def dig_lists(self,l):
        output = []
        for e in l:
            if isinstance(e, list):
                output += self.dig_lists(e)
            else:
                output.append(e)
        return(output)