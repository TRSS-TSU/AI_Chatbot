# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# !jupytext --to py chatBot.ipynb  

from transformers import AutoModelForQuestionAnswering, pipeline, AutoTokenizer
import numpy as np
import pandas as pd
import os  
import warnings
warnings.filterwarnings("ignore")
from threading import Thread
import sys
sys.path.append(os.path.join(os.getcwd(), "utils"))
from metrics import compute_exact

class ChatBot:
    """
    ChatBot class
    """
    model1 = None
    model2 = None
    model3 = None
    tokenizer1 = None
    tokenizer2 = None
    tokenizer3 = None
    context = None

    def __init__(self, models_list):
        """
        Constructor for ChatBot class

        Does not initialize context; must use read_file to assign context
        This will let us change context easily without having to reload everything else
        """

        #if we're only using one model
        if len(models_list) == 1:
            self.model1 = AutoModelForQuestionAnswering.from_pretrained(models_list[0])
            self.tokenizer1 = AutoTokenizer.from_pretrained(models_list[0], model_max_length =int(1e30))

        #if we're using 3 models for ensemble method
        elif len(models_list) == 3:
            self.model1 = AutoModelForQuestionAnswering.from_pretrained(models_list[0])
            self.model2 = AutoModelForQuestionAnswering.from_pretrained(models_list[1])
            self.model3 = AutoModelForQuestionAnswering.from_pretrained(models_list[2])

            self.tokenizer1 = AutoTokenizer.from_pretrained(models_list[0], model_max_length =int(1e30))
            self.tokenizer2 = AutoTokenizer.from_pretrained(models_list[1], model_max_length =int(1e30))
            self.tokenizer3 = AutoTokenizer.from_pretrained(models_list[2], model_max_length =int(1e30))


    def read_file(self, path):
        """
        Read file contents and save to context variable

        Parameters:
            path - path to context file

        Return:
            context - the string of context 
        """
        with open(path, 'r', encoding="utf8") as file:
            context  = file.read().replace('\n', '')

        self.context = context #save to self
        return context

    def save_results(self, question, result, ans1, ans2, ans3):
        """
        save results of the different models to results.csv

        Parameters:
            question - the question asked
            result - answer to the question
            ans1 - model1's answer
            ans2 - model2's answer
            ans3 - model3's answer

        Return:
            None
        """
        csv_file = "result/results.csv"
        if os.path.exists(csv_file):
            headers= False
        else:
            headers= True
        
        df = pd.DataFrame(list(zip([question], [result], [ans1],[ans2],[ans3])),
                columns =["question", "prediction", "model1_result","model2_result","model3_result" 
                            ])
        
        df.to_csv(csv_file, mode = 'a',encoding='utf-8', index=False, header = headers)

    def preprocess(self, text):
        """
        helper function - add question mark at the end

        Parameters:
            text

        Return:
            text - text after processing
        """
        #if "?" not in text:
        #    text = text+"?"

        if not text[len(text) - 1] == "?":
            text += "?"
        
        return text
  
    def get_ensemble_answer(self, question):
        """
        Use the ensemble method to get an answer

        Parameters:
            question - the question to ask

        Return:
            result - the answer to the question
        """
        question = self.preprocess(question)

        self.tokenizer1.encode(question, truncation = True, padding = True, verbose=False)
        self.tokenizer2.encode(question, truncation = True, padding = True, verbose=False)
        self.tokenizer3.encode(question, truncation = True, padding = True, verbose=False)


        #init pipeline
        m1 = pipeline("question-answering", model=self.model1 , tokenizer = self.tokenizer1)
        m2 = pipeline("question-answering", model=self.model2 , tokenizer = self.tokenizer2)
        m3 = pipeline("question-answering", model=self.model3 , tokenizer = self.tokenizer3)

        #generating prediction
        answer1 = m1({ 'question' : question, 'context' : self.context})
        answer2 = m2({ 'question' : question, 'context' : self.context})
        answer3 = m3({ 'question' : question, 'context' : self.context})

        ans = [answer1["answer"] ,answer2["answer"],answer3["answer"]]
        
        
        a1 = compute_exact(ans[0], ans[1]) + compute_exact(ans[0], ans[2])
        a2 = compute_exact(ans[1], ans[0]) + compute_exact(ans[1], ans[2])
        a3 = compute_exact(ans[2], ans[0]) + compute_exact(ans[2], ans[1])
        calc = [a1,a2,a3]
        
        if np.sum(calc) == 0:
            result = "no answer"
        else:
            result = ans[calc.index(max(calc))]

        self.save_results(question, result, answer1["answer"] ,answer2["answer"],answer3["answer"])

        return result

    
    def model_process(self, question, model, tokenizer):
        """
        The process that a model goes through to get an answer
        Currently used in concurrent_get_ensemble_answer

        Parameters:
            question - question to ask
            model - which model to use
            tokenizer - which tokenizer to use that's connected to the model

        Return:
            answer
        """

        tokenizer.encode(question, truncation = True, padding = True, verbose=False)
        m = pipeline("question-answering", model=model , tokenizer = tokenizer)
        answer = m({ 'question' : question, 'context' : self.context})

        return answer
    
    def concurrent_get_ensemble_answer(self, question):
            """
            Use the ensemble method to get an answer

            Parameters:
                question - the question to ask

            Return:
                result - the answer to the question
            """

            question = self.preprocess(question)

            #Need to use threads with return values in order to get the answer
            twrv1 = ThreadWithReturnValue(target=self.model_process, args=(question, self.model1, self.tokenizer1,))
            twrv2 = ThreadWithReturnValue(target=self.model_process, args=(question, self.model2, self.tokenizer2,))
            twrv3 = ThreadWithReturnValue(target=self.model_process, args=(question, self.model3, self.tokenizer3,))

            #start threads
            twrv1.start()
            twrv2.start()
            twrv3.start()

            #wait for threads to finish and get the answers
            answer1 = twrv1.join()
            answer2 = twrv2.join()
            answer3 = twrv3.join()

            """
            TODO: clean up the code above to look more like this
            threads = [
                ThreadWithReturnValue(target=self.model_process, args=(question, self.model1, self.tokenizer1,)),
                ThreadWithReturnValue(target=self.model_process, args=(question, self.model2, self.tokenizer2,)),
                ThreadWithReturnValue(target=self.model_process, args=(question, self.model3, self.tokenizer3,))
            ]

            for t in threads:
                t.start()

            ans = []
            for t in threads:
                ans.append(t.join()["answer"])"""
            

            ans = [answer1["answer"] , answer2["answer"], answer3["answer"]]
            
            a1 = compute_exact(ans[0], ans[1]) + compute_exact(ans[0], ans[2])
            a2 = compute_exact(ans[1], ans[0]) + compute_exact(ans[1], ans[2])
            a3 = compute_exact(ans[2], ans[0]) + compute_exact(ans[2], ans[1])
            calc = [a1,a2,a3]
            
            if np.sum(calc) == 0:
                result = "no answer"
            else:
                result = ans[calc.index(max(calc))]

            self.save_results(question, result, answer1["answer"] ,answer2["answer"], answer3["answer"])

            return result

    def single_model_answer(self, question):
        """
        Utilizes only one model to get a faster answer
        """

        #get the answer from the single model method
        answer = self.model_process(question, self.model1, self.tokenizer1)
        
        start1 = answer["start"]
        start2 = answer["end"]
        
        #search start of sentence
        for i in range(1500):
            count1 = start1-i+1
            if self.context[count1] == ".":
                break
        #search end of sentence
        for i in range(1500):
            count2 = start2 + 1
            if self.context[count2] == ".":
                break
            
        result = self.context[count1+1: count2-1]

        return result 


class ThreadWithReturnValue(Thread):
    """
    ThreadWithReturnValue class utilizes 'from threading import thread' to provide a
    custom thread class that allows us to retrieve return values of functions
    """
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def ensemble_run():
    """
    run the program and print out to command line
    """

    ## Constants
    NAMES = ["models/bert-large-uncased-whole-word-masking-squad2",	
            "models/roberta-base-squad2",	
            "models/minilm-uncased-squad2"
            ]
    CONTEXT_PATH = "data/context/NTLM_context.txt"

    # create ChatBot object, import names list and read the context
    cb = ChatBot(NAMES)
    cb.read_file(CONTEXT_PATH)

    #CLI loop
    exit_conditions = ("exit")
    print("type \'exit\' or CTRL-C/CTRL-D to exit")
    while True:
        try:
                
            question = input("> ")
            
            if question in exit_conditions:
                break
            else:
                response = cb.concurrent_get_ensemble_answer(question)

                print(response)

        # Press ctrl-c or ctrl-d on the keyboard to exit
        except (KeyboardInterrupt, EOFError, SystemExit):
            break

def single_model_run():
    """
    run the program and print out to command line
    while using the single model method
    """

    ## Constants
    NAMES = ["models/roberta-base-squad2"]
    CONTEXT_PATH = "data/context/day5_context_cleanup.txt"

    # create ChatBot object, import names list and read the context
    cb = ChatBot(NAMES)
    cb.read_file(CONTEXT_PATH)

    #CLI loop
    exit_conditions = ("exit")
    print("type \'exit\' or CTRL-C/CTRL-D to exit")
    while True:
        try:
                
            question = input("> ")
            
            if question in exit_conditions:
                break
            else:
                response = cb.single_model_answer(question)

                print(response)

        # Press ctrl-c or ctrl-d on the keyboard to exit
        except (KeyboardInterrupt, EOFError, SystemExit):
            break

if __name__== "__main__":
    single_model_run()
