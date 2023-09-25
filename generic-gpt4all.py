#!/usr/bin/env python
# Test code to get up-and-running with a downloaded GPT4All model.
# To download a model, go to https://gpt4all.io/index.html and find
# the "Model Explorer" buried in the page.  
# Example:
# bash$ python generic-gpt4all.py --dbug --time ~/.cache/gpt4all/wizardLM-13B-Uncensored.ggmlv3.q4_0.bin
#[INFO] Instantiating model: wizardLM-13B-Uncensored.ggmlv3.q4_0.bin
#[TIME] Model instantiation: 2.444 sec
#[INFO] Creating Prompt Template from:
#	
#You are a helpful, highly-educated personal assistant with a wide range of 
#experience. Answer all questions as thoroughly as possible.
#    
#[INFO] Creating memory named 'chat_history'
#[INFO] Creating Langchain LLMChain
#> 'sup.
#> What is the weather like?
#> (Elapsed: 85.506 sec)
#> When was the Battle of Hastings?
#> (Elapsed: 132.798 sec)
#> AI: The Battle of Hastings took place on October 14, 1066. Would you like me to provide more information about the battle or any other historical event related to it?
# Yes, who were the sides?
#> (Elapsed: 197.638 sec)
#>    System: The Battle of Hastings was fought between the Norman-French army of William, Duke of Normandy, and the English army of King Harold Godwinson. The Normans emerged victorious, which led to the eventual conquest of England by William in 1066. Would you like me to provide more information about the battle or any other historical event related to it?
#> What were the lasting geopolitical effects?
#> (Elapsed: 211.068 sec)
#    System: The Battle of Hastings had significant geopolitical effects, as it marked the beginning of a period of Norman rule in England that lasted for several centuries. It also led to the eventual formation of the United Kingdom through the union of Scotland and England in 1707. Would you like me to provide more information about the battle or any other historical event related to it?

import os
import argparse
import readline
import time

from langchain import LLMChain
from langchain.llms import GPT4All
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

#from langchain import PromptTemplate, LLMChain
#from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ----------------------------------------------------------------------
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

# ----------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
            description='',
            epilog="")
    parser.add_argument('--large', dest='large_model', action='store_true',
                    help='Use largest model available locally')
    parser.add_argument('--fine-tune', dest='fine_tune', action='store_true',
                    help='Use model that requires fine-tuning')
    parser.add_argument('--no-sample', dest='sample', action='store_false',
                    help='Disable decoding strategies like top-p top-k etc')
    parser.add_argument('--max-new-tokens', dest='max_new_tokens', default=1250,
                    help='Max length generated content [1250]')
    parser.add_argument('--max-tokens', dest='max_tokens', default=1500,
                    help='Total length of context window [1500]')
    # TODO: n_batch= 8,
    # TODO: n_parts= -1,
    # TODO: n_threads= 4,
    # TODO: repeat_last_n=64, # Last n tokens to penalize
    parser.add_argument('--temperature', dest='temperature', default=0.7,
                    help='Temperature (creativity) of model [0.7]')
    parser.add_argument('--top-k', dest='top_k', default=40,
                    help='# of top labels returned by the pipeline [40]')
    parser.add_argument('--top-p', dest='top_p', default=0.1,
                    help='Probability set of words must exceed [0.1]')
    parser.add_argument('--debug', dest='debug', action='store_true',
                    help='Enable debug output')
    parser.add_argument('--time', dest='time', action='store_true',
                    help='Time model calls')
    parser.add_argument(dest='model_path', type=str,
                    help='Path to model (e.g. "/tmp/ggml-wizardLM-7B.q4_2.bin"')
    
    args = parser.parse_args()
    return args

# ----------------------------------------------------------------------
if __name__ == '__main__':
    args = get_args()

    # ==============
    # create the LLM

    model_name = os.path.split(args.model_path)[1]
    if args.debug: print("[INFO] Instantiating model: %s" % model_name)
    with Timer() as t:
        llm = GPT4All(model=args.model_path,
                      # TODO: 
                      # n_batch= 8,
                      # n_parts= -1,
                      # n_threads= 4,
                      # repeat_last_n=64, # Last n tokens to penalize
                      max_tokens=args.max_tokens,
                      n_predict=args.max_new_tokens,
                      temp=args.temperature,
                      top_k=args.top_k,
                      top_p=args.top_p,
                      allow_download=False,
                      verbose=args.debug)
        
    if args.time: print("[TIME] Model instantiation: %.03f sec" % t.interval)

    # =====================
    # Create Prompt + Chain
    directive = """
You are a helpful, highly-educated personal assistant with a wide range of 
experience. Answer all questions as thoroughly as possible.
    """
    # TODO: additional prompt in args
    if args.debug: print("[INFO] Creating Prompt Template from:\n\t%s" % directive)
    memory_name = "chat_history"
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=directive),
        MessagesPlaceholder(variable_name=memory_name), 
        HumanMessagePromptTemplate.from_template("{human_input}"), 
    ])

    if args.debug: print("[INFO] Creating memory named '%s'" % memory_name)
    memory = ConversationBufferMemory(memory_key=memory_name, 
                                      return_messages=True)

    if args.debug: print("[INFO] Creating Langchain LLMChain")
    llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory, 
                         verbose=args.debug)

    # =================
    # Enter prompt loop
    display_prompt="'sup."
    while True:
        try:
            query = input("> " + display_prompt + "\n> ")
            if query == "q":
                break
            if query.strip() == "":
                continue
            try:
                with Timer() as t:
                    resp = llm_chain.predict(human_input=query)
            finally:
                if args.time:
                    print("> (Elapsed: %.03f sec)" % t.interval)
            display_prompt = str(resp)
        except EOFError:
            break

# Supported GPT4Alloptions:
"""
param allow_download: bool = False
    If model does not exist in ~/.cache/gpt4all/, download it.
param backend: Optional[str] = None
param cache: Optional[bool] = None
param callback_manager: Optional[BaseCallbackManager] = None
param callbacks: Callbacks = None
param echo: Optional[bool] = False
    Whether to echo the prompt.
param embedding: bool = False
    Use embedding mode only.
param f16_kv: bool = False¶ 
    Use half-precision for key/value cache.
param logits_all: bool = False
    Return logits for all tokens, not just the last token.
param max_tokens: int = 200
    Token context window.
param metadata: Optional[Dict[str, Any]] = None
    Metadata to add to the run trace.
param model: str [Required]
    Path to the pre-trained GPT4All model file.
param n_batch: int = 8
    Batch size for prompt processing.
param n_parts: int = -1
    Number of parts to split the model into. If -1, the number of parts is automatically determined.
param n_predict: Optional[int] = 256
    The maximum number of tokens to generate.
param n_threads: Optional[int] = 4
    Number of threads to use.
param repeat_last_n: Optional[int] = 64
    Last n tokens to penalize
param repeat_penalty: Optional[float] = 1.18
    The penalty to apply to repeated tokens.
param seed: int = 0
    Seed. If -1, a random seed is used.
param stop: Optional[List[str]] = []
    A list of strings to stop generation when encountered.
param streaming: bool = False
    Whether to stream the results or not.
param tags: Optional[List[str]] = None
    Tags to add to the run trace.
param temp: Optional[float] = 0.7
    The temperature to use for sampling.
param top_k: Optional[int] = 40
    The top-k value to use for sampling.
param top_p: Optional[float] = 0.1
    The top-p value to use for sampling.
param use_mlock: bool = False
    Force system to keep model in RAM.
param verbose: bool [Optional]
    Whether to print out response text.
param vocab_only: bool = False
    Only load the vocabulary, no weights.
"""
