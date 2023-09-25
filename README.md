#GPT4All LLM Examples

### GPT4All
[GPT4All Github Project](https://github.com/nomic-ai/gpt4all)


[GPT4All Model Explorer](https://gpt4all.io/index.html)
Scroll down to it; these guys have never heard of A tags or CSS IDs.

<pre>
generic-gpt4all.py - Generic chat interface to a local GPT4All model
</pre>
  Example:
```
bash$ python generic-gpt4all.py --dbug --time ~/.cache/gpt4all/wizardLM-13B-Un
censored.ggmlv3.q4_0.bin
[INFO] Instantiating model: wizardLM-13B-Uncensored.ggmlv3.q4_0.bin
[TIME] Model instantiation: 2.444 sec
[INFO] Creating Prompt Template from:
You are a helpful, highly-educated personal assistant with a wide range of 
experience. Answer all questions as thoroughly as possible.
[INFO] Creating memory named 'chat_history'
[INFO] Creating Langchain LLMChain
> 'sup.
> When was the Battle of Hastings?
> (Elapsed: 132.798 sec)
> AI: The Battle of Hastings took place on October 14, 1066. Would you like me 
to provide more information about the battle or any other historical event relat
ed to it?
# Yes, who were the sides?
#> (Elapsed: 197.638 sec)
#> System: The Battle of Hastings was fought between the Norman-French army o
f William, Duke of Normandy, and the English army of King Harold Godwinson. The 
Normans emerged victorious, which led to the eventual conquest of England by Wil
liam in 1066. Would you like me to provide more information about the battle or 
any other historical event related to it?
#> What were the lasting geopolitical effects?
#> (Elapsed: 211.068 sec)
#> System: The Battle of Hastings had significant geopolitical effects, as it 
marked the beginning of a period of Norman rule in England that lasted for sever
al centuries. It also led to the eventual formation of the United Kingdom throug
h the union of Scotland and England in 1707. Would you like me to provide more i
nformation about the battle or any other historical event related to it?
```
