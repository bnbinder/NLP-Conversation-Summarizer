we have a text classifier, one and zero

only split sentences of moderators, which we need to define manually
keep text same for candidates

question - we have a good way, but best way?
topic - risky, but fallback if question isnt good, but how to implement i dont know

if moderator text is question or new topic
    summarize candidates responses -- all text until next question or topic
    break if encounter new question or topic from moderator
    
new dictionary with one word description, with multi word description when found next question

economy - specific thing - gjireob jwen
        - speiciic thing - ngui howjubn 
        - specific thing - gnbuoejwr


# ALMOST DONE - NEED TO FINE TUNE SOME THINGS
<br>
### YOU NEED A GPU OR A BIG PHAT CPU!!! I DONT MAKE THE RULES!!!!

venvs are optional, but highly recommended just in general
<br>
To create your virtual enviornment, open the terminal in your repository folder and enter
```
/path/to/python.exe -m venv venv_name
```

Then open an admin access terminal and run this code so you can activate the virtual enviornment
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
```

Then open the terminal in your repository folder and enter
```
venv_name/Scripts/activate
```

install?? https://nlp.stanford.edu/projects/glove/ - try the big one on the beefy computer.

Make sure you have installed c++ tools from this link
<br>
https://visualstudio.microsoft.com/visual-cpp-build-tools/

https://developer.nvidia.com/cuda-toolkit <-- idk lol

and install these libraries
```

pip install sentencepiece

pip install POT
pip install rouge_score
pip install gensim
pip install nltk
pip install textstat
pip install sentence-transformers
pip install sentencepiece
pip install spacy
python -m spacy download en_core_web_sm
pip install bitsandbytes
pip install -U "huggingface_hub[cli]"
pip install accelerate
pip install transformers
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Once your done running your code, run this in a terminal inside your repository folder
```
venv_name/Scripts/deactivate
```

And run this in an admin access terminal
```
Set-ExecutionPolicy -ExecutionPolicy Undefined
```

# Sources and Help
https://medium.com/@khang.pham.exxact/text-classification-with-bert-7afaacc5e49b