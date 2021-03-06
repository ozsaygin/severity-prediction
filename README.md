# Bug Severity Detection Tool

Bug Severity Detection Tool is a software engineering tool which designed
and implemented in the domain of project supported by Siemens Turkey and 
Sabanci University. The tool helps developer to detect the severity of a bug 
report without trouble by using machine learning algorithms to automate the 
process. Tool assumes that you have 3 severity levels as 'non-critical', 'normal'
and 'critical'. If you don't have a ready training dataset, you can use the sample
dataset called summaryList extracted from Bugzilla repositories.


## Requirements

This tool uses essential data science libraries in Python. Installing current
Anaconda distribution of Python would be beneficial for the user. Otherwiser,
you should install essential Python packages by using pip package manager. You
should use a Python distribution whose version is greater than 3.0. We also 
strongly advised to use it in a Python virtual enviroment as a precaution 
just in case of breaking your global environment.

* Python Version > 3.0

### Installing packages
Anaconda
```
conda create -n $YOUR_ENV$ python=3.6
source activate $YOUR_ENV$ 
conda install numpy pandas matplotlib jupyter scikit-learn pickle
conda install -c anaconda gensim 
```

Pip
```
pip install matplotlib scikit-learn numpy pandas gensim jupyter pickle
python3 -m pip install --user $YOUR_ENV$
```


## Installation & Usage

In Unix derivative operating systems, you run the program using the following
commands.

```
chmod +x train_and_validate.py eval.py
# You should change the python interpreter at the beginner of each file
# Find the location of interpreter using `which python` command on cli.

# You training file should be in csv format with such columns respectivly:
	'summary','severity','status','assigned_to','bug_id'

./train_and_validate.py --ifile $YOUR_TRAINING_FILE$
./eval.py

# You can quit from eval.py by simply typing `quit` or pressing CTRL+C
```

## Posibble Improvements

This tool uses 100x100x3 neural network architecture to process word vectors
generated by Gensim's Word2Vec model. Accuracy of model might be increased by
several changes. Firstly, the number of bug reports used in training can be
increased. Secondly, we realized the fact that convolutional neural networks
and recurrent neural networks can be used as a neural network architure. Thirdly,
in the project, ntlk's stopwords and tokenizers are used for the preprocessing stage.
However, through the end of project, we realized that tokenizer works poorly and 
stopwords are not enough as expected. Therefore, you may look for a new natural 
language toolkit to increase metrics.


## Licence
[MIT](LICENCE)
