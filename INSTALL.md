Instillation Instructions
=========================

I'm using TensorFlow to do the deep learning so you'll need to[follow the setup instructions](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html) for that first.

I'm on a Mac and used the Python 2, CPU version:

```
pip install -r https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0-py2-none-any.whl
```

Assuming you have made a new virtual environment to capture all your module installs

```
pip install -r requirements.txt
sputnik --name spacy --repository-url http://index.spacy.io install en==1.1.0
```

The second line here installs the english language model used by the [spaCy NLP](http://spacy.io/) library which handles the tedious POS tagging details.

Environmental Variables
-----------------------

Are listed in `.env`. I have `autoenv` installed which `source .env` every time I `cd` to the project folder so these are added automatically for me (as well as the virtualenv being setup too :). This is what sets up Keras to use TensorFlow but it should work ether way (but i haven't tested with the default 'theano')
