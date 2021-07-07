# nlp_vs_humans
Experiment to compare how NLP models see text compared with humans

## Articles

This project was written up as two articles on Towards Data Science:[Part 1](https://towardsdatascience.com/are-we-thinking-what-theyre-thinking-d1224445bbb9) and [Part 2](https://medium.com/@j_casswell/are-we-thinking-what-theyre-thinking-part-2-65921e9c46b0)

## Overview

In a [previous project](https://github.com/jaycee14/tv_sentiment) I was using NLP to detect sentiment and then interrogating the model to tell me why it had come to each conclusion; by showing which words it had considered positive & negative. It provoked the question - is this the same way a human would interpret the same words? As NLP grows in influence,and thinking with a business hat on, it seemed like it would be good to know how the model was viewing the input and were some models and methods better at getting a human perspective that others. The model has learnt from a source of data - does that source agree with you, your customers or the world in general. Who knows [what has been teaching](https://en.wikipedia.org/wiki/Tay_(bot)) your model.

## Phase 1 - Getting Data

But first I'd need some data and I didn't want to generate this all myself otherwise it would be a comparison between myself and the machine. So I needed a tool to collect multiple viewpoints...

### Superintendent

I'd seen a talk on [Superintendent](https://superintendent.readthedocs.io/en/latest/index.html) at PyData London 2019 and thought it could help me out. And I wanted to learn about this really useful tool, which along with Voila is a great way to make basic web interfaces. Some quick hacking showed it could be adapted from pre-defined labels to using the input words of a phrase as labels. I also needed to add a database backend to hold the input and capture the output. It was easiest to fork the original and make my adjustments [there.](https://github.com/jaycee14/superintendent)

### Docker Setup & Remote Deployment

Like my last project I was using Docker to create a virtualised environment that could be remotely deployed. It took a bit of effot but I got docker-machine working with my remote server and it made deployments and updates a breeze. (As always volumes are the killer...)

For input data I was using tweets about television shows that had been previously categorised and collecting them as either Positive or Negative as determined by the model.

#### Points to Improve on / ToDo

* Positional encoding
* Filter repeated phrases

## Phase 2 - Reviewing Data

Added: 

* results file (results_2020_11_15.json)
* Notebooks to process the tweets through two models (Captum for LSTM and Captum BERT)
* Notebook to clean up the selected worlds (clean human words)
* Notebook to combine the data sets and review the results (process results) 
