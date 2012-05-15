# clojure-encog

Clojure wrapper for the encog (v3) machine-learning framework .

-from the official encog website:
---------------------------------
"Encog is an open source Machine Learning framework for both Java and DotNet. Encog is primarily focused on neural networks and bot programming. It allows you to  create many common neural network forms, such as feedforward perceptrons, self organizing maps, Adaline, bidirectional associative memory, Elman, Jordan and Hopfield networks and offers a variety of training schemes."

-from me:
---------
Encog has been around for almost 5 years hence, can be considered fairly mature and optimised. Apart from neural-nets version 3 introduced SVM and Bayesian classification. With this library which is a thin wrapper around encog, you can construct and train many types of neural nets in less than 10 lines of pure clojure code. The whole idea from the start, was to expose the user as less as possible to the Java side of things, thus eliminating any potential sharp edges of a rather big librabry like encog. Hopefully I've done a good job...feel free to try it out and more importantly, feel free to drop any comments/opinions/advice/critique etc etc...

ps: This is still work in progress. Nonethelessthe neural nets and training methods are pretty much complete - what's left at this point is data-models, randomization and the bayesian stuff...aaaa also I'm pretty sure we need tests :) ...  


## Usage

--Where is the jar(s)?
-------------------
As usual, it lives on clojars. 
Just add:
[org.encog/encog-core "3.1.0"] 
[clojure-encog "0.3.0-SNAPSHOT"]

to your :dependencies.


--Quick demo:
-------------
Ok, most the networks are already functional so let's go ahead and make one. Let's assume that for some reason we need a feed-forward net with 32 input neurons, 1 output neuron (classification), and 2 hidden layers with 50 and 10 neurons respectively...We don't really care about the activation function at this point because we are not going to do anything useful with this particular network.

(def network      ;;def-ing it here for demo purposes
    (make-network {:input   32
                   :output  1
                   :hidden [50 10]} ;;2 hidden layers
                  (make-activationF :sigmoid) 
                  (make-pattern     :feed-forward)))
                  
;;this is actually the neural pattern i used for my final year project at uni!                  

...and voila! we get back the complete network initialised with random weights.

Most of the constructor-functions (make-something) accept keyword based arguments. For the full list of options refer to documentation or source code. Don't worry if you accidentaly pass in wrong parameters to a network e.g wrong activation function for a specific net-type. Each concrete implementation of the 'make-network' multi-method ignores arguments that are not settable by a particular neural pattern!

Of course, now that we have the network we need to train it...well, that's easy too!
first we are going to need some dummy data...

(let [xor-input [[0.0 0.0] [1.0 0.0] [0.0 0.1] [1.0 1.0]]
      xor-ideal [[0.0] [1.0] [1.0] [0.0]] 
      dataset   (make-data :basic-dataset xor-input xor-ideal)
      trainer   ((make-trainer :back-prop) network dataset)])

;;notice how 'make-trainer' returns a function which itself expects some argumets.
;;in this case we're using simple back-propagation as our training scheme of preference.
;;feed-forward nets can be used with a variety of activations/trainers.

as soon as you have that, training is simply a matter of:

(train trainer 0.01 500 (RequiredImprovementStrategy. 5))
;train expects a training-method , error tolerance, iteration limit & strategies (optional)


and that's it really!
after training finishes you can start using the network as normal. For more in depth instructions consider looking at the 2 examples found in the examples.clj ns. These include the classic xor example (trained with resilient-propagation) and the lunar lander example (trained with genetic algorithm) from the from encog wiki/books.


--Other stuff...
----------------
Developed using Clojure 1.4
Should work with 1.3 but not lower than that!
Feel free to drop comments/opinions/suggestions/advice etc etc


This is still work in progress...If you're going to do any serious ML job with it, be prepared to write some Java simply because not everything has been wrapped. The plan is not to have to write any Java code by version 1.0. 

## License

Copyright © 2012 Dimitrios Piliouras

Distributed under the Eclipse Public License, the same as Clojure.
