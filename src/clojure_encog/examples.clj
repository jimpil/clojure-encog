(ns clojure-encog.examples
(:use [clojure-encog.nnets]
      [clojure-encog.training])
(:import (org.encog.ml.train.strategy RequiredImprovementStrategy)
         (org.encog.mathutil.randomize FanInRandomizer)))

;--------------------------------------*XOR*------------------------------------------------------------
(defn xor 
"The classic XOR example from the encog book/wiki. You can choose whether you want to train once or keep-trying until the target is met."
[^Boolean keep-trying?]
(let [xor-input [[0.0 0.0] [1.0 0.0] [0.0 0.1] [1.0 1.0]]
      xor-ideal [[0.0] [1.0] [1.0] [0.0]] 
      dataset   (make-data :basic-dataset xor-input xor-ideal)
      network   (make-network {:input   2
                               :output  1
                               :hidden [2]} ;a single hidden layer
                               (make-activationF :sigmoid) 
                               (make-pattern :feed-forward))
      trainer   ((make-trainer :resilient-prop) network dataset)]
     ;;make use of the boolean parameter
      (if-not keep-trying? 
              (train trainer 0.01 500 (RequiredImprovementStrategy. 5)) ;;train the network once
      (loop [t false counter 1 _ nil] 
      (if t (println "Nailed it the" (str counter "th") "time!")
      (recur  (train trainer 0.01 500 (RequiredImprovementStrategy. 5))  ;;train the network until it succeeds
              (inc counter) (. network reset)))) )     
(do (println "\nNeural Network Results:")
    (doseq [pair dataset] 
    (let [output (. network compute (. pair getInput ))] ;;test the network
          (println (.getData (.getInput pair) 0) "," (. (. pair getInput) getData  1) 
                                                 ", actual=" (. output getData  0) 
                                                 ", ideal=" (.getData (. pair getIdeal) 0)))))     

))
;----------------------------------------------------------------------------------------------------------
;----------------------------------*LUNAR LANDER*-----------------------------------------------------------
(defmacro pilot-score "The fitness function for the GA." 
[network] 
`(. (NeuralPilot. ~network false) scorePilot))

(defn evaluate [best-evolved] 
(println"\nHow the winning network landed:")
(let [evolved-pilot (NeuralPilot. best-evolved true)]
(println (. evolved-pilot scorePilot))
(.shutdown (org.encog.Encog/getInstance))))


(defn lunar-lander 
"The classic Lunar-Lander example that can be trained with a GA or simulated annealing."
[popu]
(let [network (make-network {:input   3
                             :output  1
                             :hidden [50]} ;a single hidden layer of 50 neurons
                             (make-activationF :tanh) 
                             (make-pattern :feed-forward))
      trainer   ((make-trainer :genetic) network (FanInRandomizer.) (pilot-score network) false  popu 0.1 0.25)
     ]
     (loop [epoch 1
            _     nil
            best  nil]
     (if (> epoch 200) best ;;return the best evolved network 
     (recur (inc epoch) (. trainer iteration) (. trainer getMethod) )))))
;---------------------------------------------------------------------------------------------------------------

(defn -main [] 
(evaluate (lunar-lander 800))
;(xor false)
)
