(ns clojure-encog.examples
(:use [clojure-encog.nnets]
      [clojure-encog.training])
(:import (org.encog.ml.train.strategy RequiredImprovementStrategy)
         (org.encog.neural.networks.training TrainingSetScore)
         (org.encog.mathutil.randomize FanInRandomizer)
         (org.encog.util EngineArray)
         (org.encog.neural.neat.training NEATTraining)
         (org.encog.neural.neat NEATPopulation NEATNetwork)
         (org.encog.util.simple EncogUtility)
         (java.text NumberFormat)))
;--------------------------------------*XOR*------------------------------------------------------------
(defn xor 
"The classic XOR example from the encog book/wiki."
[^Boolean train-to-error?]
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
      (if train-to-error? 
         (train trainer 0.01 []) ;train to max error regardless of iterations
         (train trainer 0.01 300 [] #_[(RequiredImprovementStrategy. 5)])) ;;train to max iterations and max error
      
      
    ;(loop [t false counter 0 _ nil] 
      ;(if t (println "Nailed it after" (str counter) "times!")
      ;(recur  (train trainer 0.001 #_300 [] #_[(RequiredImprovementStrategy. 5)])  ;;train the network until it succeeds
      ;        (inc counter) (. network reset))))     
(do (println "\nNeural Network Results:")
    (EncogUtility/evaluate network dataset))))
    
   #_(doseq [pair dataset] 
    (let [output (. network compute (. pair getInput ))] ;;test the network
          (println (.getData (.getInput pair) 0) "," (. (. pair getInput) getData  1) 
                                                 ", actual=" (. output getData  0) 
                                                 ", ideal=" (.getData (. pair getIdeal) 0))))     


;---------------------------------------------------------------------------------------------------------
;------------------------------------*XOR NEAT*-----------------------------------------------------------

(defn xor-neat
"The classic XOR example solved using NeuroEvolution of Augmenting Topologies (NEAT)."
[train-to-error?]
(let [xor-input [[0.0 0.0] [1.0 0.0] [0.0 0.1] [1.0 1.0]]
      xor-ideal [[0.0] [1.0] [1.0] [0.0]] 
      dataset    (make-data :basic-dataset xor-input xor-ideal)
      activation (make-activationF :step)
      population (NEATPopulation. 2 1 1000)
      trainer    (NEATTraining. (TrainingSetScore. dataset) population)]
                ;((make-trainer :neat) some-function-name true/false population)
                ;this is the alternative when you have an actual Clojure function that you wan to use
                ;as the fitness-function. Here the implementation is already provided in a Java class,
                ;that is why I'm not using my own function instead.
      (do (. activation setCenter 0.5)
          (. population setNeatActivationFunction  activation)) 
      (let [best ^NEATNetwork (if train-to-error? 
                                 (train trainer 0.01 [])       ;;error-tolerance = 1%
                                 (train trainer 0.01 200 []))] ;;iteration limit = 200
              (do (println "\nNeat Network results:")
                  (EncogUtility/evaluate best dataset))) ))
      

;----------------------------------------------------------------------------------------------------------
;----------------------------------*LUNAR LANDER*-----------------------------------------------------------
; this example requires that you have LanderSimulation.class NeuralPilot.class in your classpath.
; both of them are in the jar.

(defmacro pilot-score 
"The fitness function for the GA. You will usually pass your own to the GA. A macro that simply wraps a call to your real fitness-function is a good choice." 
[network] 
`(. (NeuralPilot. ~network false) scorePilot))

(defn evaluate [best-evolved] 
(println"\nHow the winning network landed:")
(let [evolved-pilot (NeuralPilot. best-evolved true)]
(println (. evolved-pilot scorePilot))))


(defn lunar-lander 
"The classic Lunar-Lander example that can be trained with a GA or simulated annealing."
[popu]
(let [network (make-network {:input   3
                             :output  1
                             :hidden [50]} ;a single hidden layer of 50 neurons
                             (make-activationF :tanh) 
                             (make-pattern :feed-forward))
      trainer   ((make-trainer :genetic) network (make-randomizer :nguyen-widrow) 
                                                 (pilot-score network) false  popu 0.1 0.25)
     ]    
     (loop [epoch 1
            _     nil
            best  nil]
     (if (> epoch 200)  (do (.shutdown (org.encog.Encog/getInstance)) best) ;;return the best evolved network 
     (recur (inc epoch) (. trainer iteration) (. trainer getMethod)))) ))
;---------------------------------------------------------------------------------------------------------------
;----------------------------PREDICT-SUNSPOT_SVM------------------------------------------------------------

(def sunspots 
           [0.0262,  0.0575,  0.0837,  0.1203,  0.1883,  0.3033,  
            0.1517,  0.1046,  0.0523,  0.0418,  0.0157,  0.0000,  
            0.0000,  0.0105,  0.0575,  0.1412,  0.2458,  0.3295,  
            0.3138,  0.2040,  0.1464,  0.1360,  0.1151,  0.0575,  
            0.1098,  0.2092,  0.4079,  0.6381,  0.5387,  0.3818,  
            0.2458,  0.1831,  0.0575,  0.0262,  0.0837,  0.1778,  
            0.3661,  0.4236,  0.5805,  0.5282,  0.3818,  0.2092,  
            0.1046,  0.0837,  0.0262,  0.0575,  0.1151,  0.2092,  
            0.3138,  0.4231,  0.4362,  0.2495,  0.2500,  0.1606,  
            0.0638,  0.0502,  0.0534,  0.1700,  0.2489,  0.2824,  
            0.3290,  0.4493,  0.3201,  0.2359,  0.1904,  0.1093,  
            0.0596,  0.1977,  0.3651,  0.5549,  0.5272,  0.4268,  
            0.3478,  0.1820,  0.1600,  0.0366,  0.1036,  0.4838,  
            0.8075,  0.6585,  0.4435,  0.3562,  0.2014,  0.1192,  
            0.0534,  0.1260,  0.4336,  0.6904,  0.6846,  0.6177,  
            0.4702,  0.3483,  0.3138,  0.2453,  0.2144,  0.1114,  
            0.0837,  0.0335,  0.0214,  0.0356,  0.0758,  0.1778,  
            0.2354,  0.2254,  0.2484,  0.2207,  0.1470,  0.0528,  
            0.0424,  0.0131,  0.0000,  0.0073,  0.0262,  0.0638,  
            0.0727,  0.1851,  0.2395,  0.2150,  0.1574,  0.1250,  
            0.0816,  0.0345,  0.0209,  0.0094,  0.0445,  0.0868,  
            0.1898,  0.2594,  0.3358,  0.3504,  0.3708,  0.2500,  
            0.1438,  0.0445,  0.0690,  0.2976,  0.6354,  0.7233,  
            0.5397,  0.4482,  0.3379,  0.1919,  0.1266,  0.0560,  
            0.0785,  0.2097,  0.3216,  0.5152,  0.6522,  0.5036,  
            0.3483,  0.3373,  0.2829,  0.2040,  0.1077,  0.0350,  
            0.0225,  0.1187,  0.2866,  0.4906,  0.5010,  0.4038,  
            0.3091,  0.2301,  0.2458,  0.1595,  0.0853,  0.0382,  
            0.1966,  0.3870,  0.7270,  0.5816,  0.5314,  0.3462,  
            0.2338,  0.0889,  0.0591,  0.0649,  0.0178,  0.0314,  
            0.1689,  0.2840,  0.3122,  0.3332,  0.3321,  0.2730,  
            0.1328,  0.0685,  0.0356,  0.0330,  0.0371,  0.1862,  
            0.3818,  0.4451,  0.4079,  0.3347,  0.2186,  0.1370,  
            0.1396,  0.0633,  0.0497,  0.0141,  0.0262,  0.1276,  
            0.2197,  0.3321,  0.2814,  0.3243,  0.2537,  0.2296,  
            0.0973,  0.0298,  0.0188,  0.0073,  0.0502,  0.2479,  
            0.2986,  0.5434,  0.4215,  0.3326,  0.1966,  0.1365,  
            0.0743,  0.0303,  0.0873,  0.2317,  0.3342,  0.3609,  
            0.4069,  0.3394,  0.1867,  0.1109,  0.0581,  0.0298,  
            0.0455,  0.1888,  0.4168,  0.5983,  0.5732,  0.4644,  
            0.3546,  0.2484,  0.1600,  0.0853,  0.0502,  0.1736,  
            0.4843,  0.7929,  0.7128,  0.7045,  0.4388,  0.3630,  
            0.1647,  0.0727,  0.0230,  0.1987,  0.7411,  0.9947,  
            0.9665,  0.8316,  0.5873,  0.2819,  0.1961,  0.1459,  
            0.0534,  0.0790,  0.2458,  0.4906,  0.5539,  0.5518,  
            0.5465,  0.3483,  0.3603,  0.1987,  0.1804,  0.0811,  
            0.0659,  0.1428,  0.4838,  0.8127]) 
                                  
            
(defn predict-sunspot [spots]
"The PredictSunSpots SVM example ported to Clojure. Not so trivial as the others because it involves temporal data."
(let [start-year  1700
      window-size 30 ;input layer count
      ;train-start window-size
      train-end 259
      ;evaluation-start 260
      evaluation-end (dec (count spots))
      max-error 0.0001
      normalizedSunspots (normalize spots 0.9 0.1)
      test-data          (EngineArray/arrayCopy normalizedSunspots)
      closedLoopSunspots (EngineArray/arrayCopy normalizedSunspots)
      train-set         ((make-data :temporal-window normalizedSunspots) 
                         window-size 1) 
      network ((make-network {:input   window-size
                              :output  1      ;;will be ignored anyway 
                              :hidden [0]}    ;;same here
                             (make-activationF :tanh) ;;same here
                             (make-pattern :svm)) nil nil) ;;passing nil so default values are given for svm/kernel type
      trainer                ((make-trainer :svm) network train-set) 
      nf                     (NumberFormat/getNumberInstance)]
(do (. trainer iteration) ;;SVM TRAINED AND READY FOR PREDICTIONS AFTER THIS LINE
    (. nf setMaximumFractionDigits 4)
    (. nf setMinimumFractionDigits 4)
    (println "Year" \tab "Actual" \tab "Predict" \tab "Closed Loop Predict")     
(loop [evaluation-start (inc train-end)]          
(if (== evaluation-start evaluation-end) 'DONE...
    (let [input (make-data :basic window-size)]
    (dotimes [i (. input size)] 
    (. input setData i 
            (aget normalizedSunspots (+ i (- evaluation-start window-size)))))
              (let [output (. network compute input)
                    prediction (. output getData 0)
                    _          (aset closedLoopSunspots evaluation-start prediction)]                   
                    (dotimes [y (. input size)]
                    (. input setData y 
                             (aget closedLoopSunspots  (+ y (- evaluation-start window-size)))))
                              (let [output2 (. network compute input)
                              closed-loop (. output2 getData 0)]
                              (println  (+ start-year evaluation-start)
                                        \tab (. nf format (aget normalizedSunspots evaluation-start))
                                        \tab (. nf format prediction)
                                        \tab (. nf format closed-loop)) #_(debug-repl)) )
(recur (inc evaluation-start))))))))

;---------------------------------------------------------------------------------------------------------------
;run the lunar lander example using main otherwise the repl will hang under leiningen. 
(defn -main [] 
;(evaluate (lunar-lander 800))
;(xor false)
;(xor true)
(xor-neat)
;(predict-sunspot sunspots)
)
