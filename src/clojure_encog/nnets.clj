(ns clojure-encog.nnets

(:import 
(org.encog.neural.pattern 
       FeedForwardPattern ADALINEPattern ART1Pattern BAMPattern
       BoltzmannPattern CPNPattern ElmanPattern HopfieldPattern PatternError
       JordanPattern SOMPattern PNNPattern SVMPattern RadialBasisPattern)
       
(org.encog.engine.network.activation 
       ActivationTANH ActivationSigmoid ActivationGaussian ActivationBiPolar 
       ActivationLinear  ActivationLOG ActivationRamp ActivationSoftMax ActivationStep
       ActivationSIN ActivationBipolarSteepenedSigmoid ActivationClippedLinear
       ActivationCompetitive ActivationElliott ActivationElliottSymmetric ActivationSteepenedSigmoid)     
    
   ))


 (defn make-pattern  
 "Constructs a neural base pattern from which we can generate 
 a concrete network effortlessly (see make-network for details).
 Options include:
 -------------------------------------------------------------
 :feed-forward  :adaline  :art1  :bam  :boltzman      
 :jordan        :elman    :svm   :rbf  :hopfield    
 :som           :pnn      :cpn                  
 -------------------------------------------------------------
 Returns an object that implements NeuralNetworkPattern."
  [model]
  (condp = model
       :feed-forward (FeedForwardPattern.)
       :adaline      (ADALINEPattern.)
       :art1         (ART1Pattern.)
       :bam          (BAMPattern.)
       :boltzman     (BoltzmannPattern.)
       :cpn          (CPNPattern.)
       :elman        (ElmanPattern.)
       :hopfield     (HopfieldPattern.)
       :jordan       (JordanPattern.)
       :som          (SOMPattern.)
       :pnn          (PNNPattern.)
       :svm          (SVMPattern.)
       :rbf          (RadialBasisPattern.)
 :else (throw (IllegalArgumentException. "Unsupported neural-pattern!"))) )

;;usage: 
;;(make-pattern :feed-forward)
;;(make-pattern :svm)

 (defn make-activationF  
 "Constructs an activation-function to be used by the layers.
  Expects a keyword based argument. Options include:
  --------------------------------------------------
  :tanh     :sigmoid   :gaussian    :bipolar  
  :linear   :log       :ramp        :sin
  :elliot   :soft-max  :competitive :bipolar-steepend
  :elliot-symmetric :clipped-linear :steepened-sigmoid
  ---------------------------------------------------
  Returns an ActivationFunction object." ;TODO
 [fun]
 (condp = fun 
      :tanh     (ActivationTANH.)
      :sigmoid  (ActivationSigmoid.)
      :gaussian (ActivationGaussian.)
      :bipolar  (ActivationBiPolar.)
      :linear   (ActivationLinear.) 
      :log      (ActivationLOG.)
      :ramp     (ActivationRamp.)
      :sin      (ActivationSIN.)
      :soft-max (ActivationSoftMax.)
      :step     (ActivationStep.)
      :bipolar-steepend (ActivationBipolarSteepenedSigmoid.)
      :clipped-linear   (ActivationClippedLinear.)
      :competitive      (ActivationCompetitive.)
      :elliot           (ActivationElliott.)
      :elliot-symmetric (ActivationElliottSymmetric.)
      :steepened-sigmoid (ActivationSteepenedSigmoid.)
 :else (throw (IllegalArgumentException. "Unsupported activation-function!"))))

;;usage: (make-activationF :tanh)
;;       (make-activationF :linear)

(defmulti make-network 
"Constructs a neural-network given some layers, an activation and a neural pattern. layers has to be map with 3 keys {:input x :output y :hidden [k j & more]} where :hidden holds a vector whose size is the number of desirable hidden layers. The layers are added sequentially to the input layer so first hidden layer will have k neurons, the second j neurons and so forth. Some networks do not accept hidden layers and thus the parameter is ignored. See example usage below.
 Returns the complete neural-network object with randomized weights or in case of SVMs and PNNs a function, that needs to be called again (potentially with nil arguments) in order to produce the actual SVM/PNN object.
 example: 
  (make-network {:input 32 
                 :output 1 
                 :hidden [40, 10, 5]}   ;;3 hidden layers (first has 40 neurons, second 10, third 5)
  (make-activationF :tanh)              ;;hyperbolic tangent for activation function
  (make-pattern     :feed-forward))     ;;the classic feed-forward pattern"
(fn [layers activation pattern] (class pattern)));;dispatch function only cares about the class of the pattern

(defmethod make-network FeedForwardPattern
[layers activation pattern] 
    (do 
       (.setInputNeurons pattern  (layers :input))
       (.setOutputNeurons pattern (layers :output))
       (.setActivationFunction pattern activation);many activations are allowed here
 (dotimes [i (count (layers :hidden))] 
         (.addHiddenLayer pattern ((layers :hidden) i)))
         (.generate pattern)))  ;;finally, return the complete network object
 
(defmethod make-network ADALINEPattern ;no hidden layers - only ActivationLinear
[layers _ pattern] ;;ignoring activation 
    (doto pattern  
         (.setInputNeurons  (layers :input))   
         (.setOutputNeurons (layers :output)))
    (.generate pattern)) 


(defmethod make-network ART1Pattern ;;no hidden layers - only ActivationLinear
[layers _ pattern] ;;ignoring activation 
    (doto pattern  
         (.setInputNeurons  (layers :input))
         (.setOutputNeurons (layers :output)))
    (.generate pattern))

(defmethod make-network BAMPattern ;no hidden layers - only ActivationBiPolar
[layers _ pattern] ;;ignoring layers and activation 
    (doto pattern  
         (.setF1Neurons (layers :input))
         (.setF2Neurons (layers :output)))
    (.generate pattern))
 
(defmethod make-network BoltzmannPattern ;;no hidden layers - only ActivationBipolar
[layers _ pattern] ;;ignoring activation 
    (do  
    (.setInputNeurons pattern  (layers :input))
    (.generate pattern)))
 
(defmethod make-network CPNPattern ;;one hidden layer - only ActivationBipolar + Competitive
[layers _ pattern] ;;ignoring activation 
    (doto pattern  
         (.setInputNeurons  (layers :input))
         (.setInstarCount   ((layers :hidden) 0))
         (.setOutputNeurons (layers :output)))
    (.generate pattern))
 
(defmethod make-network ElmanPattern;;one hidden layer only - settable activation
[layers activation pattern] 
    (doto pattern  
         (.setInputNeurons  (layers :input))
         (.setOutputNeurons (layers :output))
         (.addHiddenLayer ((layers :hidden) 0))
         (.setActivationFunction activation)) 
    (.generate pattern))


(defmethod make-network HopfieldPattern ;;one hidden layer only - settable activation
[layers activation pattern] 
    (doto pattern 
         (.setInputNeurons  (layers :input))
         (.setOutputNeurons (layers :output))
         (.addHiddenLayer   ((layers :hidden) 0))
         (.setActivationFunction activation))
    (.generate pattern))   
 
 
(defmethod make-network JordanPattern ;;one hidden layer only - settable activation
[layers activation pattern] 
    (doto pattern 
         (.setInputNeurons (layers :input))
         (.setOutputNeurons (layers :output))
         (.addHiddenLayer  ((layers :hidden) 0))
         (.setActivationFunction pattern activation))
   (.generate pattern))


(defmethod make-network SOMPattern ;non settable activation - no hidden layers
[layers _ pattern] ;;ignoring activation
    (doto pattern  
         (.setInputNeurons  (layers :input))
         (.setOutputNeurons (layers :output)))
    (.generate pattern))
 
(defmethod make-network RadialBasisPattern ;usually has one hidden layer - non settable activation
[layers _ pattern] ;;ignoring activation
    (doto pattern  
         (.setInputNeurons  (layers :input))
         (.setOutputNeurons (layers :output))
         (.addHiddenLayer ((layers :hidden) 0)))
    (.generate pattern))
    
    
;(defmethod make-network RSOMPattern ;;no hidden layers - non settable activation
;[layers _ p] ;;ignoring activation
; (let [pattern p]
;    (do  (.setInputNeurons pattern  (layers :input))
;         (.setOutputNeurons pattern (layers :output)) 
;    (. pattern generate))))  
;----------------------------------------------------------------------------------
;----------------------------------------------------------------------------------
;;The next 2 patterns are slightly different than the rest. When called they will return a function (not a pattern object). 
;;This function needs to be called again (with nil arguments for defaults) in order to get the actual pattern object.
    
(defmethod make-network SVMPattern ;;no hidden layers - non settable activation
[layers _ pattern] ;;ignoring activation
 (fn [svm-type kernel-type] ;;returns a function which will return the actual network
    (doto pattern 
         (.setInputNeurons (layers :input))
         (.setOutputNeurons 1) ;only one output is allowed
         (when-not (nil? kernel-type) 
                         (.setKernelType kernel-type))
         (when-not (nil? svm-type) 
                         (.setSVMType svm-type))) 
    (.generate pattern)))
    
(defmethod make-network PNNPattern ;;no hidden layers - only LinearActivation 
[layers _ p] ;;ignoring activation
(fn [kernel out-model]
 (let [pattern p]
    (doto pattern 
         (.setInputNeurons  (layers :input))
         (.setOutputNeurons (layers :output))
         (when-not (nil? kernel)    
                         (.setKernel kernel))
         (when-not (nil? out-model) 
                         (.setOutmodel out-model))) 
    (.generate pattern))))   
;-----------------------------------------------------------------------------------    

(defmethod make-network :default 
[_ _ _] ;;ignoring everything
(throw (IllegalArgumentException. "Unsupported Pattern!")))
