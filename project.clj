(defproject clojure-encog "0.3.0-SNAPSHOT"
  :description "Native Clojure wrapper for Encog(v3) Machine-Learning framework."
  :url "http://github.com/jimpil/clojure-encog"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.4.0"]
                 [org.encog/encog-core "3.1.0"]
                 ]
  :jvm-opts ["-Xmx1g"] 
  ;:javac-options {:classpath "target/dependency/encog-core-3.1.0.jar" :destdir "target/classes"}
  ;:java-source-path "src/java"
  ;:main       clojure-encog.examples
  )
