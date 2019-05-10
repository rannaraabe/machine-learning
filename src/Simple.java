import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.BufferedWriter;
import java.io.FileWriter;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesUpdateable;

public class Simple {
	public static void main(String[] args) throws Exception {

		// load data
		ArffLoader load = new ArffLoader();
		DataSource source = new DataSource("original-dataset/all.arff");

		Instances train = source.getDataSet();
		train.setClassIndex(train.numAttributes() - 1);
		Instances test = new Instances(train);
		
		// classifier
		NaiveBayesUpdateable c = new NaiveBayesUpdateable();
		c.buildClassifier(train);

		// train
		Instance newI;
		
		int hits = 0, errors = 0;
		double value = 0;
				
		for (int i = 0; i < train.numInstances(); i++) {
			value = c.classifyInstance(train.instance(i)); // índice para os valores de sequência no atributo

			c.distributionForInstance(train.instance(i));
			test.instance(i).setClassValue(value);
			
			while ((newI = load.getNextInstance(train)) != null) {
				c.updateClassifier(newI);
			}
			
			// System.out.println("classify: " + valor + " " + data.classAttribute().value((int) valor));
		}

		// test
		Evaluation e = new Evaluation(train);
		e.evaluateModel(c, test);
		
		for (int i = 0; i < train.numInstances(); i++) {
			if (train.instance(i).classAttribute().value((int) value) == test.instance(i).classAttribute().value((int) value)) {
				hits++;
			} else {
				errors++;
			}
		}

		System.out.println("Hits: " + hits);
		System.out.println("Errors: " + errors);

		//accuracy
		double acc = (train.numInstances()*100)/hits;
		System.out.println("Accuracy: " + acc + "%");

		// saving after being sorted
		BufferedWriter writer = new BufferedWriter(new FileWriter("results/all.arff"));
		writer.write(test.toString());
		writer.newLine();
		writer.flush();
		writer.close();

		/* =============================================================================================================================== */
		System.out.println("");

		// load data
		ArffLoader carregar = new ArffLoader();
		
		Instances treinar = source.getDataSet(0);				// base para treinar
		treinar.setClassIndex(treinar.numAttributes() - 1);
		Instances testar = new Instances(treinar);				// base para testar

		// classifier
		NaiveBayesUpdateable nb = new NaiveBayesUpdateable();
		nb.buildClassifier(treinar);

		// train
		Instance atual;
		while ((atual = carregar.getNextInstance(treinar)) != null) {
			   nb.updateClassifier(atual);
		}
		
		// test
		Evaluation eval = new Evaluation(treinar);
		eval.evaluateModel(nb, testar);

		System.out.println("Acertos: " + eval.correct());
		System.out.println("Erros: " + eval.incorrect());
		
		System.out.println(eval.toSummaryString());

	}
}
