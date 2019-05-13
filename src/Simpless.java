import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Simpless {
	public static void main(String [] args) throws Exception {
	
		// load data
		DataSource source = new DataSource("./original-dataset/all.arff");
	//	ArffLoader carregar = new ArffLoader();
		
		Instances treinar = source.getDataSet(0);				// base para treinar
		treinar.setClassIndex(treinar.numAttributes() - 1);
		Instances testar = new Instances(treinar);				// base para testar
	
		// classifier
		NaiveBayesUpdateable nb = new NaiveBayesUpdateable();
		nb.buildClassifier(treinar);
	
		// train
	//	Instance atual;
	//	while ((atual = carregar.getNextInstance(treinar)) != null) {
	//		   nb.updateClassifier(atual);
	//	}
		
		// test
		Evaluation eval = new Evaluation(treinar);
		eval.evaluateModel(nb, testar);
		
	
		System.out.println("Acertos: " + eval.correct());
		System.out.println("Erros: " + eval.incorrect());
		
		System.out.println(eval.toSummaryString());
	}
}
