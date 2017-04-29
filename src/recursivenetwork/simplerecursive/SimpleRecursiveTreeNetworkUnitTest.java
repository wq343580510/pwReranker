package recursivenetwork.simplerecursive;

import java.util.LinkedList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import recursivenetwork.abstractnetwork.RecursiveTreeNetwork;
import recursivenetwork.abstractnetwork.SimpleBinaryTree;
import recursivenetwork.abstractnetwork.Tree;
import recursivenetwork.util.Pair;
import recursivenetwork.util.SumOfGradient;

/** Class for testing recursive neural network. Creates a synthetic tree and 
 * label and minimizes the squared loss of the synthetic tree. A correct implementation 
 * will show that loss decreases steadily with every iteration
 * 
 * @author Dipendra Misra (dkm@cs.cornell.edu)
 * */
public class SimpleRecursiveTreeNetworkUnitTest {

	private final RecursiveTreeNetwork net;
	private final int dim;
	private final List<Pair<INDArray, SumOfGradient>> vectorAndGradients;
	private final double learningRate;
	private final double l2;
	
	public SimpleRecursiveTreeNetworkUnitTest(int dim, double learningRate, double regularizer) {
		this.dim = dim;
		this.learningRate = learningRate;
		this.l2 = regularizer;
		this.net = new SimpleRecursiveTreeNetwork(dim, learningRate, regularizer);
		this.vectorAndGradients = new LinkedList<Pair<INDArray, SumOfGradient>>();
	}
	
	public void addVectorAndGradient(INDArray vector, SumOfGradient gradient) {
		this.vectorAndGradients.add(Pair.of(vector, gradient));
	}
	
	public void learn(Tree t, INDArray label, int numEpochs) {
		
		for(int i = 1; i <= numEpochs; i++) {
			
			INDArray predict = this.net.feedForward(t);
			
			/* compute the loss and gradient (using squared loss) */
			double loss = 0;
			for(int j = 0; j < this.dim; j++) {
				double diff = predict.getDouble(j) - label.getDouble(j);
				loss = loss + diff * diff;
			}
			System.out.println("Loss is "+loss);
			INDArray error = predict.dup().subi(label).mul(2);
			
			this.net.backProp(t, error);
			this.net.updateParameters();
			this.net.flushGradients();
			
			//update the leaf vectors
			for(Pair<INDArray, SumOfGradient> vG: this.vectorAndGradients) {
				INDArray vector = vG.first();
				SumOfGradient gradient = vG.second();
				int numTerms = gradient.numTerms();
				if(numTerms == 0) {
					continue;
				}
				
				INDArray realGradient = gradient.getSumOfGradient().div((double)numTerms);
				realGradient.addi(vector.mul(this.l2));
				vector.subi(realGradient.mul(this.learningRate));
				gradient.flush();
			}
		}
	}

	public static void main(String args[]) throws Exception {
		
		/* create certain leaf vectors and their gradient */
		int dim = 30;
		INDArray a = Nd4j.rand(new int[]{1, dim});
		INDArray c = Nd4j.rand(new int[]{1, dim});
		INDArray b = Nd4j.rand(new int[]{1, dim});
		
		SumOfGradient gradA = new SumOfGradient(dim); 
		SumOfGradient gradB = new SumOfGradient(dim);
		SumOfGradient gradC = new SumOfGradient(dim);
		
		/* create the tree: (label1 (a (label2 (b c))))*/
		Tree aLeaf  = new SimpleBinaryTree("a", new LinkedList<Tree>());
		aLeaf.setVector(a); aLeaf.setGradient(gradA);
		
		Tree bLeaf  = new SimpleBinaryTree("b", new LinkedList<Tree>());
		bLeaf.setVector(b); bLeaf.setGradient(gradB);
		
		Tree cLeaf  = new SimpleBinaryTree("c", new LinkedList<Tree>());
		cLeaf.setVector(c); cLeaf.setGradient(gradC);
		
		List<Tree> n2Children = new LinkedList<Tree>();
		n2Children.add(bLeaf); n2Children.add(cLeaf);
		Tree n2 = new SimpleBinaryTree("label2", n2Children);
		
		List<Tree> n1Children = new LinkedList<Tree>();
		n1Children.add(aLeaf); n1Children.add(n2);
		Tree root = new SimpleBinaryTree("label1", n1Children);
		
		INDArray label = Nd4j.rand(new int[]{1, dim});
		
		/* train a recursive neural network */
		double learningRate = 0.01;
		double l2 = 0.01;
		SimpleRecursiveTreeNetworkUnitTest test = 
							new SimpleRecursiveTreeNetworkUnitTest(dim, learningRate, l2);
		test.addVectorAndGradient(a, gradA);
		test.addVectorAndGradient(b, gradB);
		test.addVectorAndGradient(c, gradC);
		
		int numEpochs = 100;
		test.learn(root, label, numEpochs);
	}
}
