package recursivenetwork.abstractnetwork;

import java.util.Iterator;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

import recursivenetwork.util.SumOfGradient;

/** A simple binary tree (atmost 2 children for every node)
 * @author Dipendra Misra (dkm@cs.cornell.edu) */
public class SimpleBinaryTree implements Tree {
	
	/** for leaves this contains reference to a global leaf vector */
	private INDArray vector;
	/** INDArray before non-linearity is applied */
	private INDArray preOutput; 
	/** gradient is non-null only for leaf nodes. Contains reference. Contains
	 * reference to the unique gradient vector corresponding to the leaf */
	private SumOfGradient gradient;  
	private final String label;
	private final List<Tree> children;
	private final int numChild;
	
	public SimpleBinaryTree(String label, List<Tree> children) {
		this.label = label;
		this.children = children;
		this.numChild = children.size();
	}
	
	@Override
	public int numChildren() {
		return this.numChild;
	}
	
	@Override
	public Tree getChild(int i) { 
		/* use iterator for trees with large degree */
		return this.children.get(i);
	}
	
	@Override
	public Iterator<Tree> getChildren() {
		return this.children.iterator();
	}
	
	@Override
	public INDArray getVector() {
		return this.vector;
	}
	
	@Override
	public void setVector(INDArray vector) {
		this.vector = vector;
	}
	
	@Override
	public SumOfGradient getGradient() {
		return this.gradient;
	}
	
	@Override
	public void setGradient(SumOfGradient sumOfGradient) {
		this.gradient = sumOfGradient;
	}
	
	/** accumulate gradient so that after a set of backprop through several trees
	 * you can update all the leaf vectors by the sum of gradients. Don't forget to 
	 * clear the gradients after updating. */
	@Override
	public void addGradient(INDArray gradient) {
		this.gradient.addGradient(gradient);		
	}
	
	@Override
	public String getLabel() {
		return this.label;
	}
	
	@Override
	public INDArray getPreOutput() {
		return this.preOutput;
	}
	
	@Override
	public void setPreOutput(INDArray preOutput) {
		this.preOutput = preOutput;
	}

	@Override
	public String getStr() {
		return null;
	}

	@Override
	public String getWord(){
		return null;
	}
}
