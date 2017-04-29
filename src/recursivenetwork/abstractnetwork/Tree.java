package recursivenetwork.abstractnetwork;

import java.util.Iterator;

import org.nd4j.linalg.api.ndarray.INDArray;

import recursivenetwork.util.SumOfGradient;

/** A tree structure useful with recursive network
 * @author Dipendra Misra (dkm@cs.cornell.edu) */
public interface Tree {

	/** Number of children for this node */
	public int numChildren();
	
	/** ith child of this node*/
	public Tree getChild(int i);
	
	/** iterator over children*/
	public Iterator<Tree> getChildren();
	
	/** returns encoding of the subtree rooted at this node under the current model*/
	public INDArray getVector();
	
	/** sets encoding of the subtree rooted at this node*/
	public void setVector(INDArray vector);
	
	/** gets gradient for backprop through this node*/
	public SumOfGradient getGradient();
	
	/** sets gradient for backprop through this node. Generally, null 
	 * for all internal nodes. */
	public void setGradient(SumOfGradient sumOfGradient);
	
	/** accumulate gradient so that after a set of backprop through several trees
	 * you can update all the leaf vectors by the sum of gradients. Don't forget to 
	 * clear the gradients after updating. */
	public void addGradient(INDArray gradient);
	
	/** label of this node*/
	public String getLabel();

	public String getWord();
	
	/** preOutput is the vector that after application of non-linearity gives the  
	 * encoding of the subtree rooted at this node. */
	public INDArray getPreOutput();
	
	/** sets preOutput for this node. (see getPreOutput for details) */
	public void setPreOutput(INDArray preOutput);

	public String getStr();



 }
