package recursivenetwork.abstractnetwork;

import org.nd4j.linalg.api.ndarray.INDArray;

/** Represents a general class of recursive networks that perform composition on every node.
 * @author Dipendra Misra (dkm@cs.cornell.edu) 
 * 
 * TODO: 1. feed-forward and backprop through children in parallel 
 * 	     2. optimize the code 
 *       3. extend recursive neural network for arbitrary number of children 
 *       4. add an end-to-end working example 
 *       5. generalize to Tree with arbitrary number of children
 *       6. add Recursive Neural Tensor Network
 */
public abstract class RecursiveTreeNetwork 
					  implements AbstractRecursiveTreeNetwork, AbstractEmbedding {
	
	private final Double learningRate;
	private final boolean gradientClipping;
	private final double gradientThreshold;
	
	public RecursiveTreeNetwork(Double learningRate, 
								boolean gradientClipping, double gradientThreshold ) {
		this.learningRate = learningRate;
		this.gradientThreshold = gradientThreshold;
		this.gradientClipping = gradientClipping;
	}
	
	
	/** Computes the preOutput (before application of non-linearlity). */
	public abstract INDArray preOutput(INDArray left, INDArray right);
	
	/** Apply non-linearity */
	public abstract INDArray applyNonLinearity(INDArray preOutput);
	
	/** update parameters of this node given an error. This is done as follows
	 * del loss / del y del+ y/del theta  where y is the output of this node. 
	 * where + indicates that while changing theta, the children vectors are kept
	 * constant. (notation from Pascanu et. al. ``On difficulty of training recurrent network`` */
	public abstract void gradientUpdate(INDArray error, Tree t);
	
	/** If y is a node of a tree taking two children left, right then 
	 * this function returns del y/ del left. It can be computed from 
	 * compose function using an auto-differentiator.*/
	public abstract INDArray leftDerivative(INDArray y, INDArray left);
	
	/** If y is a node of a tree taking two children left, right then 
	 * this function returns del y/ del right. It can be computed from 
	 * compose function using an auto-differentiator.*/
	public abstract INDArray rightDerivative(INDArray y, INDArray left);

	@Override
	public INDArray feedForward(Tree t) {
		
		if(t.numChildren() == 0) { //leaf vector, leaf vectors are already initialized
			//t.getVector must be a row vector
			return t.getVector();
		}
		else if(t.numChildren() == 2) { //binary

			//do these recursive calls in parallel in future
			INDArray left  = this.feedForward(t.getChild(0)); 
			INDArray right = this.feedForward(t.getChild(1));
			
			//perform user defined composition
			INDArray preOutput = this.preOutput(left, right);
			t.setPreOutput(preOutput);
			INDArray nonLinear = this.applyNonLinearity(preOutput);
			t.setVector(nonLinear);
			
			return nonLinear;
		}
		throw new IllegalStateException("Binarize the tree");
	}

	@Override
	public void backProp(Tree t, INDArray error) {
		//error term gives us del loss/ del y (y is the output of this node)
		
		if(t.numChildren() == 0) { 
			//fine tune leaf vector embeddings
			t.addGradient(error);
		}
		else if(t.numChildren() == 2) {
			
			/* add to the gradient the loss due to this node given by
			 * del+ loss /del theta = error * del+ y / del theta  */
			
			this.gradientUpdate(error, t);
						
			/* backprop through structure: (y is this node's activation)
			 * calculate del loss / del y * del y / del child-output, or equivalently
			 * error * del y / del child-output, for both branches. */
			
			Tree left = t.getChild(0);
			Tree right = t.getChild(1);
			
			INDArray y = t.getVector();
			INDArray leftVector = left.getVector();
			INDArray rightVector = right.getVector();
			
			//del loss / del leftvector
			INDArray dydleft = this.leftDerivative(y, leftVector);
			INDArray leftLoss = error.mmul(dydleft); 
			
			 //del loss / del rightvector
			INDArray dydright = this.rightDerivative(y, rightVector);
			INDArray rightLoss = error.mmul(dydright);
			
			//Gradient Clipping to prevent exploding gradients
			if(this.gradientClipping) {
			
				double leftNorm = leftLoss.norm2(1).getDouble(0);
				double rightNorm = rightLoss.norm2(1).getDouble(0);
				
				//Gradient clipping to prevent gradient explosion
				double threshold = this.gradientThreshold;
				if(leftNorm > threshold) {
					leftLoss.divi(leftNorm).muli(threshold);
				}
				
				if(rightNorm > threshold) {
					rightLoss.divi(rightNorm).muli(threshold);
				}
				
			}
			
			this.backProp(left, leftLoss);
			this.backProp(right, rightLoss);
		}
		else new IllegalStateException("Binarize the tree");
		
	}
	
	/** update parameters for this recurrent network (not word vectors) */
	public abstract void updateParameters();
	
	/** clears the gradients */
	public abstract void flushGradients();

	@Override
	public abstract int getDimension();

	@Override
	public Object getEmbedding(Object obj) {
		//check if obj can be casted as a tree
		return this.feedForward((Tree)obj);
	}
}
