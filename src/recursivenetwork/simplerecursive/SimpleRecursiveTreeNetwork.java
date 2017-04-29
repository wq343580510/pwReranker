package recursivenetwork.simplerecursive;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.HardTanh;
import org.nd4j.linalg.api.ops.impl.transforms.HardTanhDerivative;
import org.nd4j.linalg.factory.Nd4j;

import recursivenetwork.abstractnetwork.RecursiveTreeNetwork;
import recursivenetwork.abstractnetwork.Tree;

/** A simple recursive tree network that represents the composition HardTanh(W[a;b]+bias)
 * on every node of a tree.  
 * @author Dipendra Misra (dkm@cs.cornell.edu) */
public class SimpleRecursiveTreeNetwork extends RecursiveTreeNetwork {
	
	private final INDArray W, b;
	private final int n;
	private final INDArray gradW, gradb;
	private final Double learningRate, l2;
	
	/** a value that is saved globally for optimization purpose. This is however a bad 
	 * programming choice and should be removed in later versions.*/
	private INDArray nonLinearDerivativeTranspose;
	
	/** for a binarized tree, W is in Rnx2n and b is in Rn*/
	public SimpleRecursiveTreeNetwork(int n, Double learningRate, Double l2) {
		super(learningRate, false, 5.0);
		this.n = n;
		//do smarter initialization of W, b
		this.W = Nd4j.zeros(n, 2*n);
		this.b = Nd4j.zeros(n, 1); 
		this.gradW = Nd4j.zeros(n, 2*n);
		this.gradb = Nd4j.zeros(n, 1);
		this.learningRate = learningRate;
		this.l2 = l2;
		this.nonLinearDerivativeTranspose = null;
	}
	
	@Override
	public INDArray preOutput(INDArray left, INDArray right) {
		//perform composition
		INDArray concat = Nd4j.concat(1, left, right).transpose();
		INDArray preOutput = this.W.mmul(concat).add(this.b).transpose();
		return preOutput;
	}
	
	@Override
	public INDArray applyNonLinearity(INDArray preOutput) {
		//Be careful that execAndReturn works on the same copy so duplicate the INDArray
		INDArray nonLinear = Nd4j.getExecutioner().execAndReturn(new HardTanh(preOutput.dup()));
		return nonLinear;
	}
	
	@Override
	public void gradientUpdate(INDArray error, Tree t) {
		
		/* add to the gradient the loss due to this node given by
		 * del+ loss /del theta = error * del+ y / del theta  */

		//nonLinear Derivative = [g'(Wx+b)]
		//Be careful that execAndReturn works on the same copy so duplicate the INDArray
		INDArray nonLinearDerivative = Nd4j.getExecutioner()
				.execAndReturn(new HardTanhDerivative(t.getPreOutput().dup()));
		INDArray nonLinearDerivativeTranspose = nonLinearDerivative.transpose();
		this.nonLinearDerivativeTranspose = nonLinearDerivativeTranspose;
		
		/* del loss / del W 
		 * complex tensor. Handle each row of W at a time. */
		
		final int[] shape = this.W.shape();
		assert shape.length == 2;
		
		Tree left = t.getChild(0);
		Tree right = t.getChild(1);
		
		INDArray leftVector = left.getVector();
		INDArray rightVector = right.getVector();
		INDArray x  = Nd4j.concat(1, leftVector, rightVector); //can be cached in future
		
		/* TODO check x for NaNa and infinity 
		 * Collect statistics on whether gradient is vanishing or exploding. */
		
		INDArray errorTimesNonLinearDerivative = error.mul(nonLinearDerivative).transpose();
		this.gradW.addi(errorTimesNonLinearDerivative.mmul(x));

		//del loss / del b = error * del+ y / del b
		this.gradb.addi(errorTimesNonLinearDerivative);
	}

	@Override
	public INDArray leftDerivative(INDArray y, INDArray left) {
		
		INDArray leftDerivative = Nd4j.zeros(this.n, this.n);
		for(int col = 0; col < this.n; col++) {
			leftDerivative.putColumn(col, this.nonLinearDerivativeTranspose.mul(this.W.getColumn(col)));
		}
		
		return leftDerivative;
	}

	@Override
	public INDArray rightDerivative(INDArray y, INDArray right) {
		
		INDArray rightDerivative = Nd4j.zeros(this.n, this.n);
		for(int col = this.n; col < 2*this.n; col++) {
			rightDerivative.putColumn(col - this.n, this.nonLinearDerivativeTranspose.mul(this.W.getColumn(col)));
		}
		
		return rightDerivative;
	}
	
	/** update parameters as: (param is W / b)
	 *  param(t+1) = param(t) - eta {gradParam(t) + l2*param(t)} 
	 */
	public void updateParameters() {
		this.W.subi(this.gradW.mul(this.learningRate).add(this.W.mul(this.l2)));
		this.b.subi(this.gradb.mul(this.learningRate).add(this.b.mul(this.l2)));
	}
	
	/** clears the gradients */
	public void flushGradients() {
		this.gradW.muli(0);
		this.gradb.muli(0);
	}

	@Override
	public int getDimension() {
		return this.n;
	}

}
