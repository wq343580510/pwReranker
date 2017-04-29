package recursivenetwork.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/** This class contains sum of gradients along with the number of 
 * entries in the summation. This is useful during gradient descent since 
 * most updates use average gradient
 * 
 * @author Dipendra Misra (dkm@cs.cornell.edu)
 *  */
public class SumOfGradient {
	
	/** when count is 0, sumOfGradient should be the 0 tensor */
	private final INDArray sumOfGradient;
	private int count;
	
	public SumOfGradient(int dim) {
		this.sumOfGradient = Nd4j.zeros(new int[]{1, dim});
		this.count = 0;
	}
	
	public void addGradient(INDArray gradient) {
		this.sumOfGradient.addi(gradient);
		this.count++;
	}
	
	public int numTerms() {
		return this.count;
	}
	
	public INDArray getSumOfGradient() {
		return this.sumOfGradient;
	}
	
	public void flush() {
		this.sumOfGradient.muli(0);
		this.count = 0;
	}
	
	public INDArray getAverageGradient() {
		if(this.count == 0) {
			return this.sumOfGradient; //returns initial value
		} else {
			return this.sumOfGradient.div(this.count);
		}
	}

}
