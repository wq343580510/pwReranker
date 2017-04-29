package reranker;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.learning.AdaGrad;

/**
 *
 * Created by zhouh on 15-12-20.
 *
 */
public class test {

    public void testTanhDerivative() {

        //Derivative of sigmoid: ds(x)/dx = s(x)*(1-s(x))
        //s(x) = 1 / (exp(-x) + 1)
        INDArray z = Nd4j.zeros(100);
        double[] expOut = new double[100];
        for (int i = 0; i < 100; i++) {
            double x = 0.1 * (i - 50);
            z.putScalar(i, x);
            double tanh = FastMath.tanh(x);
            expOut[i] = 1.0 - tanh * tanh;
        }

        INDArray zPrime = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", z).derivative());

        for (int i = 0; i < 100; i++) {
            double relError = Math.abs(expOut[i] - zPrime.getDouble(i)) / (Math.abs(expOut[i]) + Math.abs(zPrime.getDouble(i)));
            assert (relError < 1e-3);
        }
    }

    public static void testAdagrad(String[] args) {
        int rows = 2;
        int cols = 2;


        AdaGrad grad = new AdaGrad(rows, cols, 0.1);
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1, 1);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            String learningRates = String.valueOf("\nAdagrad\n " + grad.getGradient(W, 0)).replaceAll(";", "\n");
            System.err.println(W);
            System.out.println(learningRates);
            W.addi(W);
        }
    }

    public static void main(String[] args) {


        INDArray arr1 = Nd4j.create(new double[][]{{1, 2, 14, 5}, {3,6, 8,9}});
        System.err.println("shape: "+arr1.shape()[0] + " "+arr1.shape()[1]);
        INDArray max1 = Nd4j.max(arr1, 1);
        System.err.println("shape: "+max1.shape()[0] + " "+max1.shape()[1]);
        INDArray argmax1 = Nd4j.argMax(arr1, 1);
        System.err.println("argmax1: "+argmax1.shape()[0] + " "+argmax1.shape()[1]);
        INDArray arr2 = Nd4j.create(new double[][]{{1}, {2}, {14}, {5}});
        System.err.println("shape: "+arr2.shape()[0] + " "+arr2.shape()[1]);
        INDArray max2 = Nd4j.max(arr2, 1);
        System.err.println("shape: "+max2.shape()[0] + " "+max2.shape()[1]);
        INDArray arr3 = Nd4j.create(new double[][]{{1, 2, 14, 5}});
        INDArray max3 = Nd4j.max(arr3, 1);



    }
}
