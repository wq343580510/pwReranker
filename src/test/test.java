package test;


import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.learning.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * Created by zhouh on 15-12-20.
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

//    public static void testLSTM(String[] args) throws IOException {
//
//        int numEpochs = 30;                            //Total number of training + sample generation epochs
//        String generationInitialization = null;        //Optional character initialization; a random character is used if null
//        // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
//        // Initialization characters must all be in SIterator.getMinimalCharacterSet() by default
//        Random rng = new Random(12345);
//
//        //Get a DataSetIterator that handles vectorization of text into something we can use to train
//        // our GravesLSTM network.
//        SIterator iter = new SIterator();
//        int nOut = 1;
//
//        //Set up network configuration:
//        int lstmLayerSize = 5;
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
//                .learningRate(0.1)
//                .rmsDecay(0.95)
//                .seed(12345)
//                .regularization(true)
//                .l2(0.001)
//                .list(2)
//                .layer(0, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
//                        .updater(Updater.RMSPROP)
//                        .activation("tanh").weightInit(WeightInit.DISTRIBUTION)
//                        .dist(new UniformDistribution(-0.08, 0.08)).build())
//                .layer(1, new DenseLayer.Builder()
//                        .nIn(lstmLayerSize)
//                        .nOut(1)
//                        .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(-0.08, 0.08))
//                        .build())
//                .pretrain(false).backprop(true)
//                .build();
//
//        MultiLayerNetwork net = new MultiLayerNetwork(conf);
//        net.init();
//        net.setListeners(new ScoreIterationListener(1));
//
//        //Print the  number of parameters in the network (and for each layer)
////        Layer[] layers = net.getLayers();
////        int totalNumParams = 0;
////        for( int i=0; i<layers.length; i++ ){
////            int nParams = layers[i].numParams();
////            System.out.println("Number of parameters in layer " + i + ": " + nParams);
////            totalNumParams += nParams;
////        }
////        System.out.println("Total number of network parameters: " + totalNumParams);
//
//        //Do training, and then generate and print samples from network
//        INDArray input = Nd4j.rand(new int[]{1, 5, 3});
//        INDArray labels = Nd4j.zeros(new int[]{1, 1, 3});
//        for (int i = 0; i < numEpochs; i++) {
////            net.fit(iter);
//            DataSet data = iter.next();
//            List<INDArray> indArrays = net.feedForward(input, true);
//            System.err.println(indArrays);
//            System.err.println(indArrays.size());
//            System.err.println(indArrays.get(1).shape().length);
//            System.err.println(indArrays.get(1).shape()[0]);
//            System.err.println(indArrays.get(1).shape()[1]);
//            System.err.println(indArrays.get(1).shape()[2]);
//            net.backpropGradient(labels);
//
//
//        }
//
//        System.out.println("\n\nExample complete");
//
//    }

    public static void main(String[] args) throws IOException {

        INDArray[] arrays = new INDArray[3];

        arrays[0] = Nd4j.rand(2, 1);

        INDArray sum = arrays[0].sum(0);

        System.err.println(arrays[0]);
        arrays[1] = Nd4j.rand(2, 1);
        System.err.println(arrays[1]);
        arrays[2] = Nd4j.rand(2, 1);
        System.err.println(arrays[2]);

        INDArray vArrays = Nd4j.hstack(arrays);

        System.err.println("vArrays.shape : " + vArrays.shape()[0] +" "+vArrays.shape()[1]);

        INDArray reshape = vArrays.reshape(1, 2, 3);
        System.err.println(reshape);
        System.err.println("rshape : " + reshape.shape()[0] +" "+reshape.shape()[1] +" " + reshape.shape()[2]);

        INDArray back = reshape.tensorAlongDimension(0, 1, 2);


        INDArray subresult = back.subColumnVector(Nd4j.ones(1, 2));

        System.err.println(subresult);

        INDArray reshape1 = back.transpose().reshape(1, 2, 3);

        System.err.println(reshape1);


    }
}
