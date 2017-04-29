package test;//package test;
//
//import org.deeplearning4j.datasets.iterator.DataSetIterator;
//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.dataset.DataSet;
//import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
//import org.nd4j.linalg.factory.Nd4j;
//
//import java.io.IOException;
//import java.util.*;
//
//
///** A very simple DataSetIterator for use in the GravesLSTMCharModellingExample.
// * Given a text file and a few options, generate feature vectors and labels for training,
// * where we want to predict the next character in the sequence.<br>
// * This is done by randomly choosing a position in the text file to start the sequence and
// * (optionally) scanning backwards to a new line (to ensure we don't start half way through a word
// * for example).<br>
// * Feature vectors and labels are both one-hot vectors of same length
// * @author Alex Black
// */
//public class SIterator implements DataSetIterator {
//    private static final long serialVersionUID = -7287833919126626356L;
//    private static final int MAX_SCAN_LENGTH = 200;
//    private Random rng;
//    int examplesSoFar = 0;
//    int numExamplesToFetch = 1;
//
//    public SIterator() throws IOException {
//    }
//
//
//
//    public boolean hasNext() {
//        return examplesSoFar + 1 <= numExamplesToFetch;
//    }
//
//    public DataSet next() {
//        return next(1);
//    }
//
//    public DataSet next(int num) {
//        //Allocate space:
//        INDArray input = Nd4j.rand(new int[]{num,5,3});
//        INDArray labels = Nd4j.zeros(new int[]{num,1,3});
//
//
//        examplesSoFar += num;
//        return new DataSet(input,labels);
//    }
//
//    public int totalExamples() {
//        return numExamplesToFetch;
//    }
//
//    public int inputColumns() {
//        return 1;
//    }
//
//    public int totalOutcomes() {
//        return 1;
//    }
//
//    public void reset() {
//        examplesSoFar = 0;
//    }
//
//    public int batch() {
//        return 1;
//    }
//
//    public int cursor() {
//        return examplesSoFar;
//    }
//
//    public int numExamples() {
//        return numExamplesToFetch;
//    }
//
//    public void setPreProcessor(DataSetPreProcessor preProcessor) {
//        throw new UnsupportedOperationException("Not implemented");
//    }
//
//    @Override
//    public List<String> getLabels() {
//        return null;
//    }
//
//    @Override
//    public void remove() {
//        throw new UnsupportedOperationException();
//    }
//
//}
