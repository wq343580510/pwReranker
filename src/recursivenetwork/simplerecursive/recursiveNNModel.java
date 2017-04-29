package recursivenetwork.simplerecursive;

import edu.stanford.nlp.io.IOUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution;
import org.nd4j.linalg.factory.Nd4j;
import recursivenetwork.abstractnetwork.Tree;
import reranker.Config;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by zhouh on 16-1-10.
 */
public class recursiveNNModel {

    public HashSet<String> vocabulary = null;
    private int dim;
    Distribution distribution;
    boolean bTrain = true;

    public void setVocabulary(HashSet<String> vocabulary) {
        this.vocabulary = vocabulary;
    }

    public String getVocabWord(String word) {
        String lowerCaseWord = word.toLowerCase();
        if (vocabulary.contains(lowerCaseWord))
            return lowerCaseWord;
        return Config.UNKNOWN;
    }

    public void readPreTrain(String preTrainFile, int embedSize) throws IOException {

        BufferedReader reader = IOUtils.readerFromString(preTrainFile);
        int vocabSize = vocabulary.size();
        int preTrainedWordInVocabSize = 0;

        for (String line : IOUtils.getLineIterable(reader, false)) {
            String[] tokens = line.split("\\s{1,}");
            String caseWord = tokens[0].trim();
            String word = caseWord.toLowerCase();
            if (vocabulary.contains(word)) {

                preTrainedWordInVocabSize++;

                double[] wordEmbs = new double[embedSize];
                for (int i = 0; i < wordEmbs.length; i++)
                    wordEmbs[i] = Double.valueOf(tokens[i + 1]);

                INDArray wordEmb = Nd4j.create(wordEmbs, new int[]{embedSize, 1});
                wordVectors.put(caseWord, wordEmb);
            }
        }

        System.err.println("#####################");
        System.err.println("Pre train Word Embedding Done!");
        System.err.println("Vocab Size : " + vocabSize + ", Shot PreTrain Size : " + preTrainedWordInVocabSize + " (" + new DecimalFormat("00.00").format(((double) preTrainedWordInVocabSize / vocabSize)) + ")");

    }

    public void setBeTrain(boolean bTrain) {
        this.bTrain = bTrain;
    }

    // word representations
    public Map<String, INDArray> wordVectors;

    public recursiveNNModel(int dim) {
        this.dim = dim;
        wordVectors = new HashMap<>();

        distribution = new UniformDistribution(-0.001, 0.001);
    }

    public recursiveNNModel() {
        distribution = new UniformDistribution(-0.001, 0.001);
    }


    public void insertUNKWord(String word, boolean bRandom) {
        INDArray retval = null;

        if (!wordVectors.containsKey(word)) {
            if (bRandom)
                retval = Nd4j.rand(new int[]{dim, 1}, distribution);
            else
                retval = Nd4j.zeros(dim, 1);
            wordVectors.put(word, retval);
        }
    }


    public INDArray getTransform(Tree tree){
        return null;

    }


    public INDArray getScoreLayer(Tree tree) {
        return null;

    }

    public INDArray getWordVector(Tree tree) {
        String word = getVocabWord(tree.getWord());
        if (wordVectors.containsKey(word))
            return wordVectors.get(word);
        else
            return null;
    }


}
