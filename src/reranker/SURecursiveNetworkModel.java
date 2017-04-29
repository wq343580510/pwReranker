package reranker;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.util.TwoDimensionalMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdaGrad;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by zhouh on 15-12-15.
 *
 */
public class SURecursiveNetworkModel implements Serializable {

    private static final long serialVersionUID = 1;
    public HashSet<String> vocabulary = null;
    private int dim;
    Distribution distribution;
    boolean bTrain = true;

    /*
     * maybe we could change the model into a int array, for less searching of hash map
     */
//    public TwoDimensionalMap<String, String, Integer> binaryTransformMap;
//    public Map<String, Integer> unaryTransformMap;
//    INDArray[] binaryTransform;
//    INDArray[] unaryTyTransform;
//
//
//    // score matrices for each node type
//    public TwoDimensionalMap<String, String, Integer> binaryScoreLayerMap;
//    public Map<String, Integer> unaryScoreLayerMap;
//    INDArray[] binaryScoreLayer;
//    INDArray[] unaryScoreLayer;
//
//
//    // word representations
//    public Map<String, Integer> wordVectorMap;
//    INDArray[] wordVectors;


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

    // weight matrices from children to parent
    public TwoDimensionalMap<String, String, INDArray> binaryTransform;
    public Map<String, INDArray> unaryTransform;

    // score matrices for each node type
    public TwoDimensionalMap<String, String, INDArray> binaryScoreLayer;
    public Map<String, INDArray> unaryScoreLayer;
    public TwoDimensionalMap<String, String, INDArray> binaryScoreLayer2;
    public Map<String, INDArray> unaryScoreLayer2;

    public Map<String, INDArray> forgetGate;

    // word representations
    public Map<String, INDArray> wordVectors;

    public SURecursiveNetworkModel(int dim) {
        this.dim = dim;
        binaryTransform = new TwoDimensionalMap<>();
        unaryTransform = new HashMap<>();
        forgetGate = new HashMap<>();
        binaryScoreLayer = new TwoDimensionalMap<>();
        unaryScoreLayer = new HashMap<>();
        binaryScoreLayer2 = new TwoDimensionalMap<>();
        unaryScoreLayer2 = new HashMap<>();
        wordVectors = new HashMap<>();

        distribution = new UniformDistribution(-0.001, 0.001);
    }

    public SURecursiveNetworkModel() {
        distribution = new UniformDistribution(-0.001, 0.001);
    }

    public INDArray insertBinaryTransform(String label1, String label2, boolean bRandom) {
        INDArray retval = null;
        if (!binaryTransform.contains(label1, label2)) {
            if (bRandom)
                retval = Nd4j.rand(new int[]{dim, 2 * dim + 1}, distribution);
            else
                retval = Nd4j.zeros(dim, 2 * dim + 1);

            binaryTransform.put(label1, label2, retval);

        }
        return retval;
    }

    public INDArray insertUnaryTransform(String label, boolean bRandom) {
        INDArray retval = null;

        if (!unaryTransform.containsKey(label)) {
            if (bRandom)
                retval = Nd4j.rand(new int[]{dim, dim + 1}, distribution);
            else
                retval = Nd4j.zeros(dim, dim + 1);

            unaryTransform.put(label, retval); // add 1 dimension for bias
        }

        return retval;
    }

    public INDArray insertForgetGate(String label, boolean bRandom) {
        INDArray retval = null;

        if (!forgetGate.containsKey(label)) {
            if (bRandom)
                retval = Nd4j.rand(new int[]{dim, 2 * dim + 1}, distribution);
            else
                retval = Nd4j.zeros(dim, 2 * dim + 1);

            forgetGate.put(label, retval); // add 1 dimension for bias
        }

        return retval;
    }

    public INDArray insertBinaryScoreLayer(String label1, String label2, boolean bRandom) {
        INDArray retval = null;

        if (!binaryScoreLayer.contains(label1, label2)) {
            if (bRandom)
                retval = Nd4j.rand(new int[]{1, dim}, distribution);
            else
                retval = Nd4j.zeros(1, dim);
            binaryScoreLayer.put(label1, label2, retval);
        }

        return retval;

    }

    public INDArray insertUnaryScoreLayer(String label, boolean bRandom) {
        INDArray retval = null;

        if (!unaryScoreLayer.containsKey(label)) {
            if (bRandom)
                retval = Nd4j.rand(new int[]{1, dim}, distribution);
            else
                retval = Nd4j.zeros(1, dim);
            unaryScoreLayer.put(label, retval);
        }

        return retval;

    }

    public INDArray insertBinaryScoreLayer2(String label1, String label2, boolean bRandom) {
        INDArray retval = null;

        if (!binaryScoreLayer2.contains(label1, label2)) {
            if (bRandom)
                retval = Nd4j.rand(new int[]{1, dim}, distribution);
            else
                retval = Nd4j.zeros(1, dim);
            binaryScoreLayer2.put(label1, label2, retval);
        }

        return retval;

    }

    public INDArray insertUnaryScoreLayer2(String label, boolean bRandom) {
        INDArray retval = null;

        if (!unaryScoreLayer2.containsKey(label)) {
            if (bRandom)
                retval = Nd4j.rand(new int[]{1, dim}, distribution);
            else
                retval = Nd4j.zeros(1, dim);
            unaryScoreLayer2.put(label, retval);
        }

        return retval;

    }

    public INDArray insertWordVector(String word, boolean bRandom) {
        INDArray retval = null;

        word = getVocabWord(word);

        if (!wordVectors.containsKey(word)) {
            if (bRandom)
                retval = Nd4j.rand(new int[]{dim, 1}, distribution);
            else
                retval = Nd4j.zeros(dim, 1);
            wordVectors.put(word, retval);
        }

        return retval;

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

    public void insertModelsFromOneTree(CFGRerankingTree tree) {

        // insert the forget gate
        insertForgetGate(tree.getLabel(), true);

        if (tree.numChildren() == 0) {
            insertWordVector(tree.getWord(), true);
            return;
        } else if (tree.numChildren() == 1) {
            insertUnaryScoreLayer(tree.getChild(0).getLabel(), true);
            insertUnaryScoreLayer2(tree.getChild(0).getLabel(), true);
            insertUnaryTransform(tree.getChild(0).getLabel(), true);
            insertModelsFromOneTree(tree.getChild(0));
        } else {
            insertBinaryScoreLayer(tree.getChild(0).getLabel(), tree.getChild(1).getLabel(), true);
            insertBinaryScoreLayer2(tree.getChild(0).getLabel(), tree.getChild(1).getLabel(), true);
            insertBinaryTransform(tree.getChild(0).getLabel(), tree.getChild(1).getLabel(), true);
            insertModelsFromOneTree(tree.getChild(0));
            insertModelsFromOneTree(tree.getChild(1));
        }
    }

    public INDArray getBinaryTransform(CFGRerankingTree tree) {
        if (binaryTransform.contains(
                tree.getChild(0).getLabel(),
                tree.getChild(1).getLabel())) {

//            System.err.println("Found rule : "+ tree.getChild(0).getLabel() + " "+tree.getChild(1).getLabel());

            return binaryTransform.get(
                    tree.getChild(0).getLabel(),
                    tree.getChild(1).getLabel());
        } else {
            if (bTrain) {
                System.err.println(tree.getChild(0).getLabel() + " " + tree.getChild(1).getLabel() + " has not been found!");
                return null;

            } else {
                System.err.println("Found unk rule : " + tree.getChild(0).getLabel() + " " + tree.getChild(1).getLabel());
                return Nd4j.rand(new int[]{dim, 2 * dim + 1}, distribution);
            }

        }
    }

    public INDArray getUnaryTransform(CFGRerankingTree tree) {
        if (unaryTransform.containsKey(tree.getChild(0).getLabel()))
            return unaryTransform.get(tree.getChild(0).getLabel());
        else if (bTrain) {
            System.err.println(tree.getChild(0).getLabel() + " has not been found!");
            return null;
        } else
            return Nd4j.rand(new int[]{dim, dim + 1}, distribution);
    }

    public INDArray getForgetGate(CFGRerankingTree tree) {
        if (forgetGate.containsKey(tree.getLabel()))
            return forgetGate.get(tree.getLabel());
        else if (bTrain) {
            System.err.println(tree.getLabel() + " has not been found!");
            return null;
        } else
            return Nd4j.rand(new int[]{dim, dim + 1}, distribution);
    }

    public INDArray getBinaryScoreLayer(CFGRerankingTree tree) {
        if (binaryScoreLayer.contains(
                tree.getChild(0).getLabel(),
                tree.getChild(1).getLabel()))
            return binaryScoreLayer.get(
                    tree.getChild(0).getLabel(),
                    tree.getChild(1).getLabel());
        else if (bTrain) {
            System.err.println(tree.getChild(0).getLabel() + " " + tree.getChild(1).getLabel() + " has not been found!");
            return null;


        } else
            return Nd4j.rand(new int[]{1, dim}, distribution);
    }

    public INDArray getUnaryScoreLayer(CFGRerankingTree tree) {
        if (unaryScoreLayer.containsKey(tree.getChild(0).getLabel()))
            return unaryScoreLayer.get(tree.getChild(0).getLabel());
        else if (bTrain) {
            System.err.println(tree.getChild(0).getLabel() + " has not been found!");
            return null;

        } else
            return Nd4j.rand(new int[]{1, dim}, distribution);
    }

    public INDArray getBinaryScoreLayer2(CFGRerankingTree tree) {
        if (binaryScoreLayer2.contains(
                tree.getChild(0).getLabel(),
                tree.getChild(1).getLabel()))
            return binaryScoreLayer2.get(
                    tree.getChild(0).getLabel(),
                    tree.getChild(1).getLabel());
        else if (bTrain) {
            System.err.println(tree.getChild(0).getLabel() + " " + tree.getChild(1).getLabel() + " has not been found!");
            return null;


        } else
            return Nd4j.rand(new int[]{1, dim}, distribution);
    }

    public INDArray getUnaryScoreLayer2(CFGRerankingTree tree) {
        if (unaryScoreLayer2.containsKey(tree.getChild(0).getLabel()))
            return unaryScoreLayer2.get(tree.getChild(0).getLabel());
        else if (bTrain) {
            System.err.println(tree.getChild(0).getLabel() + " has not been found!");
            return null;

        } else
            return Nd4j.rand(new int[]{1, dim}, distribution);
    }

    public INDArray getWordVector(CFGRerankingTree tree) {
        String word = getVocabWord(tree.getWord());
        if (wordVectors.containsKey(word))
            return wordVectors.get(word);
        else
            return null;
    }


    public INDArray getOrInsertBinaryTransform(CFGRerankingTree tree) {
        if (binaryTransform.contains(
                tree.getChild(0).getLabel(),
                tree.getChild(1).getLabel()))
            return binaryTransform.get(
                    tree.getChild(0).getLabel(),
                    tree.getChild(1).getLabel());
        else {

            return insertBinaryTransform(tree.getChild(0).getLabel(), tree.getChild(1).getLabel(), false);
        }
    }

    public INDArray getOrInsertUnaryTransform(CFGRerankingTree CFGRerankingTree) {
        if (unaryTransform.containsKey(CFGRerankingTree.getChild(0).getLabel()))
            return unaryTransform.get(CFGRerankingTree.getChild(0).getLabel());
        else
            return insertUnaryTransform(CFGRerankingTree.getChild(0).getLabel(), false);
    }

    public INDArray getOrInsertForgetGate(CFGRerankingTree CFGRerankingTree) {
        if (forgetGate.containsKey(CFGRerankingTree.getLabel()))
            return forgetGate.get(CFGRerankingTree.getLabel());
        else
            return insertForgetGate(CFGRerankingTree.getLabel(), false);
    }

    public INDArray getOrInsertBinaryScoreLayer(CFGRerankingTree CFGRerankingTree) {

        if (binaryScoreLayer.contains(
                CFGRerankingTree.getChild(0).getLabel(),
                CFGRerankingTree.getChild(1).getLabel()))
            return binaryScoreLayer.get(
                    CFGRerankingTree.getChild(0).getLabel(),
                    CFGRerankingTree.getChild(1).getLabel());
        else
            return insertBinaryScoreLayer(CFGRerankingTree.getChild(0).getLabel(), CFGRerankingTree.getChild(1).getLabel(), false);
    }

    public INDArray getOrInsertUnaryScoreLayer(CFGRerankingTree CFGRerankingTree) {
        if (unaryScoreLayer.containsKey(CFGRerankingTree.getChild(0).getLabel()))
            return unaryScoreLayer.get(CFGRerankingTree.getChild(0).getLabel());
        else
            return insertUnaryScoreLayer(CFGRerankingTree.getChild(0).getLabel(), false);
    }

    public INDArray getOrInsertBinaryScoreLayer2(CFGRerankingTree CFGRerankingTree) {

        if (binaryScoreLayer2.contains(
                CFGRerankingTree.getChild(0).getLabel(),
                CFGRerankingTree.getChild(1).getLabel()))
            return binaryScoreLayer2.get(
                    CFGRerankingTree.getChild(0).getLabel(),
                    CFGRerankingTree.getChild(1).getLabel());
        else
            return insertBinaryScoreLayer2(CFGRerankingTree.getChild(0).getLabel(), CFGRerankingTree.getChild(1).getLabel(), false);
    }

    public INDArray getOrInsertUnaryScoreLayer2(CFGRerankingTree CFGRerankingTree) {
        if (unaryScoreLayer2.containsKey(CFGRerankingTree.getChild(0).getLabel()))
            return unaryScoreLayer2.get(CFGRerankingTree.getChild(0).getLabel());
        else
            return insertUnaryScoreLayer2(CFGRerankingTree.getChild(0).getLabel(), false);
    }


    public INDArray getOrInsertWordVector(CFGRerankingTree CFGRerankingTree) {
        String word = getVocabWord(CFGRerankingTree.getWord());
        if (wordVectors.containsKey(word))
            return wordVectors.get(word);
        else
            return insertWordVector(word, false);
    }


    /**
     * TODO complete the read and write model module
     *
     * @param modelFileName
     */
    public static SURecursiveNetworkModel readModel(String modelFileName) {

        SURecursiveNetworkModel model = new SURecursiveNetworkModel();

        ObjectInputStream ois1 = null;
        try {
            ois1 = new ObjectInputStream(new FileInputStream(modelFileName));

            System.err.print("Begin to read vocabulary ");
            model.vocabulary = (HashSet<String>) ois1.readObject();
            System.err.print("binaryTransform ");
            model.binaryTransform = (TwoDimensionalMap<String, String, INDArray>) ois1.readObject();
            System.err.print("unaryTransform ");
            model.unaryTransform = (Map<String, INDArray>) ois1.readObject();
            System.err.print("forget gate ");
            model.forgetGate = (Map<String, INDArray>) ois1.readObject();
            System.err.print("binaryScoreLayer ");
            model.binaryScoreLayer = (TwoDimensionalMap<String, String, INDArray>) ois1.readObject();
            System.err.print("unaryScoreLayer ");
            model.unaryScoreLayer = (Map<String, INDArray>) ois1.readObject();
            System.err.print("binaryScoreLayer ");
            model.binaryScoreLayer2 = (TwoDimensionalMap<String, String, INDArray>) ois1.readObject();
            System.err.print("unaryScoreLayer ");
            model.unaryScoreLayer2 = (Map<String, INDArray>) ois1.readObject();
            System.err.print("wordVectors ");
            model.wordVectors = (Map<String, INDArray>) ois1.readObject();
            System.err.print("unknowWord and dim.");
            model.dim = (int) ois1.readObject();

            ois1.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        return model;
    }

    public static void writeModel(String modelFileName, SURecursiveNetworkModel model) {

        System.err.println("Begin to Write Model" + modelFileName);
        try {
            ObjectOutputStream oos1 = new ObjectOutputStream(new FileOutputStream(modelFileName));
            System.err.print("Begin to write vocabulary ");
            oos1.writeObject(model.vocabulary);
            System.err.print("binaryTransform ");
            oos1.writeObject(model.binaryTransform);
            System.err.print("unaryTransform ");
            oos1.writeObject(model.unaryTransform);
            System.err.print("forget gate ");
            oos1.writeObject(model.forgetGate);
            System.err.print("binaryScoreLayer ");
            oos1.writeObject(model.binaryScoreLayer);
            System.err.print("unaryScoreLayer ");
            oos1.writeObject(model.unaryScoreLayer);
            System.err.print("binaryScoreLayer ");
            oos1.writeObject(model.binaryScoreLayer2);
            System.err.print("unaryScoreLayer ");
            oos1.writeObject(model.unaryScoreLayer2);
            System.err.print("wordVectors ");
            oos1.writeObject(model.wordVectors);
            System.err.print("unknowWord and dim.");
            oos1.writeObject(model.dim);

            oos1.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.err.println("Finish to Write Model");
    }

    /**
     * New a new model style object for store gradients of matrices in model
     *
     * @return
     */
    public SURecursiveNetworkModel createGradients() {

        SURecursiveNetworkModel gradient = new SURecursiveNetworkModel(dim);
        gradient.setVocabulary(vocabulary);

        // insert the binary transform gradients
        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> btIterator = binaryTransform.iterator();
        while (btIterator.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = btIterator.next();
            gradient.insertBinaryTransform(it.getFirstKey(), it.getSecondKey(), false);
        }

        // insert the binary transform score layer
        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIterator = binaryScoreLayer.iterator();
        while (bslIterator.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIterator.next();
            gradient.insertBinaryScoreLayer(it.getFirstKey(), it.getSecondKey(), false);
        }

        // insert the binary transform score layer
        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIterator2 = binaryScoreLayer2.iterator();
        while (bslIterator2.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIterator2.next();
            gradient.insertBinaryScoreLayer2(it.getFirstKey(), it.getSecondKey(), false);
        }

        // insert the forget gate gradients
        for (Map.Entry<String, INDArray> fgEntry : forgetGate.entrySet())
            gradient.insertForgetGate(fgEntry.getKey(), false);

        // insert the unary transform gradients
        for (Map.Entry<String, INDArray> utEntry : unaryTransform.entrySet())
            gradient.insertUnaryTransform(utEntry.getKey(), false);


        // insert the unary transform score layer gradients
        for (Map.Entry<String, INDArray> uslEntry : unaryScoreLayer.entrySet())
            gradient.insertUnaryScoreLayer(uslEntry.getKey(), false);

        // insert the unary transform score layer gradients
        for (Map.Entry<String, INDArray> uslEntry : unaryScoreLayer2.entrySet())
            gradient.insertUnaryScoreLayer2(uslEntry.getKey(), false);

        // insert the word vector gradients
        for (Map.Entry<String, INDArray> wvEntry : wordVectors.entrySet())
            gradient.insertWordVector(wvEntry.getKey(), false);


        return gradient;
    }


    public IdentityHashMap<INDArray, AdaGrad> createAdaGradSquare(double fBPRate) {

        IdentityHashMap<INDArray, AdaGrad> paras2SquareGradients = new IdentityHashMap<>();

        // insert the binary transform gradients
        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> btIterator = binaryTransform.iterator();
        while (btIterator.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = btIterator.next();
            paras2SquareGradients.put(it.getValue(), new AdaGrad(it.getValue().shape(), fBPRate));
        }

        // insert the binary transform score layer
        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIterator = binaryScoreLayer.iterator();
        while (bslIterator.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIterator.next();
            paras2SquareGradients.put(it.getValue(), new AdaGrad(it.getValue().shape(), fBPRate));
        }
        // insert the binary transform score layer
        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIterator2 = binaryScoreLayer2.iterator();
        while (bslIterator2.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIterator2.next();
            paras2SquareGradients.put(it.getValue(), new AdaGrad(it.getValue().shape(), fBPRate));
        }

        // insert the unary forget gate
        for (Map.Entry<String, INDArray> fgEntry : forgetGate.entrySet())
            paras2SquareGradients.put(fgEntry.getValue(), new AdaGrad(fgEntry.getValue().shape(), fBPRate));

        // insert the unary transform gradients
        for (Map.Entry<String, INDArray> utEntry : unaryTransform.entrySet())
            paras2SquareGradients.put(utEntry.getValue(), new AdaGrad(utEntry.getValue().shape(), fBPRate));

        // insert the unary transform score layer gradients
        for (Map.Entry<String, INDArray> uslEntry : unaryScoreLayer.entrySet())
            paras2SquareGradients.put(uslEntry.getValue(), new AdaGrad(uslEntry.getValue().shape(), fBPRate));
        // insert the unary transform score layer gradients
        for (Map.Entry<String, INDArray> uslEntry : unaryScoreLayer2.entrySet())
            paras2SquareGradients.put(uslEntry.getValue(), new AdaGrad(uslEntry.getValue().shape(), fBPRate));

        // insert the word vector gradients
        for (Map.Entry<String, INDArray> wvEntry : wordVectors.entrySet())
            paras2SquareGradients.put(wvEntry.getValue(), new AdaGrad(wvEntry.getValue().shape(), fBPRate));

        return paras2SquareGradients;
    }

    /**
     * update the gradients from one batch examples to the network paras
     * Using Adagrad Updating and add the l2-norm
     *
     * @param gradients
     * @param gradientSquareMap
     * @param batchSize
     * @param regRate
     */
    public void updateModel(SURecursiveNetworkModel gradients, IdentityHashMap<INDArray, AdaGrad> gradientSquareMap,
                            int batchSize, double regRate) {

        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> btIterator = gradients.binaryTransform.iterator();
        while (btIterator.hasNext()) {

            TwoDimensionalMap.Entry<String, String, INDArray> it = btIterator.next();
            INDArray toBeUpdated = binaryTransform.get(it.getFirstKey(), it.getSecondKey());
            INDArray gradient = it.getValue();
            gradient.muli(1.0 / batchSize);
            gradient.addi(toBeUpdated.mul(regRate)); // add l-2 norm to gradients

            INDArray learningRates = gradientSquareMap.get(toBeUpdated).getGradient(gradient, 0);
//            gradient.muli( learningRates );
            toBeUpdated.subi(learningRates);
//            toBeUpdated.subi(gradient.muli(0.1));
            gradient.muli(0);
        }

        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIterator = gradients.binaryScoreLayer.iterator();
        while (bslIterator.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIterator.next();

            INDArray toBeUpdated = binaryScoreLayer.get(it.getFirstKey(), it.getSecondKey());
            INDArray gradient = it.getValue();
            gradient.muli(1.0 / batchSize);
            gradient.addi(toBeUpdated.mul(regRate)); // add l-2 norm to gradients

            INDArray learningRates = gradientSquareMap.get(toBeUpdated).getGradient(gradient, 0);
//            gradient.muli( learningRates );

            toBeUpdated.subi(learningRates);
//            toBeUpdated.subi(gradient.muli(0.1));
            gradient.muli(0);
        }

        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIterator2 = gradients.binaryScoreLayer2.iterator();
        while (bslIterator2.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIterator2.next();

            INDArray toBeUpdated = binaryScoreLayer2.get(it.getFirstKey(), it.getSecondKey());
            INDArray gradient = it.getValue();
            gradient.muli(1.0 / batchSize);
            gradient.addi(toBeUpdated.mul(regRate)); // add l-2 norm to gradients

            INDArray learningRates = gradientSquareMap.get(toBeUpdated).getGradient(gradient, 0);
//            gradient.muli( learningRates );

            toBeUpdated.subi(learningRates);
//            toBeUpdated.subi(gradient.muli(0.1));
            gradient.muli(0);
        }

        // update forget gate weight
        for (Map.Entry<String, INDArray> fgEntry : gradients.forgetGate.entrySet()) {


            INDArray toBeUpdated = forgetGate.get(fgEntry.getKey());
            INDArray gradient = fgEntry.getValue();
            gradient.muli(1.0 / batchSize);
            gradient.addi(toBeUpdated.mul(regRate)); // add l-2 norm to gradients

            INDArray learningRates = gradientSquareMap.get(toBeUpdated).getGradient(gradient, 0);

//            gradient.muli( learningRates );
            toBeUpdated.subi(learningRates);
//            toBeUpdated.subi(gradient.muli(0.1));
            gradient.muli(0);
        }

        // insert the unary transform gradients
        for (Map.Entry<String, INDArray> utEntry : gradients.unaryTransform.entrySet()) {

            INDArray toBeUpdated = unaryTransform.get(utEntry.getKey());
            INDArray gradient = utEntry.getValue();
            gradient.muli(1.0 / batchSize);
            gradient.addi(toBeUpdated.mul(regRate)); // add l-2 norm to gradients

            INDArray learningRates = gradientSquareMap.get(toBeUpdated).getGradient(gradient, 0);

//            gradient.muli( learningRates );
            toBeUpdated.subi(learningRates);
//            toBeUpdated.subi(gradient.muli(0.1));
            gradient.muli(0);
        }

        // insert the unary transform score layer gradients
        for (Map.Entry<String, INDArray> uslEntry : gradients.unaryScoreLayer.entrySet()) {

            INDArray toBeUpdated = unaryScoreLayer.get(uslEntry.getKey());
            INDArray gradient = uslEntry.getValue();
            gradient.muli(1.0 / batchSize);
            gradient.addi(toBeUpdated.mul(regRate)); // add l-2 norm to gradients

            INDArray learningRates = gradientSquareMap.get(toBeUpdated).getGradient(gradient, 0);
//            gradient.muli( learningRates );
            toBeUpdated.subi(learningRates);
//            toBeUpdated.subi(gradient.muli(0.1));
            gradient.muli(0);
        }
// insert the unary transform score layer gradients
        for (Map.Entry<String, INDArray> uslEntry : gradients.unaryScoreLayer2.entrySet()) {

            INDArray toBeUpdated = unaryScoreLayer2.get(uslEntry.getKey());
            INDArray gradient = uslEntry.getValue();
            gradient.muli(1.0 / batchSize);
            gradient.addi(toBeUpdated.mul(regRate)); // add l-2 norm to gradients

            INDArray learningRates = gradientSquareMap.get(toBeUpdated).getGradient(gradient, 0);
//            gradient.muli( learningRates );
            toBeUpdated.subi(learningRates);
//            toBeUpdated.subi(gradient.muli(0.1));
            gradient.muli(0);
        }

        // insert the word vector gradients
        for (Map.Entry<String, INDArray> wvEntry : gradients.wordVectors.entrySet()) {
            INDArray toBeUpdated = wordVectors.get(wvEntry.getKey());
            INDArray gradient = wvEntry.getValue();
            gradient.muli(1.0 / batchSize);
            gradient.addi(toBeUpdated.mul(regRate)); // add l-2 norm to gradients

            INDArray learningRates = gradientSquareMap.get(toBeUpdated).getGradient(gradient, 0);
//            gradient.muli( learningRates );
            toBeUpdated.subi(learningRates);
//            toBeUpdated.subi(gradient.muli(0.1));
            gradient.muli(0);
        }
    }

    /**
     * Merge the gradients from one sentence to the final gradients for updating paras
     *
     * @param gradients
     * @param rate
     */
    public void merge(SURecursiveNetworkModel gradients, double rate) {

        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> btIterator = gradients.binaryTransform.iterator();
        while (btIterator.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = btIterator.next();
            binaryTransform.get(it.getFirstKey(), it.getSecondKey()).addi(it.getValue().muli(rate));
        }

        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIterator = gradients.binaryScoreLayer.iterator();
        while (bslIterator.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIterator.next();
            binaryScoreLayer.get(it.getFirstKey(), it.getSecondKey()).addi(it.getValue().muli(rate));
        }

        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIterator2 = gradients.binaryScoreLayer2.iterator();
        while (bslIterator2.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIterator2.next();
            binaryScoreLayer2.get(it.getFirstKey(), it.getSecondKey()).addi(it.getValue().muli(rate));
        }

        // insert the forget gate gradients
        for (Map.Entry<String, INDArray> fgEntry : gradients.forgetGate.entrySet()) {

            forgetGate.get(fgEntry.getKey()).addi(fgEntry.getValue().muli(rate));
        }

        // insert the unary transform gradients
        for (Map.Entry<String, INDArray> utEntry : gradients.unaryTransform.entrySet()) {

            unaryTransform.get(utEntry.getKey()).addi(utEntry.getValue().muli(rate));
        }

        // insert the unary transform score layer gradients
        for (Map.Entry<String, INDArray> uslEntry : gradients.unaryScoreLayer.entrySet()) {

            unaryScoreLayer.get(uslEntry.getKey()).addi(uslEntry.getValue().muli(rate));
        }
        // insert the unary transform score layer gradients
        for (Map.Entry<String, INDArray> uslEntry : gradients.unaryScoreLayer2.entrySet()) {

            unaryScoreLayer2.get(uslEntry.getKey()).addi(uslEntry.getValue().muli(rate));
        }

        // insert the word vector gradients
        for (Map.Entry<String, INDArray> wvEntry : gradients.wordVectors.entrySet()) {
            wordVectors.get(wvEntry.getKey()).addi(wvEntry.getValue().muli(rate));
        }
    }
}