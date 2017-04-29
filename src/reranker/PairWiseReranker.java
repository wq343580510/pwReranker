package reranker;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Timing;
import edu.stanford.nlp.util.Triple;
import edu.stanford.nlp.util.concurrent.MulticoreWrapper;
import edu.stanford.nlp.util.concurrent.ThreadsafeProcessor;
import nncon.CFGTree;
import nncon.CFGTreeNode;
import nndep.Util;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdaGrad;
import recursivenetwork.simplerecursive.SURecursiveNetwork;

import java.io.*;
import java.util.*;

import static java.lang.System.exit;
import static nncon.CFGTree.getTokens;

/**
 * Created by zhouh on 16-2-17.
 *
 */
public class PairWiseReranker {
    SURecursiveNetwork pairWiseGatedRNN;

    Config config;

    Evaluator evalb;

    IdentityHashMap<INDArray, AdaGrad> pwgRNNGradientSquareMap; // map from neural network paras to the gradient square history

    public PairWiseReranker(Properties properties) {

        config = new Config(properties);
        evalb = new Evaluator(config.tlp);
    }

    public DataSet genDataOnline(BufferedReader goldReader,
                                 BufferedReader kbestReader, int readExampleNum) throws IOException {

        List<CFGRerankingTree> goldTree;
        List<List<CFGRerankingTree>> kbestTrees;
        List<List<Double>> kbestTreesBaseModelScore;
        List<List<Double>> kbestTreesF1Score;

        goldTree = new ArrayList<>();
        kbestTrees = new ArrayList<>();
        kbestTreesBaseModelScore = new ArrayList<>();
        kbestTreesF1Score = new ArrayList<>();

        DataSet retval = new DataSet(goldTree, kbestTrees, kbestTreesBaseModelScore, kbestTreesF1Score);


        System.err.println("Begin to generate gold trees online!");


        //import the gold trees
        int goldTreeReadNum = 0;
        while (true) {

            String line = goldReader.readLine();

            if (line == null)
                break;

            if ((line.trim().length() == 0)) // skip the empty line
                continue;

            CFGTree tree = new CFGTree();
            tree.CTBReadNote(getTokens(line));
            CFGRerankingTree goldCFGRerankingTree = CFGRerankingTree.CFGTree2RerankingTree(tree);
            goldTree.add(goldCFGRerankingTree);

            //if the read gold trees get the read limitation
            goldTreeReadNum++;
            if (goldTreeReadNum >= readExampleNum)
                break;
        }
        System.err.println("Generate gold trees done, totally read " + goldTreeReadNum + " trees!");

        // inport the k-best trees
        int kbestTreesReadNum = 0;
        String wsjStr = null;
        System.err.println("Begin to generate k-best trees!");
        ArrayList<CFGRerankingTree> treesOfOneSent = new ArrayList<>();
        ArrayList<Double> baseTreeScoresOfOneSent = new ArrayList<>();
        while (true) {

            String line = kbestReader.readLine();

            if (line == null)
                break;

            if (line.contains("WSJ") && !(line.trim().charAt(0) == '(')) {

                wsjStr = line;
                continue;
            }

            if (line.trim().length() == 0) { // if empty line?

                if (treesOfOneSent.size() == 0) {
                    System.err.println(wsjStr);

                    exit(0);
                }
                kbestTrees.add(treesOfOneSent);
                kbestTreesBaseModelScore.add(baseTreeScoresOfOneSent);
                treesOfOneSent = new ArrayList<>();
                baseTreeScoresOfOneSent = new ArrayList<>();


                // if the read gold trees get the read limitation
                kbestTreesReadNum++;
                if (kbestTreesReadNum >= readExampleNum)
                    break;

                if ((kbestTrees.size() % 500) == 0)
                    System.err.println(kbestTrees.size());

                continue;
            }

            if (!(line.trim().charAt(0) == '(')) {
                baseTreeScoresOfOneSent.add(Double.valueOf(line.trim()));
                continue;
            } else {

                /*
                 * get the scores and trees for kbest
                 */
                CFGTree tree = new CFGTree();
                tree.CTBReadNote(getTokens(line.trim()));
                CFGRerankingTree cfgRerankingTree = CFGRerankingTree.CFGTree2RerankingTree(tree);
                treesOfOneSent.add(cfgRerankingTree);

                continue;

            }

        }

        int size = goldTree.size();
        // TODO add the kbestTreesF1Score size check after we import the kbestTreesF1Score data
        //      in the next step.
        if (kbestTrees.size() != size || kbestTreesBaseModelScore.size() != size)
            throw new RuntimeException("gold size : " + size + "kbest size : " + kbestTrees.size() + " Size of data set do not match!");

        retval.size = size;

        if (config.bUseRankingRNN)
            generateF1Score(retval);

        return retval;
    }

    /**
     * compute the F1 score for the k-best reranking candidates
     *
     * @param data the reranking data set
     */
    public void generateF1Score(DataSet data) {

        List<CFGRerankingTree> goldTree = data.goldTree;
        List<List<CFGRerankingTree>> kbestTrees = data.kbestTrees;
        List<List<Double>> kbestF1 = new ArrayList<>();
        List<CFGRerankingTree> bestF1Trees = new ArrayList<>();

        for (int i = 0; i < goldTree.size(); i++) {

            double bestF1Score = Double.NEGATIVE_INFINITY;
            CFGRerankingTree bestF1Tree = null;

            String goldStr = goldTree.get(i).getStr();
            List<CFGRerankingTree> kbest = kbestTrees.get(i);
            List<Double> f1ScoreOfOneKBestTrees = new ArrayList<>();

            for (int j = 0; j < kbest.size(); j++) {

                try {
                    double f1 = evalb.evaluateSent(kbest.get(j).getStr(), goldStr);
                    f1ScoreOfOneKBestTrees.add(f1);

                    if (f1 > bestF1Score) {
                        bestF1Score = f1;
                        bestF1Tree = kbest.get(j);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }

            }
            kbestF1.add(f1ScoreOfOneKBestTrees);
            bestF1Trees.add(bestF1Tree);

        }

        data.setBestF1ScoreTrees(bestF1Trees);
        data.setKbestTreesF1Score(kbestF1);
    }

    /**
     * generate the reranking data set for the testing data in all batch.
     *
     * @param goldTreesFile
     * @param kbestTreesFile
     * @param bStoreCFGTree
     * @return
     */
    public DataSet genTestData(String goldTreesFile, String kbestTreesFile, boolean bStoreCFGTree) {

        List<CFGRerankingTree> goldTree;
        List<List<CFGRerankingTree>> kbestTrees;
        List<List<Double>> kbestTreesBaseModelScore;
        List<List<Double>> kbestTreesF1Score;
        List<List<Double>> kbestTreesCharniarkScore;

        goldTree = new ArrayList<>();
        kbestTrees = new ArrayList<>();
        kbestTreesBaseModelScore = new ArrayList<>();
        kbestTreesF1Score = new ArrayList<>();
        kbestTreesCharniarkScore = new ArrayList<>();

        DataSet retval = new DataSet(goldTree, kbestTrees, kbestTreesBaseModelScore, kbestTreesF1Score);

        BufferedReader goldReader = null;
        BufferedReader kbestReader = null;
        try {
            goldReader = IOUtils.readerFromString(goldTreesFile);
            kbestReader = IOUtils.readerFromString(kbestTreesFile);

            System.err.println("Begin to generate gold trees");
            /*
             * import the gold trees
             */
            for (String line : IOUtils.getLineIterable(goldReader, false)) {

                if ((line.trim().length() == 0)) // skip the empty line
                    continue;

//                String[] splits = line.trim().split("\\s{1,}");
                CFGTree tree = new CFGTree();
                tree.CTBReadNote(getTokens(line));
                CFGRerankingTree goldCFGRerankingTree = CFGRerankingTree.CFGTree2RerankingTree(tree);
                goldTree.add(goldCFGRerankingTree);

                if (bStoreCFGTree)
                    retval.goldCFGTrees.add(tree);

            }
            System.err.println("Generate gold trees done!");

            /*
             * inport the k-best trees
             */
            System.err.println("Begin to generate k-best trees!");
            ArrayList<CFGRerankingTree> treesOfOneSent = new ArrayList<>();
            ArrayList<Double> baseTreeScoresOfOneSent = new ArrayList<>();
            ArrayList<Double> treeCharnairkScoresOfOneSent = new ArrayList<>();
            for (String line : IOUtils.getLineIterable(kbestReader, false)) {

                if (line.contains("WSJ") && !(line.trim().charAt(0) == '('))
                    continue;

                if (line.trim().length() == 0) { // if empty line?
                    kbestTrees.add(treesOfOneSent);
                    kbestTreesBaseModelScore.add(baseTreeScoresOfOneSent);
                    kbestTreesCharniarkScore.add(treeCharnairkScoresOfOneSent);
                    treesOfOneSent = new ArrayList<>();
                    baseTreeScoresOfOneSent = new ArrayList<>();
                    if ((kbestTrees.size() % 500) == 0)
                        System.err.println(kbestTrees.size());
                    continue;
                }

                if (!(line.trim().charAt(0) == '(')) {

                    if (config.bUseCharniarkScoreTest) {

                        String[] tokens = line.trim().split(" ");
                        if (tokens.length != 2) {
                            throw new RuntimeException("Wrong line!");
                        }

                        treeCharnairkScoresOfOneSent.add(Double.valueOf(tokens[0]));
                        baseTreeScoresOfOneSent.add(Double.valueOf(tokens[1]));
                    } else {

                        baseTreeScoresOfOneSent.add(Double.valueOf(line));
                    }


                    continue;
                } else {

                /*
                 * get the scores and trees for kbest
                 */
                    CFGTree tree = new CFGTree();
                    tree.CTBReadNote(getTokens(line.trim()));
                    CFGRerankingTree cfgRerankingTree = CFGRerankingTree.CFGTree2RerankingTree(tree);
                    treesOfOneSent.add(cfgRerankingTree);
                    if (bStoreCFGTree)
                        retval.tree2CFGTreeIdentityHashMap.put(cfgRerankingTree, tree);
                    continue;

                }

            }
        } catch (IOException e) {
            throw new RuntimeIOException(e);
        } finally {
            IOUtils.closeIgnoringExceptions(goldReader);
            IOUtils.closeIgnoringExceptions(kbestReader);
        }

         /*
         * check the size of tree examples
         */
        int size = goldTree.size();
        // TODO add the kbestTreesF1Score size check after we import the kbestTreesF1Score data
        //      in the next step.
        if (kbestTrees.size() != size || kbestTreesBaseModelScore.size() != size)
            throw new RuntimeException("gold size : " + size + "kbest size : " + kbestTrees.size() + " Size of data set do not match!");

        retval.size = size;

        if (config.bUseRankingRNN)
            generateF1Score(retval);

        retval.setKbestTreesCharniarkScore(kbestTreesCharniarkScore);

        return retval;
    }


    /**
     * Explicitly specifies the number of arguments expected with
     * particular command line options.
     */
    private static final Map<String, Integer> numArgs = new HashMap<>();

    static {
        numArgs.put("textFile", 1);
        numArgs.put("outFile", 1);
    }

    public static void main(String[] args) throws IOException {

        Properties props = StringUtils.argsToProperties(args, numArgs);


        PairWiseReranker reranker = new PairWiseReranker(props);

        if (props.containsKey("evalb")) {

            reranker.evalb.evaluateFile(props.getProperty("p"), props.getProperty("h"));
            return;
        }


        // Train
        if (props.containsKey("trainFile")) {
            reranker.train(
                    props.getProperty("trainFile") + ".kbest",
                    props.getProperty("trainFile") + ".gold",
                    props.getProperty("devFile") + ".kbest",
                    props.getProperty("devFile") + ".gold",
                    props.getProperty("model"),
                    props.getProperty("embedFile"));
        }

        // Test
        if (props.containsKey("testFile")) {

            /*
             * new the neural net
             */
            SURecursiveNetworkModel pwgModel = SURecursiveNetworkModel.readModel("PWGRNN." + props.getProperty("model"));
            pwgModel.setBeTrain(false);
            reranker.pairWiseGatedRNN = new SURecursiveNetwork(reranker.config.hiddenSize, pwgModel);

            reranker.test(props.getProperty("testFile") + ".kbest",
                    props.getProperty("testFile") + ".gold");
        }
    }


    public Triple<CFGRerankingTree, Pair<CFGRerankingTree, CFGRerankingTree>, Double>
    getBiDirectionPWGHighestScoreTree(Example example, boolean bTrain) {

        CFGRerankingTree right2leftBestTree = null;
        CFGRerankingTree left2rightBestTree = null;
        int right2leftBestTreeIndex = -1;
        int left2rightBestTreeIndex = -1;

        CFGRerankingTree bestTree = null;

        CFGRerankingTree bestF1ScoreTree = example.bestF1ScoreTree;


        //right to left!
        for (int i = 0; i < example.kbestTrees.size(); i++) {
            CFGRerankingTree tree = example.kbestTrees.get(i);

            // if is the first tree, just assign
            if (right2leftBestTree == null) {
                right2leftBestTree = tree;
                right2leftBestTreeIndex = example.kbestTrees.size() - 1;
                continue;
            }

            Pair<Double, Double> pwgRerankingScores = pairWiseGatedRNN.pairwiseGetScore(right2leftBestTree, tree);
            double bestScore = pwgRerankingScores.first;
            double current_i_score = pwgRerankingScores.second;


            bestScore += example.kbestTreesBaseModelScore.get(right2leftBestTreeIndex);
            current_i_score += example.kbestTreesBaseModelScore.get(i);

            if (bestScore < current_i_score) {

                if(bTrain){
                    if(right2leftBestTree == bestF1ScoreTree){
                        return new Triple<>(null, new Pair<>(right2leftBestTree, tree), (-bestScore + current_i_score));
                    }
                }

                right2leftBestTreeIndex = i;
                right2leftBestTree = tree;

            }
            else{

                if(bTrain){
                    if(tree == bestF1ScoreTree){
                        return new Triple<>(null, new Pair<>(tree, right2leftBestTree), (bestScore - current_i_score));
                    }
                }
            }
        }

        // left to right
        for (int i = 0; i < example.kbestTrees.size(); i++) {
            CFGRerankingTree tree = example.kbestTrees.get(i);

            // if is the first tree, just assign
            if (left2rightBestTree == null) {
                left2rightBestTree = tree;
                left2rightBestTreeIndex = example.kbestTrees.size() - 1;
                continue;
            }

            Pair<Double, Double> pwgRerankingScores = pairWiseGatedRNN.pairwiseGetScore(left2rightBestTree, tree);
            double bestScore = pwgRerankingScores.first;
            double current_i_score = pwgRerankingScores.second;


            bestScore += example.kbestTreesBaseModelScore.get(left2rightBestTreeIndex);
            current_i_score += example.kbestTreesBaseModelScore.get(i);

            if (bestScore < current_i_score) {
                left2rightBestTreeIndex = i;
                left2rightBestTree = tree;
            }
        }

        if(left2rightBestTree == right2leftBestTree)
            bestTree = left2rightBestTree;
        else{
            Pair<Double, Double> pwgRerankingScores = pairWiseGatedRNN.pairwiseGetScore(left2rightBestTree, right2leftBestTree);

            bestTree = pwgRerankingScores.first > pwgRerankingScores.second ? left2rightBestTree : right2leftBestTree;

            if(bTrain){
                if (pwgRerankingScores.first > pwgRerankingScores.second) {
                    return new Triple<>(null, new Pair<>(right2leftBestTree, left2rightBestTree), (pwgRerankingScores.first - pwgRerankingScores.second));
                }

            }

        }



        return new Triple<>(bestTree, null, 0.0);
    }


    public void train(String trainFileKBest, String trainFileGold, String devFileKBest, String devFileGold,
                      String modelFile, String embedFile) throws IOException {

        // output the paras info
        System.err.println("Train File: " + trainFileKBest + " " + trainFileGold);
        System.err.println("Dev File: " + devFileKBest + " " + devFileGold);
        System.err.println("Model File: " + modelFile);
        System.err.println("Embedding File: " + embedFile);


        // generate the training and dev data
        System.out.println("begin to gen data!");
        DataSet devData = genTestData(devFileGold, devFileKBest, true);
        System.out.println("finish to gen data!");

        // add the word embedding and matrix into the model
        SURecursiveNetworkModel pwgRNNModel = generateSURNNModelOnline(trainFileGold, trainFileKBest);

        // prepare for pre-train
        if (embedFile != null) {
            pwgRNNModel.readPreTrain(embedFile, config.hiddenSize);

        }

        // create the adagrad cache
        pwgRNNGradientSquareMap = pwgRNNModel.createAdaGradSquare(config.adaAlpha);
        SURecursiveNetworkModel pairwiseEmptyGradients = pwgRNNModel.createGradients();

        // create the networks
        pairWiseGatedRNN = new SURecursiveNetwork(config.hiddenSize, pwgRNNModel);

        double bestDevF1 = Double.NEGATIVE_INFINITY;
        double loss;

        for (int iter = 0; iter < config.maxIter; iter++) {
            System.err.println("iteration : " + iter);
            loss = 0;

            BufferedReader goldReader = IOUtils.readerFromString(trainFileGold);
            BufferedReader kbestReader = IOUtils.readerFromString(trainFileKBest);

            while (true) {
                DataSet oneTrainingDataSet = genDataOnline(goldReader, kbestReader, config.onlieOnceReadNum);
                boolean lastRead = oneTrainingDataSet.size < config.onlieOnceReadNum ? true : false;

                List<Example> examples = oneTrainingDataSet.genTrainingExamples();
                Collections.shuffle(examples);

                int numBatches = examples.size() / config.batchSize + 1;
                int oneReadSize = examples.size();


                //test on the dev set
                Timing testTime = new Timing();
                testTime.doing("Dev trees");
                double F1 = 0;
                try {
                    F1 = test(devData);
                } catch (Exception e) {
                    e.printStackTrace();
                }

                if (F1 > bestDevF1) {
                    bestDevF1 = F1;
                    System.err.println("New F1 : " + F1);
                    System.err.println("New best F1 : " + bestDevF1);
                    SURecursiveNetworkModel.writeModel("PWGRNN." + modelFile, pwgRNNModel);
                } else {
                    System.err.println("New F1 : " + F1);
                }
                testTime.done();

                // training batch by batch
                for (int nBatchIndex = 0; nBatchIndex < numBatches; nBatchIndex++) {

                    System.err.print("The " + nBatchIndex + " ");
                    int start = nBatchIndex * config.batchSize;
                    int end = (nBatchIndex + 1) * config.batchSize;
                    if (end + config.batchSize > oneReadSize) {
                        end = oneReadSize;
                    }

                    // one batch training
                    double l = batchTraining(examples.subList(start, end), null, pairwiseEmptyGradients);

                    loss += l;

                    pairWiseGatedRNN.updateGradients(pairwiseEmptyGradients, pwgRNNGradientSquareMap, config.batchSize, config.regParameter);

                }  // end #for batch


                if (lastRead) {
                    break;
                }
            }

            System.err.println("iteration cost : " + loss);

            // close reader
            IOUtils.closeIgnoringExceptions(goldReader);
            IOUtils.closeIgnoringExceptions(kbestReader);

        } // end for iteration

    }

    class BiDirectionPWGRankProcessor implements ThreadsafeProcessor<Example,
            Triple<Boolean, Double, Gradients>> {
        @Override
        public Triple<Boolean, Double, Gradients> process(Example example) {

            double loss = 0;
            CFGRerankingTree bestTree = example.bestF1ScoreTree;
            List<CFGRerankingTree> kbestTrees = example.kbestTrees;


            Triple<CFGRerankingTree, Pair<CFGRerankingTree, CFGRerankingTree>, Double> biDirectionPWGHighestScoreTree
                    = getBiDirectionPWGHighestScoreTree(example, true);


            SURecursiveNetworkModel g3 = new SURecursiveNetworkModel(config.hiddenSize);
            g3.setVocabulary(pairWiseGatedRNN.model.vocabulary);
            SURecursiveNetworkModel g4 = new SURecursiveNetworkModel(config.hiddenSize);
            g4.setVocabulary(pairWiseGatedRNN.model.vocabulary);


            if (biDirectionPWGHighestScoreTree.second != null) {
                loss += biDirectionPWGHighestScoreTree.third;

                    pairWiseGatedRNN.pairwiseBackProp(biDirectionPWGHighestScoreTree.second.first, biDirectionPWGHighestScoreTree.second.second, g3, g4);
            }

            if (biDirectionPWGHighestScoreTree.second == null)
                return Triple.makeTriple(false, loss, null);
            else
                return Triple.makeTriple(true, loss, new Gradients(null, null, g3, g4));

        }

        @Override
        public ThreadsafeProcessor<Example,
                Triple<Boolean, Double, Gradients>> newInstance() {
            // should be threadsafe
            return this;
        }
    }



    private double batchTraining(List<Example> examples, SURecursiveNetworkModel emptyGradient,
                                 SURecursiveNetworkModel pairWiseEmptyGradient) {

        double loss = 0; // loss for a batch

        MulticoreWrapper<Example, Triple<Boolean, Double, Gradients>> wrapper =
                new MulticoreWrapper<>(
                        config.trainingThreads,
                        new BiDirectionPWGRankProcessor());

        for (Example example : examples) {
            wrapper.put(example);
        } // end for one batch

        wrapper.join();

        while (wrapper.peek()) {
            Triple<Boolean, Double, Gradients> updates = wrapper.poll();

            if (updates.first) {
                pairWiseEmptyGradient.merge(updates.third.g3, -1.0);
                pairWiseEmptyGradient.merge(updates.third.g4, 1.0);
                loss += updates.second;
            }
//            else
//                System.err.println("pass");
        }


        System.err.println("batch loss : " + loss);
        return loss;
    }

    public void test(String testFileKBest, String testFileGold) {

        DataSet testData = genTestData(testFileGold, testFileKBest, true);
        double F1 = 0;
        try {
            F1 = test(testData);
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.err.println("F1 : " + F1);

    }

    public double test(DataSet testDataSet) throws Exception {

        double retval;

        pairWiseGatedRNN.model.setBeTrain(false);

        List<Example> examples = testDataSet.genTrainingExamples();

        List<CFGRerankingTree> result = new ArrayList<>();

        for (int inst = 0; inst < testDataSet.size; inst++) {


            // get the kbest candidates in the first round reranking
            List<RerankCandidate> candidates = new ArrayList<>();
            Example example = examples.get(inst);

            Triple<CFGRerankingTree, Pair<CFGRerankingTree, CFGRerankingTree>, Double> biDirectionPWGHighestScoreTree =
                    getBiDirectionPWGHighestScoreTree(example, false);

            result.add(biDirectionPWGHighestScoreTree.first);
        }

        retval = evalb.evaluate(result, testDataSet.goldTree);

        pairWiseGatedRNN.model.setBeTrain(true);
        return retval;
    }


    /**
     * Insert all the paras from the training data into the
     * Recursive Model
     * <p>
     * #NOTE we haven't use this function in current train.
     *
     * @param trainData
     * @return
     */
//    private SURecursiveNetworkModel generateSURNNModel(DataSet trainData) {
//
//        SURecursiveNetworkModel retval = new SURecursiveNetworkModel(config.hiddenSize);
//        /*
//         * Insert all the paras in the recursive model
//         */
//        // tranverse all trees
//        for (int treeIdx = 0; treeIdx < trainData.getSize(); treeIdx++) {
//            Tree goldTree = trainData.goldTree.get(treeIdx);
//            List<CFGRerankingTree> kBestTrees = trainData.kbestTrees.get(treeIdx);
//            retval.insertModelsFromOneTree(goldTree);
//            for (Tree kBestTree : kBestTrees)
//                retval.insertModelsFromOneTree(kBestTree);
//        }
//
//        return retval;
//
//    }

    /**
     * Insert all the paras from the training data into the
     * Recursive Model
     */
    private SURecursiveNetworkModel generateSURNNModelOnline(String goldTreesFile, String kbestTreesFile) {

        SURecursiveNetworkModel retval = new SURecursiveNetworkModel(config.hiddenSize);

        List<String> knowWords = new ArrayList<>();

        BufferedReader goldReader = null;
        BufferedReader kbestReader = null;
        try {
            goldReader = IOUtils.readerFromString(goldTreesFile);
            kbestReader = IOUtils.readerFromString(kbestTreesFile);

            System.err.println("Begin to generate the surnn model online");

            /*
             * import the gold trees
             */
            int totalGoldNum = 0;
            for (String line : IOUtils.getLineIterable(goldReader, false)) {

                if ((line.trim().length() == 0)) { // skip the empty line

                    continue;
                }

                CFGTree tree = new CFGTree();
                tree.CTBReadNote(getTokens(line));
                List<CFGTreeNode> sentFromTree = tree.getSentFromTree();
                totalGoldNum++;

                for (CFGTreeNode node : sentFromTree)
                    knowWords.add(node.word.toLowerCase());
            }

            /*
             * filter the known word list,
             * and generate vocabulary
             */
            knowWords = Util.generateDict(knowWords, config.wordCutOff);
            HashSet<String> vocabulary = new HashSet<>(knowWords);
            retval.setVocabulary(vocabulary);
            retval.insertUNKWord(Config.UNKNOWN, true);

            IOUtils.closeIgnoringExceptions(goldReader);
            goldReader = IOUtils.readerFromString(goldTreesFile);
            for (String line : IOUtils.getLineIterable(goldReader, false)) {

                if ((line.trim().length() == 0)) { // skip the empty line
                    continue;
                }

                CFGTree tree = new CFGTree();
                tree.CTBReadNote(getTokens(line));

                CFGRerankingTree goldCFGRerankingTree = CFGRerankingTree.CFGTree2RerankingTree(tree);

                retval.insertModelsFromOneTree(goldCFGRerankingTree);
            }


            /*
             * inport the k-best trees
             */
            int totalKBestNum = 0;
            System.err.println("Begin to generate k-best trees!");
            List<CFGRerankingTree> kbestOneSent = new ArrayList<>();
            for (String line : IOUtils.getLineIterable(kbestReader, false)) {

                if (line.contains("WSJ") && !(line.trim().charAt(0) == '('))
                    continue;

                if (line.trim().length() == 0) { // if empty line?

//                    if (kbestOneSent.size() == 0) {
//                        System.err.println("kbest num == 0!");
//
//                        exit(0);
//                    }
                    totalKBestNum++;

                    kbestOneSent = new ArrayList<>();
                    continue;
                }

                if (!(line.trim().charAt(0) == '(')) {
                    continue;
                } else {

                /*
                 * get the scores and trees for kbest
                 */
                    CFGTree tree = new CFGTree();
                    tree.CTBReadNote(getTokens(line.trim()));
                    CFGRerankingTree cfgRerankingTree = CFGRerankingTree.CFGTree2RerankingTree(tree);
                    kbestOneSent.add(cfgRerankingTree);
                    retval.insertModelsFromOneTree(cfgRerankingTree);
                    continue;

                }


            }

            if (totalGoldNum != totalKBestNum) {
                System.err.println("gold trees and kbest trees num do not equal! gold: " + totalGoldNum + "kbest: " + totalKBestNum);
            }

        } catch (IOException e) {
            throw new RuntimeIOException(e);
        } finally {
            IOUtils.closeIgnoringExceptions(goldReader);
            IOUtils.closeIgnoringExceptions(kbestReader);
        }

        return retval;


    }
}
