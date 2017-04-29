package reranker;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.concurrent.MulticoreWrapper;
import edu.stanford.nlp.util.concurrent.ThreadsafeProcessor;
import nncon.CFGTree;
import nncon.CFGTreeNode;
import nndep.Util;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdaGrad;
import recursivenetwork.simplerecursive.SURecursiveNetwork;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.*;

import static java.lang.System.exit;
import static nncon.CFGTree.getTokens;

/**
 * Created by zhouh on 15-12-12.
 *
 */
public class SURNNReranker {

    SURecursiveNetwork surnn;

    Config config;

    Evaluator evalb;

    IdentityHashMap<INDArray, AdaGrad> gradientSquareMap; // map from neural network paras to the gradient square history

    public SURNNReranker(Properties properties) {

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

            if (line.contains("WSJ") && !(line.trim().charAt(0) == '(')){

                wsjStr = line;
                continue;
            }

            if (line.trim().length() == 0) { // if empty line?

                if(treesOfOneSent.size() == 0){
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

        if(config.bUseRankingRNN) {
            generateF1Score(retval);
            System.err.println("gen F1 score!");
        }

        return retval;
    }

    public void generateF1Score(DataSet data){

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

                    if(f1 > bestF1Score){
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

    public DataSet genTestData(String goldTreesFile, String kbestTreesFile, boolean bStoreCFGTree) {

        List<CFGRerankingTree> goldTree;
        List<List<CFGRerankingTree>> kbestTrees;
        List<List<Double>> kbestTreesBaseModelScore;
        List<List<Double>> kbestTreesF1Score;

        goldTree = new ArrayList<>();
        kbestTrees = new ArrayList<>();
        kbestTreesBaseModelScore = new ArrayList<>();
        kbestTreesF1Score = new ArrayList<>();

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
            for (String line : IOUtils.getLineIterable(kbestReader, false)) {

                if (line.contains("WSJ") &&  !(line.trim().charAt(0) == '('))
                    continue;

                if (line.trim().length() == 0) { // if empty line?
                    kbestTrees.add(treesOfOneSent);
                    kbestTreesBaseModelScore.add(baseTreeScoresOfOneSent);
                    treesOfOneSent = new ArrayList<>();
                    baseTreeScoresOfOneSent = new ArrayList<>();
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

        if(config.bUseRankingRNN) {
            generateF1Score(retval);
            System.err.println("gen F1 score!");
        }

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

        System.out.println("why do not stop? 1");
        Properties props = StringUtils.argsToProperties(args, numArgs);

        SURNNReranker reranker = new SURNNReranker(props);

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
            SURecursiveNetworkModel model = SURecursiveNetworkModel.readModel(props.getProperty("model"));
            model.setBeTrain(false);
            reranker.surnn = new SURecursiveNetwork(reranker.config.hiddenSize, model);

            reranker.test(props.getProperty("testFile") + ".kbest",
                    props.getProperty("testFile") + ".gold");
        }


    }

    public Pair<CFGRerankingTree, Double> getHighestScoredTree(Example example, boolean bAddBaseScore) {

        CFGRerankingTree bestTree = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for(int i = 0; i < example.kbestTrees.size(); i++){
            CFGRerankingTree tree = example.kbestTrees.get(i);
            double score = surnn.getScore(tree);
            if (bAddBaseScore)
                score += example.kbestTreesBaseModelScore.get(i);
            if (score > bestScore) {
//                System.err.println(score);
                bestTree = tree;
                bestScore = score;
            }
        }

        return new Pair<>(bestTree, bestScore);
    }

    public void rankTrain(String trainFileKBest, String trainFileGold, String devFileKBest, String devFileGold,
                          String modelFile, String embedFile)throws IOException {

        // output the paras info
        System.err.println("Rank Training Progress:");
        System.err.println("########## Config Info #############");
        System.err.println("Train File: " + trainFileKBest + " " + trainFileGold);
        System.err.println("Dev File: " + devFileKBest + " " + devFileGold);
        System.err.println("Model File: " + modelFile);
        System.err.println("Embedding File: " + embedFile);


        // generate the training and dev data
        System.out.println("###Begin to gen dev data!");
        DataSet devData = genTestData(devFileGold, devFileKBest, true);
        System.out.println("###Finish to gen dev data!");

        System.err.println("\n\n");
        // add the word embedding and matrix into the model
        System.err.println("###Begin to Generate Model Online!");
        SURecursiveNetworkModel surnnModel = generateSURNNModelOnline(trainFileGold, trainFileKBest);

        if (embedFile != null)
            surnnModel.readPreTrain(embedFile, config.hiddenSize);

        gradientSquareMap = surnnModel.createAdaGradSquare(config.adaAlpha);
        SURecursiveNetworkModel emptyGradients = surnnModel.createGradients();
        surnn = new SURecursiveNetwork(config.hiddenSize, surnnModel);

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

                List<Example> oneReadExamples = oneTrainingDataSet.genTrainingExamples();
                Collections.shuffle(oneReadExamples);
                int numBatches = oneReadExamples.size() / config.batchSize + 1;
                int oneReadSize = oneReadExamples.size();


                //test on the dev set
                Timing testTime = new Timing();
                testTime.doing("Dev trees");
                List<CFGRerankingTree> testResult = test(devData);
                double F1 = 0;
                try {
                    F1 = evalb.evaluate(
                            null,
                            devData.rerankingTrees2CFGTrees(testResult),
                            devData.goldCFGTrees);

                } catch (IOException e) {
                    e.printStackTrace();
                    exit(0);
                }
                if (F1 > bestDevF1) {
                    bestDevF1 = F1;
                    System.err.println("New F1 : " + F1);
                    System.err.println("New best F1 : " + bestDevF1);
                    SURecursiveNetworkModel.writeModel(modelFile, surnnModel);
                } else {
                    System.err.println("New F1 : " + F1);
                }
                testTime.done();

                for (int nBatchIndex = 0; nBatchIndex < numBatches; nBatchIndex++) {

                    System.err.print("The " + nBatchIndex + " ");
                    int start = nBatchIndex * config.batchSize;
                    int end = (nBatchIndex + 1) * config.batchSize;
                    if (end + config.batchSize > oneReadSize) {
                        end = oneReadSize;
                    }

//                System.err.println("nBatchIndex: " + nBatchIndex);

                    double l = batchTraining(oneReadExamples.subList(start, end), emptyGradients);

                    loss += l;

                    surnn.updateGradients(emptyGradients, gradientSquareMap, config.batchSize, config.regParameter);

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
        SURecursiveNetworkModel surnnModel = generateSURNNModelOnline(trainFileGold, trainFileKBest);

        if (embedFile != null)
            surnnModel.readPreTrain(embedFile, config.hiddenSize);

//        if (config.bCheckUnknowRule) {
//
//            for (int i = 0; i < devData.size; i++) {
//                if(devData.goldTree.get(i).numChildren() == 2){
//
//                    surnnModel.getBinaryScoreLayer(devData.goldTree.get(i));
//                    surnnModel.getBinaryTransform(devData.goldTree.get(i));
//                }
//                else if(devData.goldTree.get(i).numChildren() == 1){
//
//                    surnnModel.getUnaryTransform(devData.goldTree.get(i));
//                    surnnModel.getUnaryScoreLayer(devData.goldTree.get(i));
//                }
//
//                for (Tree tree : devData.kbestTrees.get(i)) {
//                    if(tree.numChildren() == 2){
//                        surnnModel.getBinaryScoreLayer(tree);
//                        surnnModel.getBinaryTransform(tree);
//
//                    }
//                    else if(tree.numChildren() == 1){
//
//                        surnnModel.getUnaryTransform(tree);
//                        surnnModel.getUnaryScoreLayer(tree);
//                    }
//
//                }
//            }
//
//        }
        gradientSquareMap = surnnModel.createAdaGradSquare(config.adaAlpha);
        SURecursiveNetworkModel emptyGradients = surnnModel.createGradients();
        surnn = new SURecursiveNetwork(config.hiddenSize, surnnModel);

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

                List<Example> oneReadExamples = oneTrainingDataSet.genTrainingExamples();
                Collections.shuffle(oneReadExamples);
                int numBatches = oneReadExamples.size() / config.batchSize + 1;
                int oneReadSize = oneReadExamples.size();


                //test on the dev set
                Timing testTime = new Timing();
                testTime.doing("Dev trees");
                List<CFGRerankingTree> testResult = test(devData);
                double F1 = 0;
                try {
                    F1 = evalb.evaluate(
                            testResult,
                            devData.goldTree);

                } catch (IOException e) {
                    e.printStackTrace();
                    exit(0);
                }
                if (F1 > bestDevF1) {
                    bestDevF1 = F1;
                    System.err.println("New F1 : " + F1);
                    System.err.println("New best F1 : " + bestDevF1);
                    SURecursiveNetworkModel.writeModel(modelFile, surnnModel);
                } else {
                    System.err.println("New F1 : " + F1);
                }
                testTime.done();

                for (int nBatchIndex = 0; nBatchIndex < numBatches; nBatchIndex++) {

                    System.err.print("The " + nBatchIndex + " ");
                    int start = nBatchIndex * config.batchSize;
                    int end = (nBatchIndex + 1) * config.batchSize;
                    if (end + config.batchSize > oneReadSize) {
                        end = oneReadSize;
                    }

//                System.err.println("nBatchIndex: " + nBatchIndex);

                    double l = batchTraining(oneReadExamples.subList(start, end), emptyGradients);

                    loss += l;

                    surnn.updateGradients(emptyGradients, gradientSquareMap, config.batchSize, config.regParameter);

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

    class ScoringProcessor implements ThreadsafeProcessor<Example,
            Triple<Double, SURecursiveNetworkModel, SURecursiveNetworkModel>> {
        @Override
        public Triple<Double, SURecursiveNetworkModel, SURecursiveNetworkModel> process(Example example) {
            // For each tree, move in the direction of the gold tree, and
            // move away from the direction of the best scoring hypothesis

            double loss = 0;
            CFGRerankingTree goldTree = example.goldTree;

            Pair<CFGRerankingTree, Double> predictScoredTree = getHighestScoredTree(example, false);  // in training, do not add base score

            double goldScore = surnn.getScore(goldTree);

            /*
             * update the parameters
             */
//                if (Math.abs(predictScoredTree.second - goldScore) <= 0.00001
//                        || goldScore > predictScoredTree.second) {

//            boolean isDone = goldScore > predictScoredTree.second ||
//                    Math.abs(predictScoredTree.second - goldScore) <= 0.00001;

            boolean isDone = goldScore > predictScoredTree.second;

            if (!isDone) {
                // update parameters
                loss += goldScore - predictScoredTree.second;

                INDArray errorArray = Nd4j.zeros(config.hiddenSize, 1);

                SURecursiveNetworkModel goldGradients = new SURecursiveNetworkModel(config.hiddenSize);
                goldGradients.setVocabulary(surnn.model.vocabulary);
                surnn.backProp(goldTree, errorArray, goldGradients);
                SURecursiveNetworkModel predictGradients = new SURecursiveNetworkModel(config.hiddenSize);
                predictGradients.setVocabulary(surnn.model.vocabulary);

                if (predictScoredTree.first == null)
                    System.err.println(predictScoredTree.second + " tree is null!");
                surnn.backProp(predictScoredTree.first, errorArray, predictGradients);

                return Triple.makeTriple(loss, goldGradients, predictGradients);
            }

            return Triple.makeTriple(0.0, null, null);

        }

        @Override
        public ThreadsafeProcessor<Example,
                Triple<Double, SURecursiveNetworkModel, SURecursiveNetworkModel>> newInstance() {
            // should be threadsafe
            return this;
        }
    }

    class ScoringRankProcessor implements ThreadsafeProcessor<Example,
            Triple<Double, SURecursiveNetworkModel, SURecursiveNetworkModel>> {
        @Override
        public Triple<Double, SURecursiveNetworkModel, SURecursiveNetworkModel> process(Example example) {

            double loss = 0;
            CFGRerankingTree bestTree = example.bestF1ScoreTree;
            List<CFGRerankingTree> kbestTrees = example.kbestTrees;
            List<Double> kbestTreesBaseModelScore = example.kbestTreesBaseModelScore;

            Pair<CFGRerankingTree, Double> predictScoredTree = getHighestScoredTree(example, config.bRerankerAddBaseScore);  // in training, do not add base score

            /*
             * update the parameters
             */
//                if (Math.abs(predictScoredTree.second - goldScore) <= 0.00001
//                        || goldScore > predictScoredTree.second) {

//            boolean isDone = goldScore > predictScoredTree.second ||
//                    Math.abs(predictScoredTree.second - goldScore) <= 0.00001;

            boolean isDone = predictScoredTree.first == bestTree;

            if (!isDone) {
                // update parameters

                double bestTreeScore = surnn.getScore(bestTree);
                if(config.bRerankerAddBaseScore){
                    int i = 0;
                    for (; i < kbestTrees.size(); i++) {
                        if(kbestTrees.get(i) == bestTree){
                            bestTreeScore += kbestTreesBaseModelScore.get(i);

                            break;
                        }
                    }
                    if(kbestTrees.size() == i)
                        throw new RuntimeException("Not found the best tree in the k-best");

                }

                loss += bestTreeScore - predictScoredTree.second;

                INDArray errorArray = Nd4j.zeros(config.hiddenSize, 1);

                SURecursiveNetworkModel goldGradients = new SURecursiveNetworkModel(config.hiddenSize);
                goldGradients.setVocabulary(surnn.model.vocabulary);
                surnn.backProp(bestTree, errorArray, goldGradients);
                SURecursiveNetworkModel predictGradients = new SURecursiveNetworkModel(config.hiddenSize);
                predictGradients.setVocabulary(surnn.model.vocabulary);

                if (predictScoredTree.first == null)
                    System.err.println(predictScoredTree.second + " tree is null!");
                surnn.backProp(predictScoredTree.first, errorArray, predictGradients);

                return Triple.makeTriple(loss, goldGradients, predictGradients);
            }

            System.err.println("No Update");
            return Triple.makeTriple(0.0, null, null);

        }

        @Override
        public ThreadsafeProcessor<Example,
                Triple<Double, SURecursiveNetworkModel, SURecursiveNetworkModel>> newInstance() {
            // should be threadsafe
            return this;
        }
    }

    private double batchTraining(List<Example> examples, SURecursiveNetworkModel emptyGradient) {

        double loss = 0; // loss for a batch

        MulticoreWrapper<Example, Triple<Double, SURecursiveNetworkModel, SURecursiveNetworkModel>> wrapper =
                new MulticoreWrapper<>(
                        config.trainingThreads,
                       config.bUseRankingRNN ? new ScoringRankProcessor() :
                               new ScoringProcessor());

        for (Example example : examples) {
            if (example.kbestTrees.size() == 0)
                continue; // skip the 0 size kbest list
            wrapper.put(example);
        } // end for one batch

        wrapper.join();

        while (wrapper.peek()) {
            Triple<Double, SURecursiveNetworkModel, SURecursiveNetworkModel> updates = wrapper.poll();

            if (updates.second != null) {
                emptyGradient.merge(updates.second, -1.0);
                emptyGradient.merge(updates.third, 1.0);
                loss += updates.first;
            }
        }


        System.err.println("batch loss : " + loss);
        return loss;
    }

    public void test(String testFileKBest, String testFileGold) {

        DataSet testData = genTestData(testFileGold, testFileKBest, true);
        List<CFGRerankingTree> testResult = test(testData);

        try {
            double F1 = evalb.evaluate(
                    testResult,
                    testData.goldTree);
        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    public List<CFGRerankingTree> test(DataSet testDataSet) {

        surnn.model.setBeTrain(false);

        ArrayList<CFGRerankingTree> retval = new ArrayList<>();
        List<Example> examples = testDataSet.genTrainingExamples();

        for (int inst = 0; inst < testDataSet.size; inst++) {

//            System.err.print(inst + " ");
//            if(inst % 50 == 0)
//                System.err.println();


            // in test, we always add the base model score
            Pair<CFGRerankingTree, Double> predictScoredTree = getHighestScoredTree(examples.get(inst), true);

            retval.add(predictScoredTree.first);
        }

        surnn.model.setBeTrain(true);
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
