package reranker;

import nncon.CFGTree;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;

/**
 * Created by zhouh on 15-12-15.
 */
public class DataSet {
    List<CFGRerankingTree> goldTree;
    List<List<CFGRerankingTree>> kbestTrees;
    List<List<Double>> kbestTreesBaseModelScore;
    List<List<Double>> kbestTreesF1Score;
    List<List<Double>> kbestTreesCharniarkScore;
    List<CFGTree> goldCFGTrees = new ArrayList<>();
    List<CFGRerankingTree> bestF1ScoreTrees = new ArrayList<>();
    IdentityHashMap<CFGRerankingTree, CFGTree> tree2CFGTreeIdentityHashMap = new IdentityHashMap<>();
    int size;

    public DataSet(List<CFGRerankingTree> goldTree,
                   List<List<CFGRerankingTree>> kbestTrees,
                   List<List<Double>> kbestTreesBaseModelScore,
                   List<List<Double>> kbestTreesF1Score) {

        this.goldTree = goldTree;
        this.kbestTrees = kbestTrees;
        this.kbestTreesBaseModelScore = kbestTreesBaseModelScore;
        this.kbestTreesF1Score = kbestTreesF1Score;


    }

    public void setBestF1ScoreTrees(List<CFGRerankingTree> bestF1ScoreTrees) {
        this.bestF1ScoreTrees = bestF1ScoreTrees;
    }

    public void setKbestTreesF1Score(List<List<Double>> kbestTreesF1Score) {
        this.kbestTreesF1Score = kbestTreesF1Score;
    }

    public List<CFGTree> rerankingTrees2CFGTrees(List<CFGRerankingTree> testResult) {

        List<CFGTree> testResultCFGTree = new ArrayList<>();
        for (CFGRerankingTree CFGRerankingTree : testResult) {
            testResultCFGTree.add(tree2CFGTreeIdentityHashMap.get(CFGRerankingTree));
        }
        return testResultCFGTree;
    }

    public List<Example> genTrainingExamples() {
        List<Example> retval = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            Example example = new Example(
                    goldTree.get(i),
                    kbestTrees.get(i),
                    kbestTreesF1Score.get(i),
                    kbestTreesBaseModelScore.get(i), // #TODO add the kbest f1 score!
                    null);
            if (bestF1ScoreTrees.size() != 0)
                example.setBestF1ScoreTree(bestF1ScoreTrees.get(i));
            retval.add(example);

        }

        return retval;
    }

    public void setKbestTreesCharniarkScore(List<List<Double>> kbestTreesCharniarkScore) {
        this.kbestTreesCharniarkScore = kbestTreesCharniarkScore;
    }

    public int getSize() {
        return size;
    }

    /**
     * construct the pair wise training examples
     * @return
     */
    public List<PairWiseTrainingExamples> genPairwiseTrainingExamples(){

        List<PairWiseTrainingExamples> retval = new ArrayList<>();

        // generate pair wise training examples
        // we would like to rank all highest scored candidates better
        // than other all
        for (int i = 0; i < size; i++) {

            double bestTreeF1Score = 0;
            List<CFGRerankingTree> kbest = kbestTrees.get(i);
            CFGRerankingTree bestTree = bestF1ScoreTrees.get(i);
                List<PairWiseTrainingExamples> examplesFromOneKbest = new ArrayList<>();

            for (int j = 0; j < kbest.size(); j++) {

                CFGRerankingTree tree = kbest.get(j);

                if (tree == bestTree)
                    bestTreeF1Score = kbestTreesF1Score.get(i).get(j);

            }

            for (int j = 0; j < kbest.size(); j++) {

                CFGRerankingTree tree = kbest.get(j);

                if (tree != bestTree) {

                    examplesFromOneKbest.add(new PairWiseTrainingExamples(bestTree, tree, bestTreeF1Score, kbestTreesF1Score.get(i).get(j)));

                }
            }

            retval.addAll(examplesFromOneKbest);

        }


        return retval;
    }


}
