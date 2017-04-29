package reranker;

import nncon.CFGTree;

import java.util.List;

/**
 * Created by zhouh on 15-12-24.
 *
 */
public class Example {

    CFGRerankingTree goldTree;
    List<CFGRerankingTree> kbestTrees;
    List<Double> kbestTreesBaseModelScore;
    List<Double> kbestTreesF1Score;
    List<Double> kbestTreesCharniarkScore;
    CFGTree goldCFGTrees;
    CFGRerankingTree bestF1ScoreTree;

    public Example(CFGRerankingTree goldTree, List<CFGRerankingTree> kbestTrees, List<Double> kbestTreesF1Score,
                   List<Double> kbestTreesBaseModelScore, CFGTree goldCFGTrees) {
        this.goldTree = goldTree;
        this.kbestTrees = kbestTrees;
        this.kbestTreesF1Score = kbestTreesF1Score;
        this.kbestTreesBaseModelScore = kbestTreesBaseModelScore;
        this.goldCFGTrees = goldCFGTrees;
    }

    public void setBestF1ScoreTree(CFGRerankingTree bestF1ScoreTree) {
        this.bestF1ScoreTree = bestF1ScoreTree;
    }

    public void setKbestTreesCharniarkScore(List<Double> kbestTreesCharniarkScore) {
        this.kbestTreesCharniarkScore = kbestTreesCharniarkScore;
    }
}
