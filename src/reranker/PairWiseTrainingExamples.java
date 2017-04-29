package reranker;

/**
 * Created by zhouh on 16-5-13.
 *
 * example object for pair-wise training
 */
public class PairWiseTrainingExamples {

    CFGRerankingTree tree1;
    CFGRerankingTree tree2;

    double tree1FScore;
    double tree2FScore;

    public PairWiseTrainingExamples(CFGRerankingTree tree1, CFGRerankingTree tree2, double tree1FScore, double tree2FScore) {
        this.tree1 = tree1;
        this.tree2 = tree2;
        this.tree1FScore = tree1FScore;
        this.tree2FScore = tree2FScore;
    }
}
