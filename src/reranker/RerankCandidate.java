package reranker;

/**
 * Created by zhouh on 16-5-15.
 *
 */
public class RerankCandidate implements Comparable{

    CFGRerankingTree tree;
    double score;
    double F1Score;
    double oneVsAllScore;
    double charniarkScore;

    double surnnScore;

    public RerankCandidate(CFGRerankingTree tree, double score, double f1Score) {
        this.tree = tree;
        this.score = score;
        F1Score = f1Score;
    }

    public void setSurnnScore(double surnnScore) {
        this.surnnScore = surnnScore;
    }

    public void setCharniarkScore(double charniarkScore) {
        this.charniarkScore = charniarkScore;
    }

    public void setOneVsAllScore(double oneVsAllScore) {
        this.oneVsAllScore = oneVsAllScore;
    }

    @Override
    public int compareTo(Object o) {

        RerankCandidate s = (RerankCandidate)o;
        int retval = score > s.score ? -1 : (score == s.score ? 0 : 1);
        return retval;
    }
}
