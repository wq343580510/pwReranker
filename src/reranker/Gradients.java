package reranker;


/**
 * Created by zhouh on 16-2-18.
 */
public class Gradients {
    SURecursiveNetworkModel g1;
    reranker.SURecursiveNetworkModel g2;
    SURecursiveNetworkModel g3;
    SURecursiveNetworkModel g4;

    public Gradients(SURecursiveNetworkModel g1, SURecursiveNetworkModel g2, SURecursiveNetworkModel g3, SURecursiveNetworkModel g4) {
        this.g1 = g1;
        this.g2 = g2;
        this.g3 = g3;
        this.g4 = g4;
    }
}
