package reranker;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by zhouh on 16-2-17.
 *
 */
public class TreeLattice {

    List<CFGRerankingTree> postOrderList;

    List<Integer> leafIndex;

    public TreeLattice(List<CFGRerankingTree> postOrderList) {
        this.postOrderList = postOrderList;

        // construct the leaf index array
        leafIndex = new ArrayList<>();
        int index = 0;
        for (CFGRerankingTree tree : postOrderList) {
            if (tree.numChildren() == 0)
                leafIndex.add(index++);
        }

        computeSpanIndex();
    }

    public void computeSpanIndex(){

        // set the leaf node span index
        int index = 0;
        for (int i = 0; i <  postOrderList.size(); i++) {
            CFGRerankingTree tree = postOrderList.get(i);
            if(tree.numChildren() == 0){
                tree.begin = index;

                tree.end = index++;
            }
            else{
                tree.begin = tree.children.getFirst().begin;
                tree.end = tree.children.getLast().end;
            }
        }
    }

    public List<CFGRerankingTree> getSpanTrees(int i, int j) {

        List<CFGRerankingTree> retval = new ArrayList<>();

        int beginIndex = leafIndex.get(i);

        int endIndex = (j+1) >= leafIndex.size() ? (leafIndex.size() - 1) : (leafIndex.get(j+1) - 1);

        int currentBegin = -1;
        int currentEnd = -1;
        for(int p = endIndex; p >= beginIndex; p--){

            CFGRerankingTree currentTree = postOrderList.get(p);
            if(currentTree.begin >= i && currentTree.end <= j
                    && (currentBegin == -1 || currentTree.begin < currentBegin || currentTree.end > currentEnd )){

                retval.add(currentTree);
                currentBegin = currentBegin == -1 ? currentTree.begin : Math.min(currentBegin, currentTree.begin);
                currentEnd = Math.max(currentEnd, currentTree.end);
            }

            if (currentBegin == i && currentEnd == j)
                break;

        }

        return retval;

    }
}
